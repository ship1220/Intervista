# ai_service.py — Robust async AI service with Ollama → Gemini fallback

import os
import re
import json
import time
import math
import asyncio
import statistics
import httpx
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional, Dict, Any

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Configuration
OLLAMA_TIMEOUT_SECONDS = 120    # per-chunk timeout (with streaming, only triggers if model hangs)
GEMINI_TIMEOUT_SECONDS = 30
MAX_RETRIES = 1                 # single attempt — streaming eliminates timeout-based retries
RETRY_DELAY_SECONDS = 0.5
CACHE_TTL_SECONDS = 300
MAX_CACHE_SIZE = 200
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Startup diagnostics
if not GEMINI_API_KEY:
    print("[ai_service] WARNING: GEMINI_API_KEY not set — Gemini fallback disabled. "
          "Ollama will be the only backend. Set GEMINI_API_KEY in .env for a cloud fallback.")
print(f"[ai_service] Config: model={OLLAMA_MODEL}, timeout={OLLAMA_TIMEOUT_SECONDS}s, "
      f"retries={MAX_RETRIES}, gemini={'enabled' if GEMINI_API_KEY else 'disabled'}")

# ---------------------------------------------------------------------------
# TTL Cache
# ---------------------------------------------------------------------------
_cache: Dict[str, Dict] = {}


def _cache_get(key: str) -> Optional[str]:
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        return entry["value"]
    _cache.pop(key, None)
    return None


def _cache_set(key: str, value: str) -> None:
    if len(_cache) > MAX_CACHE_SIZE:
        oldest = min(_cache, key=lambda k: _cache[k]["ts"])
        _cache.pop(oldest, None)
    _cache[key] = {"value": value, "ts": time.time()}


# ---------------------------------------------------------------------------
# Timeout configuration
# ---------------------------------------------------------------------------
_OLLAMA_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=OLLAMA_TIMEOUT_SECONDS,
    write=10.0,
    pool=5.0,
)

_GEMINI_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=GEMINI_TIMEOUT_SECONDS,
    write=10.0,
    pool=5.0,
)

# ---------------------------------------------------------------------------
# Reusable httpx client
# ---------------------------------------------------------------------------
_ollama_client: Optional[httpx.AsyncClient] = None


def _get_ollama_client() -> httpx.AsyncClient:
    global _ollama_client
    if _ollama_client is None or _ollama_client.is_closed:
        _ollama_client = httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT)
    return _ollama_client


# ---------------------------------------------------------------------------
# JSON Extraction Helper
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Any:
    """Parse JSON from text, handling markdown fences and common AI mistakes."""
    if not text or not text.strip():
        raise ValueError("Empty text — no JSON to extract")

    cleaned = text.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Find outermost JSON object or array
        first_brace = None
        for i, ch in enumerate(cleaned):
            if ch in ('{', '['):
                first_brace = i
                break
        if first_brace is not None:
            closer = '}' if cleaned[first_brace] == '{' else ']'
            last_brace = cleaned.rfind(closer)
            if last_brace > first_brace:
                cleaned = cleaned[first_brace:last_brace + 1]

    # Attempt 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: strip control characters
    fixed = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 4: fix unescaped newlines inside strings
    fixed = re.sub(r'(?<=[\":])((?:[^"\\\\]|\\\\.)*)(?=[":])',
                   lambda m: m.group(0).replace('\n', '\\n'), fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 5: repair truncated JSON by closing unclosed brackets/braces
    repaired = fixed.rstrip().rstrip(',')
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_braces > 0 or open_brackets > 0:
        repaired += ']' * max(open_brackets, 0)
        repaired += '}' * max(open_braces, 0)
        try:
            result = json.loads(repaired)
            print(f"[extract_json] Recovered truncated JSON (closed {open_braces} braces, {open_brackets} brackets)")
            return result
        except json.JSONDecodeError:
            pass

    print(f"[extract_json] All parse attempts failed")
    print(f"[extract_json] Raw text ({len(text)} chars): {text[:500]}")
    raise ValueError(f"Could not extract valid JSON from text ({len(text)} chars)")


# ---------------------------------------------------------------------------
# Response Validator
# ---------------------------------------------------------------------------
def validate_response(result: Dict, required_keys: list = None) -> bool:
    """Validate that the response contains expected structure."""
    if not isinstance(result, dict):
        return False
    
    if required_keys is None:
        required_keys = ["evaluations"]
    
    for key in required_keys:
        if key not in result:
            return False
    
    # Validate evaluations list exists
    if not isinstance(result.get("evaluations"), list):
        return False
    
    # Validate each evaluation object
    for answer in result.get("evaluations", []):
        if not isinstance(answer, dict):
            return False
    
        required_fields = ["score", "feedback", "strengths", "weaknesses", "ideal_answer"]
        
        for field in required_fields:
            if field not in answer:
                return False
    
    return True


# ---------------------------------------------------------------------------
# Default Fallback Response
# ---------------------------------------------------------------------------
def get_fallback_response(num_questions: int = 5) -> Dict:
    """Return a valid fallback response when AI fails."""
    return {
        "answers": [
            {
                "score": 50,
                "feedback": (
                    "AI evaluation service temporarily unavailable. "
                    "Your answer was recorded. Please retry for detailed feedback."
                ),
                "strengths": ["Attempted the question."],
                "weaknesses": ["Evaluation unavailable due to AI service issue."],
                "ideal_answer": "Ideal answer could not be generated.",
            }
            for _ in range(num_questions)
        ],
        "overall_feedback": "Evaluation service temporarily unavailable. Please retry the interview.",
        "aggregate": {
            "relevance_score": 50,
            "depth_score": 50,
            "star_method_score": 50,
        },
    }


# ---------------------------------------------------------------------------
# Ollama Implementation
# ---------------------------------------------------------------------------
async def _try_ollama_with_retry(prompt: str, json_mode: bool = False) -> Optional[str]:
    """Try Ollama /api/chat with limited retry.

    Only uses /api/chat (not /api/generate) to avoid doubling latency.
    Fails fast so the Gemini fallback can be reached without excessive delay.
    """
    for attempt in range(MAX_RETRIES):
        try:
            text = await _ollama_chat(prompt, json_mode)
            if text:
                return text
            print(f"[ollama] attempt {attempt + 1}: empty response")
        except httpx.ReadTimeout:
            print(f"[ollama] attempt {attempt + 1}: timeout ({OLLAMA_TIMEOUT_SECONDS}s)")
        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionRefusedError, OSError):
            print(f"[ollama] Ollama not reachable at {OLLAMA_URL} — skipping retries")
            return None
        except Exception as e:
            print(f"[ollama] attempt {attempt + 1}: {type(e).__name__}: {e}")

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY_SECONDS)

    return None


async def _ollama_chat(prompt: str, json_mode: bool = False) -> Optional[str]:
    """Call Ollama /api/chat with streaming.

    Streaming is critical: with stream=False, Ollama buffers the entire
    response before sending anything.  If generation takes longer than the
    httpx read-timeout the request is killed even though Ollama is working.
    With stream=True each token resets the read-timeout, so total generation
    time is effectively unlimited.
    """
    try:
        client = _get_ollama_client()
        system_msg = (
            "Respond with valid JSON only. No markdown fences, no extra text."
        ) if json_mode else "You are a helpful assistant."

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "options": {
                "num_predict": 2048,
                "temperature": 0.3 if json_mode else 0.7,
                "top_p": 0.9,
            },
        }
        if json_mode:
            payload["format"] = "json"

        start_time = time.time()
        chunks: list[str] = []

        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                print(f"[ollama] /api/chat HTTP {resp.status_code}: {body[:200]}")
                return None

            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        chunks.append(token)
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

        text = "".join(chunks).strip()
        latency = time.time() - start_time
        print(f"[ollama] /api/chat stream done: {latency:.1f}s, {len(chunks)} chunks, {len(text)} chars  json_mode={json_mode}")

        if text:
            return text
        print("[ollama] /api/chat streaming completed but empty content")
    except httpx.ReadTimeout:
        print(f"[ollama] /api/chat ReadTimeout ({OLLAMA_TIMEOUT_SECONDS}s between chunks — model may be hung)")
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionRefusedError, OSError):
        raise  # Let _try_ollama_with_retry handle connection failures
    except Exception as e:
        print(f"[ollama] /api/chat error: {type(e).__name__}: {e}")
    return None



# ---------------------------------------------------------------------------
# Gemini Fallback — fixed model name + SDK compatibility
# ---------------------------------------------------------------------------
async def _try_gemini(prompt: str) -> Optional[str]:
    """Run Gemini as fallback. Single attempt to minimise total latency."""
    if not GEMINI_API_KEY:
        print("[gemini] No GEMINI_API_KEY configured")
        return None

    try:
        result = await _gemini_async(prompt)
        if result:
            print(f"[gemini] Success ({len(result)} chars)")
            return result
        print("[gemini] Empty response")
    except Exception as e:
        print(f"[gemini] Error: {type(e).__name__}: {e}")

    return None


async def _gemini_async(prompt: str) -> str:
    """Execute Gemini call asynchronously.

    Tries the new Client API first (google-generativeai >= 0.8 / google-genai),
    then falls back to the legacy GenerativeModel API.
    """
    def _call_gemini():
        start_time = time.time()
        try:
            # New SDK style (google-generativeai >= 0.8 or google-genai package)
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        except (AttributeError, TypeError):
            # Old SDK style (google-generativeai < 0.8)
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)

        latency = time.time() - start_time
        print(f"[gemini] latency: {latency:.2f}s  model={GEMINI_MODEL}")
        return response.text.strip()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _call_gemini)


# ---------------------------------------------------------------------------
# Main Content Generation
# ---------------------------------------------------------------------------
async def generate_content(
    prompt: str,
    *,
    use_cache: bool = True,
    json_mode: bool = False
) -> str:
    """Send prompt to AI and return text response with fallback."""
    
    # Check cache
    if use_cache:
        cached = _cache_get(prompt)
        if cached is not None:
            print("[ai_service] Returning cached response")
            return cached
    
    # Try Ollama first
    print(f"[ai_service] Trying Ollama (model: {OLLAMA_MODEL})...")
    result = await _try_ollama_with_retry(prompt, json_mode=json_mode)
    
    # Fallback to Gemini
    if not result:
        if GEMINI_API_KEY:
            print("[ai_service] Ollama failed, trying Gemini fallback...")
            result = await _try_gemini(prompt)
        else:
            print("[ai_service] Ollama failed. Gemini unavailable (no GEMINI_API_KEY).")
    
    # If all failed, return empty
    if not result:
        print(f"[ai_service] WARNING: All AI backends failed for prompt ({len(prompt)} chars): {prompt[:120]}...")
        return ""
    
    # Log preview
    preview = result[:300].replace("\n", " ")
    print(f"[ai_service] Response preview ({len(result)} chars): {preview}")
    
    # Cache the result
    if use_cache:
        _cache_set(prompt, result)
    
    return result


# ---------------------------------------------------------------------------
# Evaluate Content - Main function for interview evaluation
# ---------------------------------------------------------------------------
async def evaluate_content(
    role: str,
    level: str,
    questions_answers: list
) -> Dict:
    """Evaluate interview content and return structured analysis."""
    from prompts import interview_evaluation_prompt

    num_questions = len(questions_answers)
    prompt = interview_evaluation_prompt(role, questions_answers)

    raw = await generate_content(prompt, use_cache=False, json_mode=True)
    print(f"[evaluate_content] Raw response length: {len(raw) if raw else 0}")

    if raw and raw.strip():
        try:
            result = extract_json(raw)
            if isinstance(result, dict):
                normalized = _normalize_response(result, num_questions)
                print(f"[evaluate_content] Success: {len(normalized.get('answers', []))} answers")
                return normalized
            print(f"[evaluate_content] Unexpected type: {type(result).__name__}")
        except Exception as e:
            print(f"[evaluate_content] Parse error: {e}")
            print(f"[evaluate_content] Raw text preview: {raw[:500]}")

    print("[evaluate_content] Using fallback")
    return get_fallback_response(num_questions)


def _normalize_response(result: Dict, num_questions: int) -> Dict:
    """Normalize AI response to the format expected by main.py / report.html."""

    # Find answers from various possible keys
    answers = None
    for key in ["answers", "feedback_per_question", "evaluations", "results"]:
        if key in result and isinstance(result[key], list):
            answers = result[key]
            break

    if answers is None:
        answers = []

    # Normalize each answer entry
    normalized_answers = []
    
    for i in range(num_questions):
        if i < len(answers) and isinstance(answers[i], dict):
            ans = answers[i]
    
            normalized_answers.append({
                "score": ans.get("score", 50),
                "feedback": ans.get("feedback", ans.get("comment", "Feedback unavailable.")),
                "strengths": ans.get("strengths", []),
                "weaknesses": ans.get("weaknesses", []),
                "ideal_answer": ans.get("ideal_answer", ""),
            })

        else:
            normalized_answers.append({
                "score": 50,
                "feedback": "Feedback unavailable for this question.",
                "strengths": [],
                "weaknesses": [],
                "ideal_answer": "",
            })

    # Preserve the original aggregate dict and add defaults for any
    # keys that report.html needs but the AI may not have returned.
    aggregate = result.get("aggregate", result.get("scores", {}))
    if not isinstance(aggregate, dict):
        aggregate = {}

    avg_score = sum(a["score"] for a in normalized_answers) / max(len(normalized_answers), 1)
    aggregate.setdefault("relevance_score", round(avg_score))
    aggregate.setdefault("depth_score", round(avg_score * 0.9))
    aggregate.setdefault("star_method_score", round(avg_score * 0.7))

    overall_feedback = result.get("overall_feedback", result.get("summary", ""))
    if not overall_feedback:
        overall_feedback = "Evaluation completed. Review individual feedback above."

    return {
        "answers": normalized_answers,
        "overall_feedback": overall_feedback,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Speech Analysis
# ---------------------------------------------------------------------------
FILLER_WORDS_SINGLE: set = {
    "um", "uh", "like", "so", "well", "actually", "basically",
    "literally", "totally", "right", "anyway", "okay",
}
MULTI_WORD_FILLERS: list = ["you know", "i mean", "kind of", "sort of"]


def analyze_speech_delivery(answer: str, duration_seconds: float) -> dict:
    """Analyze speech delivery metrics."""
    if not answer or not answer.strip():
        return {
           "word_count": 0,
           "duration_seconds": round(duration_seconds, 1),
           "speaking_pace_wpm": 0.0,
           "filler_word_count": 0,
           "filler_words_found": [],
           "clarity_score": 0,
           "engagement_score": 0,
           "sentence_count": 0,
           "avg_words_per_sentence": 0.0,
        }

    words = answer.split()
    word_count = len(words)
    duration_minutes = max(duration_seconds / 60.0, 0.01)
    wpm = word_count / duration_minutes

    # Filler detection
    cleaned_words = [w.lower().strip(".,!?;:'\"") for w in words]
    single_hits = [w for w in cleaned_words if w in FILLER_WORDS_SINGLE]
    lower_text = answer.lower()
    multi_count = sum(lower_text.count(mf) for mf in MULTI_WORD_FILLERS)
    filler_count = len(single_hits) + multi_count
    filler_rate = filler_count / max(word_count, 1)

    # Sentence analysis
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_wps = word_count / sentence_count

    # Clarity score
    clarity = 80.0
    clarity -= filler_rate * 150
    if avg_wps > 30:
        clarity -= (avg_wps - 30) * 1.0
    elif avg_wps < 6:
        clarity -= (6 - avg_wps) * 3.0
    if word_count < 15:
        clarity -= 25
    elif word_count < 30:
        clarity -= 10
    if wpm > 200:
        clarity -= (wpm - 200) * 0.15
    elif wpm < 80 and word_count > 10:
        clarity -= (80 - wpm) * 0.2
    clarity = max(0, min(100, round(clarity)))

    # Engagement score
    unique_words = set(cleaned_words) - FILLER_WORDS_SINGLE
    unique_ratio = len(unique_words) / max(word_count, 1)
    vocab_score = min(40.0, unique_ratio * 60)
    
    if len(sentence_lengths) > 1:
        try:
            length_std = statistics.stdev(sentence_lengths)
        except statistics.StatisticsError:
            length_std = 0.0
        variation_score = min(30.0, length_std * 3)
    else:
        variation_score = 5.0
    
    if word_count > 80:
        substance_score = 30.0
    elif word_count > 40:
        substance_score = 20.0
    elif word_count > 20:
        substance_score = 12.0
    else:
        substance_score = 5.0
    
    engagement = max(0, min(100, round(vocab_score + variation_score + substance_score)))

    return {
        "word_count": word_count,
        "duration_seconds": round(duration_seconds, 1),
        "speaking_pace_wpm": round(wpm, 1),
        "filler_word_count": filler_count,
        "filler_words_found": sorted(set(single_hits)),
        "clarity_score": clarity,
        "engagement_score": engagement,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": round(avg_wps, 1),
    }


# ---------------------------------------------------------------------------
# Confidence Score
# ---------------------------------------------------------------------------
def compute_confidence_score(speech_analyses: list) -> int:
    """Estimate confidence from speech metrics."""
    if not speech_analyses:
        return 0

    total_fillers = sum(a.get("filler_word_count", 0) for a in speech_analyses)
    total_words = sum(a.get("word_count", 0) for a in speech_analyses)
    avg_filler_rate = total_fillers / max(total_words, 1)
    filler_conf = max(0.0, 100 - avg_filler_rate * 300)

    paces = [a["speaking_pace_wpm"] for a in speech_analyses if a.get("speaking_pace_wpm", 0) > 0]
    if paces:
        avg_pace = sum(paces) / len(paces)
        if 120 <= avg_pace <= 160:
            pace_conf = 100.0
        elif 100 <= avg_pace <= 180:
            pace_conf = 75.0
        else:
            pace_conf = max(0.0, 50 - abs(avg_pace - 140) * 0.5)
    else:
        pace_conf = 30.0

    n = len(speech_analyses)
    avg_words = total_words / max(n, 1)
    if avg_words > 60:
        length_conf = 90.0
    elif avg_words > 30:
        length_conf = 70.0
    elif avg_words > 15:
        length_conf = 50.0
    else:
        length_conf = 20.0

    score = 0.4 * filler_conf + 0.3 * pace_conf + 0.3 * length_conf
    return round(max(0, min(100, score)))


# ---------------------------------------------------------------------------
# Overall Score & Verdict
# ---------------------------------------------------------------------------
def compute_overall_score(content_avg: float, clarity_avg: float, engagement_avg: float) -> float:
    """Weighted overall interview score."""
    score = 0.5 * content_avg + 0.3 * clarity_avg + 0.2 * engagement_avg
    return round(max(0.0, min(100.0, score)), 1)


def compute_recruiter_verdict(overall_score: float, role: str) -> dict:
    """Determine hire recommendation."""
    if overall_score >= 70:
        recommendation = "SHORTLISTED"
    elif overall_score >= 50:
        recommendation = "BORDERLINE"
    else:
        recommendation = "REJECT"

    base = role.strip()
    if overall_score >= 70:
        suitable_roles = [base, f"Senior {base}", f"{base} Lead"]
    elif overall_score >= 50:
        suitable_roles = [base, f"Junior {base}"]
    else:
        suitable_roles = [f"Trainee {base}", f"Intern {base}"]

    confidence_level = "High" if overall_score >= 75 or overall_score < 35 else "Medium"

    return {
        "recommendation": recommendation,
        "suitable_roles": suitable_roles,
        "confidence_level": confidence_level,
    }


# ---------------------------------------------------------------------------
# Performance Summary
# ---------------------------------------------------------------------------
async def generate_performance_summary(report_data: dict) -> str:
    """Generate executive performance summary."""
    from prompts import performance_summary_prompt
    
    prompt = performance_summary_prompt(report_data)
    raw = await generate_content(prompt, use_cache=False)
    
    if raw and raw.strip():
        cleaned = raw.strip()
        fence = re.search(r"```(?:\w+)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        return cleaned
    
    return "Performance summary could not be generated. Please review the detailed metrics below."


# ---------------------------------------------------------------------------
# Legacy evaluate_answers function
# ---------------------------------------------------------------------------
async def evaluate_answers(role: str, questions_answers: list) -> dict:
    """Legacy helper for batch evaluation."""
    from prompts import batch_evaluation_prompt
    
    prompt = batch_evaluation_prompt(role, questions_answers)
    raw = await generate_content(prompt, use_cache=False)
    
    try:
        evaluation = extract_json(raw)
        return evaluation
    except Exception:
        return {
            "feedback_per_question": [
                {
                    "question": qa.get("question", ""),
                    "candidate_answer": qa.get("answer", ""),
                    "feedback": "AI evaluation could not be parsed. Please try again.",
                    "improved_answer": "",
                }
                for qa in questions_answers
            ],
            "improvement_tips": [
                "Practice structuring your answers with concrete examples.",
                "Use the STAR method for behavioral questions.",
                "Research the company and role thoroughly.",
            ],
            "learning_resources": [{"topic": "Interview Preparation", "resource": "Practice common questions for your role."}],
        }
