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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Configuration
OLLAMA_TIMEOUT_SECONDS = 120
GEMINI_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
CACHE_TTL_SECONDS = 300
MAX_CACHE_SIZE = 200

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
    connect=10.0,
    read=OLLAMA_TIMEOUT_SECONDS,
    write=30.0,
    pool=10.0,
)

_GEMINI_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=GEMINI_TIMEOUT_SECONDS,
    write=30.0,
    pool=10.0,
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
    fixed = re.sub(r'(?<=["])((?:[^"\\]|\\.)*)(?=[":])',
                   lambda m: m.group(0).replace('\n', '\\n'), fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as exc:
        print(f"[extract_json] All parse attempts failed: {exc}")
        print(f"[extract_json] Raw text ({len(text)} chars): {text[:500]}")
        raise


# ---------------------------------------------------------------------------
# Response Validator
# ---------------------------------------------------------------------------
def validate_response(result: Dict, required_keys: list = None) -> bool:
    """Validate that the response contains expected structure."""
    if not isinstance(result, dict):
        return False
    
    if required_keys is None:
        required_keys = ["answers", "overall_feedback", "aggregate"]
    
    for key in required_keys:
        if key not in result:
            return False
    
    # Validate answers is a list
    if not isinstance(result.get("answers"), list):
        return False
    
    # Validate each answer has required fields
    for answer in result.get("answers", []):
        if not isinstance(answer, dict):
            return False
        if "score" not in answer or "feedback" not in answer:
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
                "feedback": "AI evaluation service temporarily unavailable. Please retry the interview for detailed feedback.",
                "strengths": ["Answer recorded successfully"],
                "improvements": ["Retry interview when service is available"]
            }
            for _ in range(num_questions)
        ],
        "overall_feedback": "Evaluation service temporarily unavailable. Please retry the interview to receive comprehensive feedback on your responses.",
        "aggregate": {
            "technical_score": 50,
            "communication_score": 50,
            "overall_score": 50
        }
    }


# ---------------------------------------------------------------------------
# Ollama Implementation
# ---------------------------------------------------------------------------
async def _try_ollama_with_retry(prompt: str, json_mode: bool = False) -> Optional[str]:
    """Try Ollama with retry logic."""
    
    for attempt in range(MAX_RETRIES):
        try:
            # Try chat endpoint first
            text = await _ollama_chat(prompt, json_mode)
            if text:
                print(f"[ollama] Chat attempt {attempt + 1} succeeded")
                return text
            
            # Try generate endpoint
            text = await _ollama_generate(prompt, json_mode)
            if text:
                print(f"[ollama] Generate attempt {attempt + 1} succeeded")
                return text
                
        except httpx.ReadTimeout:
            print(f"[ollama] ReadTimeout on attempt {attempt + 1}/{MAX_RETRIES}")
        except Exception as e:
            print(f"[ollama] Error on attempt {attempt + 1}: {type(e).__name__}: {e}")
        
        if attempt < MAX_RETRIES - 1:
            print(f"[ollama] Retrying in {RETRY_DELAY_SECONDS}s...")
            await asyncio.sleep(RETRY_DELAY_SECONDS)
    
    return None


async def _ollama_chat(prompt: str, json_mode: bool = False) -> Optional[str]:
    """Call Ollama /api/chat endpoint."""
    try:
        client = _get_ollama_client()
        system_msg = (
            "You are a helpful AI assistant. You MUST respond with valid JSON only. "
            "No markdown fences, no explanation before or after the JSON."
        ) if json_mode else "You are a helpful AI assistant."
        
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        
        start_time = time.time()
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        latency = time.time() - start_time
        print(f"[ollama] /api/chat latency: {latency:.2f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            text = (data.get("message", {}).get("content", "")).strip()
            if text:
                return text
            print("[ollama] /api/chat returned 200 but empty content")
        else:
            print(f"[ollama] /api/chat HTTP {resp.status_code}: {resp.text[:200]}")
    except httpx.ReadTimeout:
        print("[ollama] /api/chat ReadTimeout")
        raise
    except Exception as e:
        print(f"[ollama] /api/chat error: {type(e).__name__}: {e}")
    return None


async def _ollama_generate(prompt: str, json_mode: bool = False) -> Optional[str]:
    """Call Ollama /api/generate endpoint."""
    try:
        client = _get_ollama_client()
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        
        start_time = time.time()
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        latency = time.time() - start_time
        print(f"[ollama] /api/generate latency: {latency:.2f}s")
        
        if resp.status_code == 200:
            text = resp.json().get("response", "").strip()
            if text:
                return text
            print("[ollama] /api/generate returned 200 but empty response")
        else:
            print(f"[ollama] /api/generate HTTP {resp.status_code}: {resp.text[:200]}")
    except httpx.ReadTimeout:
        print("[ollama] /api/generate ReadTimeout")
        raise
    except Exception as e:
        print(f"[ollama] /api/generate error: {type(e).__name__}: {e}")
    return None


# ---------------------------------------------------------------------------
# Gemini Implementation (Fixed for new SDK)
# ---------------------------------------------------------------------------
async def _try_gemini(prompt: str) -> Optional[str]:
    """Run Gemini with retry logic."""
    if not GEMINI_API_KEY:
        print("[gemini] No GEMINI_API_KEY configured")
        return None
    
    for attempt in range(MAX_RETRIES):
        try:
            result = await _gemini_async(prompt)
            if result:
                print(f"[gemini] Attempt {attempt + 1} succeeded")
                return result
        except Exception as e:
            print(f"[gemini] Error on attempt {attempt + 1}: {type(e).__name__}: {e}")
        
        if attempt < MAX_RETRIES - 1:
            print(f"[gemini] Retrying in {RETRY_DELAY_SECONDS}s...")
            await asyncio.sleep(RETRY_DELAY_SECONDS)
    
    return None


async def _gemini_async(prompt: str) -> str:
    """Execute Gemini call asynchronously."""
    def _call_gemini():
        # Configure the API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Use the correct model name
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        start_time = time.time()
        response = model.generate_content(prompt)
        latency = time.time() - start_time
        print(f"[gemini] latency: {latency:.2f}s")
        
        return response.text.strip()
    
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _call_gemini)
    except Exception as e:
        print(f"[gemini] Async error: {type(e).__name__}: {e}")
        raise


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
        print("[ai_service] Ollama failed, trying Gemini...")
        result = await _try_gemini(prompt)
    
    # If all failed, return empty
    if not result:
        print("[ai_service] WARNING: All AI backends failed")
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
    from prompts import content_analysis_prompt
    
    num_questions = len(questions_answers)
    prompt = content_analysis_prompt(role, level, questions_answers)
    
    # Try to get AI response
    raw = await generate_content(prompt, use_cache=False, json_mode=True)
    
    print(f"[evaluate_content] Raw response length: {len(raw) if raw else 0}")
    
    # Try to parse and validate
    if raw and raw.strip():
        try:
            result = extract_json(raw)
            print(f"[evaluate_content] Parsed JSON keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            
            if isinstance(result, dict):
                # Try multiple times with retry for valid structure
                for retry in range(2):
                    if validate_response(result):
                        # Normalize the response
                        result = _normalize_response(result, num_questions)
                        return result
                    else:
                        print(f"[evaluate_content] Response validation failed, attempt {retry + 1}")
                        # Try one more time with different prompt or just return what we have
                        if retry == 0:
                            # Try to extract answers from different key formats
                            result = _normalize_response(result, num_questions)
                
                # If still not valid, use fallback
                if not validate_response(result):
                    print("[evaluate_content] Using fallback - invalid structure")
                    return get_fallback_response(num_questions)
                
                return result
                
        except Exception as e:
            print(f"[evaluate_content] Parse error: {e}")
    
    # Return fallback
    print("[evaluate_content] Using fallback - no valid response")
    return get_fallback_response(num_questions)


def _normalize_response(result: Dict, num_questions: int) -> Dict:
    """Normalize response to expected format."""
    
    # Find answers from various possible keys
    answers = None
    for key in ["answers", "feedback_per_question", "evaluations", "results", "response"]:
        if key in result and isinstance(result[key], list):
            answers = result[key]
            break
    
    # Ensure we have answers for all questions
    if answers is None:
        answers = []
    elif len(answers) < num_questions:
        # Pad with default entries
        while len(answers) < num_questions:
            answers.append({
                "score": 50,
                "feedback": "Additional feedback pending.",
                "strengths": [],
                "improvements": []
            })
    
    # Ensure each answer has required fields
    normalized_answers = []
    for i, ans in enumerate(answers[:num_questions]):
        if isinstance(ans, dict):
            normalized_answers.append({
                "score": ans.get("score", 50),
                "feedback": ans.get("feedback", ans.get("comment", "Feedback unavailable.")),
                "strengths": ans.get("strengths", ans.get("strength", [])),
                "improvements": ans.get("improvements", ans.get("improvements", []))
            })
        else:
            normalized_answers.append({
                "score": 50,
                "feedback": str(ans) if ans else "Feedback unavailable.",
                "strengths": [],
                "improvements": []
            })
    
    # Get aggregate scores
    aggregate = result.get("aggregate", result.get("scores", {}))
    if not isinstance(aggregate, dict):
        aggregate = {}
    
    # Ensure aggregate has required fields
    normalized_aggregate = {
        "technical_score": aggregate.get("technical_score", aggregate.get("technical", 50)),
        "communication_score": aggregate.get("communication_score", aggregate.get("communication", 50)),
        "overall_score": aggregate.get("overall_score", aggregate.get("overall", 50))
    }
    
    # Get overall feedback
    overall_feedback = result.get("overall_feedback", result.get("summary", ""))
    if not overall_feedback:
        overall_feedback = "Evaluation completed. Please review individual answer feedback above."
    
    return {
        "answers": normalized_answers,
        "overall_feedback": overall_feedback,
        "aggregate": normalized_aggregate
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
            "word_count": 0, "duration_seconds": round(duration_seconds, 1),
            "speaking_pace_wpm": 0.0, "filler_word_count": 0,
            "filler_words_found": [], "clarity_score": 0,
            "engagement_score": 0, "sentence_count": 0,
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
