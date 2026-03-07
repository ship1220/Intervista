# ai_service.py — Async AI service with caching and Ollama → Gemini fallback

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

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ---------------------------------------------------------------------------
# Simple TTL cache — avoids redundant AI calls for identical prompts
# ---------------------------------------------------------------------------
_cache: dict[str, dict] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _cache_get(key: str) -> str | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        return entry["value"]
    _cache.pop(key, None)
    return None


def _cache_set(key: str, value: str) -> None:
    if len(_cache) > 200:
        oldest = min(_cache, key=lambda k: _cache[k]["ts"])
        _cache.pop(oldest, None)
    _cache[key] = {"value": value, "ts": time.time()}


# ---------------------------------------------------------------------------
# Shared timeout — read must be generous for local LLM generation
# ---------------------------------------------------------------------------
_OLLAMA_TIMEOUT = httpx.Timeout(
    connect=10.0,   # TCP connect
    read=300.0,     # wait for Ollama to finish generating (5 min)
    write=30.0,     # sending the prompt
    pool=10.0,      # connection-pool acquisition
)

# ---------------------------------------------------------------------------
# Reusable httpx client — avoids TCP handshake overhead per request
# ---------------------------------------------------------------------------
_ollama_client: httpx.AsyncClient | None = None


def _get_ollama_client() -> httpx.AsyncClient:
    global _ollama_client
    if _ollama_client is None or _ollama_client.is_closed:
        _ollama_client = httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT)
    return _ollama_client


# ---------------------------------------------------------------------------
# Core async generation — tries Ollama first, then Gemini
# ---------------------------------------------------------------------------
async def generate_content(prompt: str, *, use_cache: bool = True, json_mode: bool = False) -> str:
    """Send *prompt* to an LLM and return the text response.

    When *json_mode* is True, Ollama is instructed to constrain output to
    valid JSON (``format: "json"``) and the system prompt is adjusted.
    """
    if use_cache:
        cached = _cache_get(prompt)
        if cached is not None:
            print("[ai_service] returning cached response")
            return cached

    result = await _try_ollama(prompt, json_mode=json_mode)
    if not result:
        result = await _try_gemini(prompt)

    if not result:
        print(
            "[ai_service] WARNING: All AI backends failed. "
            "Ensure Ollama is running (ollama serve) or set GEMINI_API_KEY in .env"
        )
        return ""

    # Log first 300 chars for debugging
    preview = result[:300].replace("\n", " ")
    print(f"[ai_service] response preview ({len(result)} chars): {preview}")

    if use_cache:
        _cache_set(prompt, result)
    return result


# ---------------------------------------------------------------------------
# Ollama — try /api/chat (better for instruction-tuned models), then
#          fall back to /api/generate
# ---------------------------------------------------------------------------
async def _try_ollama(prompt: str, json_mode: bool = False) -> str | None:
    """Attempt to get a response from Ollama. Returns None only if both
    /api/chat and /api/generate fail or return empty text."""

    # --- Attempt 1: /api/chat (preferred for llama3-style models) ---------
    text = await _ollama_chat(prompt, json_mode=json_mode)
    if text:
        return text

    # --- Attempt 2: /api/generate (raw completion) ------------------------
    text = await _ollama_generate(prompt, json_mode=json_mode)
    if text:
        return text

    return None


async def _ollama_chat(prompt: str, json_mode: bool = False) -> str | None:
    """Call Ollama /api/chat with a system + user message."""
    try:
        client = _get_ollama_client()
        system_msg = (
            "You are a helpful AI assistant. You MUST respond with valid JSON only. "
            "No markdown fences, no explanation before or after the JSON."
        ) if json_mode else (
            "You are a helpful AI assistant."
        )
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
        print(f"[ai_service] POST {OLLAMA_URL}/api/chat  model={OLLAMA_MODEL}  json_mode={json_mode}")
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)

        if resp.status_code == 200:
            data = resp.json()
            text = (data.get("message", {}).get("content", "")).strip()
            if text:
                return text
            print("[ai_service] /api/chat returned 200 but message.content is empty")
        else:
            body = resp.text[:500]
            print(f"[ai_service] /api/chat HTTP {resp.status_code}: {body}")
    except Exception as exc:
        print(f"[ai_service] /api/chat error ({type(exc).__name__}): {exc}")
    return None


async def _ollama_generate(prompt: str, json_mode: bool = False) -> str | None:
    """Call Ollama /api/generate (raw completion)."""
    try:
        client = _get_ollama_client()
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        print(f"[ai_service] POST {OLLAMA_URL}/api/generate  model={OLLAMA_MODEL}  json_mode={json_mode}")
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)

        if resp.status_code == 200:
            text = resp.json().get("response", "").strip()
            if text:
                return text
            print("[ai_service] /api/generate returned 200 but response field is empty")
        else:
            body = resp.text[:500]
            print(f"[ai_service] /api/generate HTTP {resp.status_code}: {body}")
    except Exception as exc:
        print(f"[ai_service] /api/generate error ({type(exc).__name__}): {exc}")
    return None


# ---------------------------------------------------------------------------
# Gemini fallback
# ---------------------------------------------------------------------------
async def _try_gemini(prompt: str) -> str | None:
    """Run Gemini in a thread executor so it doesn't block the event loop."""
    if not GEMINI_API_KEY:
        print("[ai_service] Gemini skipped — no GEMINI_API_KEY configured")
        return None
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _gemini_sync, prompt)
    except Exception as exc:
        print(f"[ai_service] Gemini error ({type(exc).__name__}): {exc}")
    return None


def _gemini_sync(prompt: str) -> str:
    """Synchronous Gemini call — executed inside run_in_executor."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# JSON extraction helper — handles markdown fences and stray text
# ---------------------------------------------------------------------------
def extract_json(text: str):
    """Parse JSON from *text*, stripping markdown fences and surrounding text."""
    if not text or not text.strip():
        raise ValueError("Empty text — no JSON to extract")

    cleaned = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Try to find the outermost JSON object or array
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

    # Attempt 2: fix trailing commas (common AI mistake)
    fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: strip stray control characters and retry
    fixed = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 4: fix unescaped newlines inside JSON string values
    fixed = re.sub(r'(?<=["])((?:[^"\\]|\\.)*)(?=[":])',
                   lambda m: m.group(0).replace('\n', '\\n'), fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as exc:
        # Log the FULL raw text so the root cause is always visible
        print(f"[extract_json] all parse attempts failed: {exc}")
        print(f"[extract_json] FULL raw text ({len(text)} chars):\n{text}")
        print(f"[extract_json] FULL cleaned text ({len(fixed)} chars):\n{fixed}")
        raise


# ---------------------------------------------------------------------------
# Filler word constants
# ---------------------------------------------------------------------------
FILLER_WORDS_SINGLE: set[str] = {
    "um", "uh", "like", "so", "well", "actually", "basically",
    "literally", "totally", "right", "anyway", "okay",
}
MULTI_WORD_FILLERS: list[str] = ["you know", "i mean", "kind of", "sort of"]


# ---------------------------------------------------------------------------
# FEATURE 2 — Speech Delivery Analysis
# ---------------------------------------------------------------------------
def analyze_speech_delivery(answer: str, duration_seconds: float) -> dict:
    """Analyse speech delivery from transcript text and duration.

    Returns a dict with speaking_pace_wpm, filler_word_count,
    filler_words_found, clarity_score (0-100), engagement_score (0-100),
    word_count, sentence_count, avg_words_per_sentence.
    """
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

    # --- Speaking Pace (WPM) ---
    wpm = word_count / duration_minutes

    # --- Filler Detection ---
    cleaned_words = [w.lower().strip(".,!?;:'\"") for w in words]
    single_hits = [w for w in cleaned_words if w in FILLER_WORDS_SINGLE]
    lower_text = answer.lower()
    multi_count = sum(lower_text.count(mf) for mf in MULTI_WORD_FILLERS)
    filler_count = len(single_hits) + multi_count
    filler_rate = filler_count / max(word_count, 1)

    # --- Sentence Analysis ---
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_wps = word_count / sentence_count

    # --- Clarity Score (0-100) ---
    clarity = 80.0
    # Filler penalty (10 % fillers ≈ −15 pts)
    clarity -= filler_rate * 150
    # Sentence length: optimal 10–25 words
    if avg_wps > 30:
        clarity -= (avg_wps - 30) * 1.0
    elif avg_wps < 6:
        clarity -= (6 - avg_wps) * 3.0
    # Very short answers lack clarity context
    if word_count < 15:
        clarity -= 25
    elif word_count < 30:
        clarity -= 10
    # Speaking pace: too fast or too slow
    if wpm > 200:
        clarity -= (wpm - 200) * 0.15
    elif wpm < 80 and word_count > 10:
        clarity -= (80 - wpm) * 0.2
    clarity = max(0, min(100, round(clarity)))

    # --- Engagement Score (0-100) ---
    # Vocabulary richness (unique non-filler words / total)
    unique_words = set(cleaned_words) - FILLER_WORDS_SINGLE
    unique_ratio = len(unique_words) / max(word_count, 1)
    vocab_score = min(40.0, unique_ratio * 60)
    # Sentence length variation
    if len(sentence_lengths) > 1:
        try:
            length_std = statistics.stdev(sentence_lengths)
        except statistics.StatisticsError:
            length_std = 0.0
        variation_score = min(30.0, length_std * 3)
    else:
        variation_score = 5.0
    # Answer substantiveness
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
# FEATURE 4 — Confidence Score
# ---------------------------------------------------------------------------
def compute_confidence_score(speech_analyses: list[dict]) -> int:
    """Estimate confidence (0-100) from aggregated speech metrics."""
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
# FEATURE 6 — Overall Score & Recruiter Verdict
# ---------------------------------------------------------------------------
def compute_overall_score(
    content_avg: float, clarity_avg: float, engagement_avg: float
) -> float:
    """Weighted overall interview score (0-100).

    Formula: 0.5 * content + 0.3 * clarity + 0.2 * engagement
    """
    score = 0.5 * content_avg + 0.3 * clarity_avg + 0.2 * engagement_avg
    return round(max(0.0, min(100.0, score)), 1)


def compute_recruiter_verdict(overall_score: float, role: str) -> dict:
    """Determine hire recommendation and suggest suitable roles."""
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

    if overall_score >= 75 or overall_score < 35:
        confidence_level = "High"
    else:
        confidence_level = "Medium"

    return {
        "recommendation": recommendation,
        "suitable_roles": suitable_roles,
        "confidence_level": confidence_level,
    }


# ---------------------------------------------------------------------------
# FEATURE 3 — Content Evaluation via LLM
# ---------------------------------------------------------------------------
async def evaluate_content(
    role: str, level: str, questions_answers: list[dict]
) -> dict:
    """Evaluate interview content using LLM and return structured analysis.

    Returns a dict with ``answers`` (list of per-question dicts each containing
    ``score`` and ``feedback``), ``overall_feedback``, and ``aggregate``.
    """
    from prompts import content_analysis_prompt

    prompt = content_analysis_prompt(role, level, questions_answers)
    raw = await generate_content(prompt, use_cache=False, json_mode=True)

    if not raw or not raw.strip():
        print("[evaluate_content] AI returned empty response — using fallback")
    else:
        try:
            result = extract_json(raw)
            if isinstance(result, dict):
                # Normalise: accept variant top-level keys → "answers"
                for alt in ("feedback_per_question", "evaluations", "results"):
                    if alt in result and "answers" not in result:
                        result["answers"] = result.pop(alt)
                        break
                return result
            print(f"[evaluate_content] parsed JSON is {type(result).__name__}, expected dict")
        except Exception as exc:
            print(f"[evaluate_content] JSON parse error: {exc}")

    # Fallback — single-paragraph feedback per question
    return {
        "answers": [
            {
                "score": 50,
                "feedback": (
                    "AI evaluation could not be parsed. Please try again. "
                    "Your answer was recorded but automatic feedback is unavailable."
                ),
            }
            for _ in questions_answers
        ],
        "overall_feedback": "Evaluation could not be completed. Please retry the interview.",
        "aggregate": {
            "relevance_score": 50,
            "depth_score": 50,
            "star_method_score": 30,
        },
    }


# ---------------------------------------------------------------------------
# FEATURE 5 — Performance Summary via LLM
# ---------------------------------------------------------------------------
async def generate_performance_summary(report_data: dict) -> str:
    """Generate an executive performance summary using LLM."""
    from prompts import performance_summary_prompt

    prompt = performance_summary_prompt(report_data)
    raw = await generate_content(prompt, use_cache=False)

    if raw and raw.strip():
        cleaned = raw.strip()
        # Strip markdown code fences if the model wrapped its response
        fence = re.search(r"```(?:\w+)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        return cleaned

    return (
        "Performance summary could not be generated. "
        "Please review the detailed metrics below."
    )


# --------------------------------------------------------------------------- 
# Evaluate answers using AI (legacy helper)
# ---------------------------------------------------------------------------
async def evaluate_answers(role: str, questions_answers: list[dict]) -> dict:
    """Evaluate interview answers using AI and return structured feedback."""
    from prompts import batch_evaluation_prompt  # Import here to avoid circular import
    
    prompt = batch_evaluation_prompt(role, questions_answers)
    raw = await generate_content(prompt, use_cache=False)
    
    try:
        evaluation = extract_json(raw)
        return evaluation
    except Exception:
        # Fallback similar to main.py
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
                "Use the STAR method (Situation, Task, Action, Result) for behavioral questions.",
                "Research the company and role thoroughly before interviews.",
            ],
            "learning_resources": [{"topic": "Interview Preparation", "resource": "Practice common behavioral and technical questions for your role."}],
        }
