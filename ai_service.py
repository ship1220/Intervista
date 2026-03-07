# ai_service.py — Async AI service with caching and Ollama → Gemini fallback

import os
import re
import json
import time
import asyncio
import httpx
import google.genai as genai
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
# Core async generation — tries Ollama first, then Gemini
# ---------------------------------------------------------------------------
async def generate_content(prompt: str, *, use_cache: bool = True) -> str:
    """Send *prompt* to an LLM and return the text response."""
    if use_cache:
        cached = _cache_get(prompt)
        if cached is not None:
            return cached

    result = await _try_ollama(prompt)
    if result is None:
        result = await _try_gemini(prompt)
    if result is None:
        result = ""

    # Log first 300 chars for debugging
    preview = result[:300].replace("\n", " ") if result else "(empty)"
    print(f"[ai_service] response preview: {preview}")

    if use_cache and result:
        _cache_set(prompt, result)
    return result


async def _try_ollama(prompt: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
    except Exception as exc:
        print(f"[ai_service] Ollama error: {exc}")
    return None


async def _try_gemini(prompt: str) -> str | None:
    """Run Gemini in a thread executor so it doesn't block the event loop."""
    if not GEMINI_API_KEY:
        return None
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _gemini_sync, prompt)
    except Exception as exc:
        print(f"[ai_service] Gemini error: {exc}")
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
    return json.loads(fixed)


# --------------------------------------------------------------------------- 
# Evaluate answers using AI
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
