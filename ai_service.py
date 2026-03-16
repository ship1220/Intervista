# ai_service.py
# Core AI + speech analysis utilities

import os
import re
import json
import time
import asyncio
import statistics
import hashlib
import whisper
import subprocess
from typing import Optional, Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

CACHE_TTL_SECONDS = 300
MAX_CACHE_SIZE = 200


# ============================================================
# Whisper Lazy Loader
# ============================================================

_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("small")
    return _whisper_model


# ============================================================
# TTL CACHE
# ============================================================

_cache: Dict[str, Dict] = {}


def _make_cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        return entry["value"]

    _cache.pop(key, None)
    return None


def _cache_set(key: str, value: str):

    if len(_cache) > MAX_CACHE_SIZE:
        oldest = min(_cache, key=lambda k: _cache[k]["ts"])
        _cache.pop(oldest, None)

    _cache[key] = {
        "value": value,
        "ts": time.time()
    }


# ============================================================
# JSON Extraction
# ============================================================

def extract_json(text: str):

    if not text:
        raise ValueError("Empty response")

    cleaned = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)

    if fence_match:
        cleaned = fence_match.group(1)

    first_brace = cleaned.find("{")
    first_bracket = cleaned.find("[")

    if first_bracket != -1 and (first_bracket < first_brace or first_brace == -1):
        start = first_bracket
        end = cleaned.rfind("]")
    else:
        start = first_brace
        end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON detected")

    cleaned = cleaned[start:end + 1]

    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("[JSON ERROR]", e)
        print("[BROKEN JSON]", cleaned)
        raise


# ============================================================
# GROQ CALL
# ============================================================

client = Groq(api_key=GROQ_API_KEY)

async def _call_groq(prompt: str, json_mode: bool = False) -> str:

    def run():
        messages = []

        if json_mode:
            messages.append({
                "role": "system",
                "content": "Return ONLY valid JSON. No explanation."
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=4000
        )

        return response.choices[0].message.content.strip()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run)

# ============================================================
# GENERATE CONTENT
# ============================================================

async def generate_content(
    prompt: str,
    *,
    use_cache=True,
    json_mode=False,
    api_key=None
) -> str:

    key = _make_cache_key(prompt)

    if use_cache:
        cached = _cache_get(key)

        if cached:
            return cached

    try:
        result = await _call_groq(prompt, json_mode=json_mode)

    except Exception as e:
        print("[groq] error:", e)
        result = ""

    if use_cache and result:
        _cache_set(key, result)

    return result


# ============================================================
# SPEECH ANALYSIS
# ============================================================

FILLER_WORDS = {
    "um", "uh", "like", "so", "well",
    "actually", "basically", "literally",
    "totally"
}


def analyze_speech_delivery(answer: str, duration_seconds: float) -> dict:

    if not answer.strip():

        return {
            "word_count": 0,
            "duration_seconds": duration_seconds,
            "speaking_pace_wpm": 0,
            "filler_word_count": 0,
            "filler_words_found": [],
            "clarity_score": 0,
            "engagement_score": 0
        }

    words = answer.split()

    word_count = len(words)

    duration_minutes = max(duration_seconds / 60.0, 0.01)

    wpm = word_count / duration_minutes

    cleaned = [w.lower().strip(".,!?") for w in words]

    fillers = [w for w in cleaned if w in FILLER_WORDS]

    filler_rate = len(fillers) / max(word_count, 1)

    sentences = re.split(r"[.!?]+", answer)

    sentences = [s for s in sentences if s.strip()]

    sentence_lengths = [len(s.split()) for s in sentences]

    avg_sentence = word_count / max(len(sentences), 1)

    clarity = 80

    clarity -= filler_rate * 150

    if avg_sentence > 30:
        clarity -= (avg_sentence - 30)

    clarity = max(0, min(100, round(clarity)))

    unique_words = len(set(cleaned))

    vocab_ratio = unique_words / max(word_count, 1)

    engagement = min(100, round(vocab_ratio * 100))

    return {

        "word_count": word_count,
        "duration_seconds": round(duration_seconds, 1),
        "speaking_pace_wpm": round(wpm, 1),
        "filler_word_count": len(fillers),
        "filler_words_found": fillers,
        "clarity_score": clarity,
        "engagement_score": engagement
    }


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def compute_confidence_score(speech_analyses: list) -> int:

    if not speech_analyses:
        return 0

    total_fillers = sum(a["filler_word_count"] for a in speech_analyses)

    total_words = sum(a["word_count"] for a in speech_analyses)

    filler_rate = total_fillers / max(total_words, 1)

    filler_conf = max(0, 100 - filler_rate * 300)

    paces = [a["speaking_pace_wpm"] for a in speech_analyses]

    avg_pace = sum(paces) / len(paces)

    pace_conf = 100 if 120 <= avg_pace <= 160 else 70

    avg_words = total_words / len(speech_analyses)

    length_conf = 90 if avg_words > 60 else 60

    score = 0.4 * filler_conf + 0.3 * pace_conf + 0.3 * length_conf

    return round(max(0, min(100, score)))


# ============================================================
# CONTENT EVALUATION
# ============================================================

# ============================================================
# CONTENT EVALUATION
# ============================================================

async def evaluate_content(role: str, level: str, questions_answers: list, api_key=None):

    results = []
    weak_topics = []

    for qa in questions_answers:

        question = qa.get("question", "")[:120]
        answer = qa.get("answer", "").strip()[:1500]

        skipped = not answer or answer.lower() in ["(skipped)", "(no response)"]

        if skipped:
            answer = "(skipped)"

        prompt = f"""
You are evaluating an interview answer for a {level} {role} candidate.

IMPORTANT:
- Answers come from speech transcription. Ignore grammar errors.
Scoring rules:

0 → Question skipped or empty answer
10-20 → Completely incorrect answer
50-70 → Conceptually correct but poorly structured
80–90 → Correct, clear explanation
90–100 → Excellent with strong explanation

If the candidate answer is "(skipped)":
- Score MUST be 0
- Mention that the candidate skipped the question
- ALWAYS generate a full ideal_answer teaching the correct concept

Question:
{question}

Answer:
{answer}

Return JSON:

{{
 "score": number,
 "strengths": ["strength"],
 "weaknesses": ["weakness"],
 "ideal_answer": "better answer",
 "weak_topics": ["topic"]
}}
"""

        raw = await generate_content(prompt, use_cache=False, json_mode=True)

        try:
            parsed = extract_json(raw)

            topics = parsed.get("weak_topics", [])
            weak_topics.extend(topics)

            score = parsed.get("score", 0)

            try:
                score = float(score)
            except:
                score = 0

            results.append({
                "score": score,
                "strengths": parsed.get("strengths", ["Answer attempted."]),
                "weaknesses": parsed.get("weaknesses", ["Needs improvement."]),
                "ideal_answer": parsed.get("ideal_answer", "Ideal answer unavailable."),
                "weak_topics": topics
            })

        except Exception:

            results.append({
                "score": 0,
                "strengths": [],
                "weaknesses": ["Evaluation unavailable due to parsing error."],
                "ideal_answer": "Ideal answer could not be generated.",
                "weak_topics": []
            })

    scores = [r["score"] for r in results]

    avg = sum(scores) / len(scores) if scores else 0

    return {
        "answers": results,
        "weak_topics": list(set(weak_topics)),
        "overall_feedback": "",
        "aggregate": {
            "technical_score": round(avg),
            "communication_score": round(avg * 0.9),
            "overall_score": round(avg)
        }
    }

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

async def generate_performance_summary(report_data: dict, api_key=None):

    role = report_data.get("candidate_profile", {}).get("role")

    score = report_data.get("overall_score")

    answers = report_data.get("detailed_answers", [])

    attempted = [
        a for a in answers
        if a.get("transcript") not in ["", "(skipped)", "(no response)"]
    ]

    if len(attempted) == 0:
        return (
            "The candidate did not provide answers to the interview questions. "
            "As a result, the system could not evaluate technical knowledge or communication ability. "
            "Overall performance is considered very poor due to lack of responses. "
            "The candidate should attempt answering questions to receive meaningful feedback."
        )

    prompt = f"""
Write a concise 4 sentence interview performance summary.

Role: {role}
Overall Score: {score}

Explain strengths, weaknesses, communication quality,
and one recommendation for improvement.
"""

    raw = await generate_content(prompt, use_cache=False, json_mode=False)

    return raw.strip() if raw else "Summary unavailable."


    

# ============================================================
# OVERALL SCORE
# ============================================================

def compute_overall_score(content_avg, clarity_avg, engagement_avg, answers=None):

    # If candidate answered nothing
    attempted = [a for a in answers if a["answer"] not in ["(skipped)", "(no response)", ""]]

    if len(attempted) == 0:
        return 0

    score = 0.5 * content_avg + 0.3 * clarity_avg + 0.2 * engagement_avg
    return round(max(0, min(100, score)), 1)


def compute_recruiter_verdict(overall_score: float, role: str):

    if overall_score >= 70:

        rec = "SHORTLISTED"

    elif overall_score >= 50:

        rec = "BORDERLINE"

    else:

        rec = "REJECT"

    return {

        "recommendation": rec,

        "confidence_level": "High" if overall_score >= 70 else "Medium",

        "suitable_roles": [role]
    }


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def convert_audio(input_file, output_file):

    subprocess.run([

        "ffmpeg",
        "-i", input_file,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_file

    ])


def transcribe_audio(file_path):

    try:

        converted = f"converted_{int(time.time()*1000)}.wav"

        convert_audio(file_path, converted)

        model = get_whisper_model()

        result = model.transcribe(

            converted,
            language="en",
            beam_size=5,
            temperature=0
        )

        return result["text"].strip()

    except Exception as e:

        print("[whisper] error:", e)

        return ""

async def evaluate_answers(role: str, questions_answers: list) -> dict:
    """
    Legacy compatibility function used by main.py.
    Uses batch evaluation prompt.
    """

    from prompts import batch_evaluation_prompt

    prompt = batch_evaluation_prompt(role, questions_answers)

    raw = await generate_content(prompt, use_cache=False, json_mode=True)

    try:
        evaluation = extract_json(raw)
        return evaluation

    except Exception:
        return {
            "feedback_per_question": [
                {
                    "question": qa.get("question", ""),
                    "candidate_answer": qa.get("answer", ""),
                    "feedback": "Evaluation could not be parsed.",
                    "improved_answer": ""
                }
                for qa in questions_answers
            ],
            "improvement_tips": [
                "Structure answers clearly.",
                "Use the STAR method for behavioral questions.",
                "Explain your reasoning step-by-step."
            ],
            "learning_resources": [
                {
                    "topic": "Interview preparation",
                    "resource": "Practice common interview questions."
                }
            ]
        }