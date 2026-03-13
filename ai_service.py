# ai_service.py — Robust async AI service with Ollama → Gemini fallback

import os
import re
import json
import time
import whisper
import subprocess
import math
import asyncio
import statistics
from groq import Groq
from dotenv import load_dotenv
from typing import Optional, Dict, Any

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

groq_client = Groq(api_key=GROQ_API_KEY)
# Whisper model (loaded once)
whisper_model = whisper.load_model("small")
# Configuration

MAX_RETRIES = 1                 # single attempt — streaming eliminates timeout-based retries
RETRY_DELAY_SECONDS = 0.5
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


def _cache_set(key: str, value: str):
    if len(_cache) > MAX_CACHE_SIZE:
        oldest = min(_cache, key=lambda k: _cache[k]["ts"])
        _cache.pop(oldest, None)
    _cache[key] = {"value": value, "ts": time.time()}

def compress_prompt(prompt: str, max_chars: int = 2500):
    if len(prompt) > max_chars:
        return prompt[:max_chars]
    return prompt

# ---------------------------------------------------------------------------
# JSON Extraction Helper
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Any:

    if not text:
        raise ValueError("Empty response")

    cleaned = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)

    if fence_match:
        cleaned = fence_match.group(1)

    first = cleaned.find("{")
    last = cleaned.rfind("}")

    if first != -1 and last != -1:
        cleaned = cleaned[first:last + 1]

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

# ------------------------------------------------------------------
# GROQ CALL
# ------------------------------------------------------------------
async def _call_groq(prompt: str, json_mode=False) -> str:

    def _run():

        messages = []

        if json_mode:
            messages.append({
                "role": "system",
                "content": (
                     "You are a strict JSON generator. "
                     "Return ONLY valid JSON. "
                     "Do NOT add explanations. "
                     "Do NOT add text before or after the JSON. "
                     "Do NOT use markdown."
                 )
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
        )

        content = response.choices[0].message.content

        if not content:
            return ""

        return content.strip()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run)
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
# Main Content Generation
# ---------------------------------------------------------------------------
async def generate_content(
    prompt: str,
    *,
    use_cache: bool = True,
    json_mode: bool = False
) -> str:

    if use_cache:
        cached = _cache_get(prompt)
        if cached:
            return cached

    prompt = compress_prompt(prompt)

    try:
        result = await _call_groq(prompt, json_mode=json_mode)

    except Exception as e:
        print("[groq] error:", e)
        result = ""

    if use_cache and result:
        _cache_set(prompt, result)

    return result
# ---------------------------------------------------------------------------
# Evaluate Content - Main function for interview evaluation
# ---------------------------------------------------------------------------
async def evaluate_content(role: str, level: str, questions_answers: list) -> Dict:

    from prompts import interview_evaluation_prompt

    prompt = interview_evaluation_prompt(role, questions_answers)

    raw = await generate_content(prompt, use_cache=False, json_mode=True)

    print("RAW EVALUATION:", raw[:1200])

    try:
        result = extract_json(raw)

        answers = result.get("answers")

        # If answers is a dict → convert to list
        if isinstance(answers, dict):
            answers = list(answers.values())

        # If answers missing → search for Q1/Q2 keys
        if not answers:
            extracted = []
            for key, value in result.items():
                if key.lower().startswith("q") and isinstance(value, dict):
                    extracted.append(value)

            if extracted:
                answers = extracted

        # If still invalid → fallback
        if not isinstance(answers, list) or len(answers) == 0:
            answers = [
                {
                    "score": 50,
                    "feedback": "Evaluation could not be generated.",
                    "strengths": [],
                    "weaknesses": [],
                    "ideal_answer": ""
                }
                for _ in questions_answers
            ]

        result["answers"] = answers

        result.setdefault("overall_feedback", "")
        result.setdefault("aggregate", {})

        # Ensure each answer has required fields
        for ans in result["answers"]:
            ans.setdefault("score", 50)
            ans.setdefault("feedback", "No detailed feedback generated.")
            ans.setdefault("strengths", ["Answer attempted."])
            ans.setdefault("weaknesses", ["More explanation needed."])
            ans.setdefault(
                "ideal_answer",
                "A more structured answer with examples would improve this response."
            )

    except Exception:

        # Fallback: parse plain-text evaluation
        answers = []
        blocks = raw.split("**Q")

        for block in blocks[1:]:
            try:
                score_match = re.search(r"Score:\s*(\d+)", block)
                feedback_match = re.search(r"Feedback:\s*(.*?)(?:Strengths:)", block, re.S)

                score = int(score_match.group(1)) if score_match else 50
                feedback = feedback_match.group(1).strip() if feedback_match else "No feedback available."

                answers.append({
                    "score": score,
                    "feedback": feedback,
                    "strengths": ["Answer attempted."],
                    "weaknesses": ["Needs clearer structure and examples."],
                    "ideal_answer": ""
                })

            except Exception:
                answers.append({
                    "score": 50,
                    "feedback": "Evaluation unavailable.",
                    "strengths": [],
                    "weaknesses": [],
                    "ideal_answer": ""
                })

        # Ensure answer count matches questions
        if len(answers) < len(questions_answers):
            answers.extend([
                {
                    "score": 50,
                    "feedback": "Evaluation unavailable.",
                    "strengths": [],
                    "weaknesses": [],
                    "ideal_answer": ""
                }
                for _ in range(len(questions_answers) - len(answers))
            ])

        result = {
            "answers": answers,
            "overall_feedback": "",
            "aggregate": {}
        }

    return result
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

def convert_audio(input_file: str, output_file: str):
    subprocess.run([
        "ffmpeg",
        "-i", input_file,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_file
    ])

def transcribe_audio(file_path: str) -> str:
    """
    Convert speech audio to text using Whisper.
    """

    try:
        converted_path = "converted_audio.wav"

        # Convert browser audio (.webm etc) → wav 16khz mono
        convert_audio(file_path, converted_path)

        result = whisper_model.transcribe(
                 converted_path,
                 language="en",
                 fp16=False
             )

        return result["text"].strip()

    except Exception as e:
        print("[whisper] transcription error:", e)
        return ""


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
