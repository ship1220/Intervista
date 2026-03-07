# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Job PrepMate** — a FastAPI web application for AI-powered job interview preparation and course generation. Users select a job role/level, optionally upload a resume, then either take a voice-based mock interview with AI-generated questions and feedback, or generate a structured learning course. Uses Ollama (local) as primary LLM with Google Gemini as fallback.

## Build & Run Commands

```powershell
# Activate virtualenv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Initialize the SQLite database (creates jobprepmate.db)
python create_db.py

# Run the dev server
uvicorn main:app --reload
```

The app requires a `.env` file with at minimum:
```
GEMINI_API_KEY=<key>
```
Optional `.env` variables: `OLLAMA_URL` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `llama3`).

There is **no test suite** in this project currently.

## Architecture

### Request Flow
`main.py` is the single FastAPI entrypoint. All routes live there: auth (signup/login via cookie), interview workflow, resume upload, and course generation. Routes call into `ai_service.py` for LLM interaction and use prompt templates from `prompts.py`.

### AI Service Layer (`ai_service.py`)
- `generate_content(prompt, *, use_cache, json_mode)` — tries Ollama first, falls back to Gemini. Has an in-memory TTL cache (5 min, max 200 entries). When `json_mode=True`, Ollama is configured with `format: "json"` and a JSON-only system prompt to guarantee valid JSON output.
- `extract_json(text)` — robust JSON parser that strips markdown fences, fixes trailing commas, and finds outermost `{}`/`[]`. On failure, logs the **full** raw and cleaned text for debugging.
- `evaluate_answers(role, qa_list)` — legacy convenience wrapper that prompts the LLM and parses evaluation JSON.
- `analyze_speech_delivery(answer, duration)` — computes speaking pace, filler count, clarity score (0-100), engagement score (0-100).
- `compute_confidence_score(speech_analyses)` — aggregated confidence estimate from speech metrics.
- `evaluate_content(role, level, qa_list)` — async LLM-based content evaluation. Returns per-answer `score` and a single `feedback` paragraph (strengths + weaknesses + improved approach combined). Also returns `overall_feedback` which is used as the performance summary to avoid a second LLM call.
- `generate_performance_summary(report_data)` — async LLM-generated executive summary paragraph. Only called as a fallback when `evaluate_content` does not return `overall_feedback`.
- `compute_overall_score(content_avg, clarity_avg, engagement_avg)` — weighted formula: 50% content + 30% clarity + 20% engagement.
- `compute_recruiter_verdict(overall_score, role)` — returns SHORTLISTED/BORDERLINE/REJECT with suitable roles.

A reusable `httpx.AsyncClient` (`_get_ollama_client()`) is shared across Ollama requests to avoid per-request TCP handshake overhead. Gemini calls are run via `asyncio.run_in_executor` because the `google-generativeai` SDK is synchronous.

### Data Layer
- `database.py` — SQLAlchemy engine and `SessionLocal` bound to `sqlite:///./jobprepmate.db`.
- `models.py` — ORM models: `User`, `UserTarget`, `SkillProgress`, `InterviewSession`, `InterviewAttempt`, `Course`, `Chapter`, `Unit`, `QuizAttempt`. Not all models are actively used by routes yet (e.g. `Course`, `Chapter`, `Unit`, `QuizAttempt` are defined but not wired up).
- DB dependency injected via `get_db()` generator in `main.py`.

### In-Memory State (not persisted across restarts)
- `interview_sessions` — maps username → current interview session (questions, answers, role, level).
- `resume_store` — maps username → extracted resume text.
- `report_store` — maps username → latest interview performance report dict.

### Frontend
Jinja2 templates in `templates/` with inline `<script>` blocks (no build step, no JS framework). Key interactions:
- **Interview page** (`interview.html`): uses Web Speech API (`SpeechRecognition` for STT, `SpeechSynthesisUtterance` for TTS) to conduct voice interviews. Fetches questions from `/api/interview/questions`, posts answers to `/api/interview/evaluate`.
- **Course page** (`course.html`): connects to `/api/course/stream` via Server-Sent Events (SSE) to progressively render course modules as the AI generates them.

### Prompt Engineering (`prompts.py`)
All LLM prompts are centralized here. Each function returns a formatted string. Prompts instruct the LLM to return pure JSON (no markdown fences), but `extract_json` handles non-compliance. When modifying prompts, keep the JSON schema examples in sync with the parsing/normalization logic in `main.py`.

**Feedback format**: `content_analysis_prompt` requests a single `feedback` paragraph per question that combines strengths, weaknesses, and what a stronger answer would look like. Do **not** split these into separate fields (`strengths`, `weaknesses`, `ideal_answer`).

## Key Patterns to Follow

- **AI response normalization**: `main.py` normalizes variant key names from AI responses (e.g. `evaluations` → `feedback_per_question`, `better_answer` → `improved_answer`). When adding new AI-powered features, include similar normalization and fallback defaults.
- **json_mode**: All `generate_content` calls that expect JSON output should pass `json_mode=True`. This enables Ollama’s `format: "json"` constraint and adjusts the system prompt.
- **Fallback data**: Every AI call has a hardcoded fallback if parsing fails, so the UI always gets valid data.
- **Auth**: Cookie-based (`request.cookies.get("user")`), no JWT/session middleware. `get_current_user()` returns `None` for unauthenticated requests; routes redirect to `/login` or raise 401.
- **Passwords**: Truncated to 72 chars (bcrypt limit) before hashing.
