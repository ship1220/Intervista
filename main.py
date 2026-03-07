# main.py — Refactored async FastAPI application

import io
import json
import re
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Form, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from sse_starlette.sse import EventSourceResponse

from database import SessionLocal
from models import User, InterviewAttempt, SkillProgress
from ai_service import (
    generate_content, extract_json, evaluate_answers,
    analyze_speech_delivery, compute_confidence_score,
    evaluate_content, generate_performance_summary,
    compute_overall_score, compute_recruiter_verdict,
)
from prompts import (
    interview_questions_prompt,
    interview_questions_with_resume_prompt,
    batch_evaluation_prompt,
    course_outline_prompt,
    course_module_detail_prompt,
)

# ===========================================================================
# APP INIT
# ===========================================================================
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory interview session store (swap for Redis in production)
interview_sessions: dict[str, dict] = {}

# In-memory resume text store (swap for Redis/DB in production)
resume_store: dict[str, str] = {}

# In-memory report store (latest report per user)
report_store: dict[str, dict] = {}

# ===========================================================================
# DATABASE DEPENDENCY
# ===========================================================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===========================================================================
# SPEECH ANALYSIS HELPERS
# ===========================================================================
def analyze_speech_metrics(answer: str, duration_seconds: int) -> dict:
    """Analyze speech delivery metrics from answer text and duration."""
    words = answer.split()
    word_count = len(words)
    duration_minutes = duration_seconds / 60.0
    
    # Speaking pace (words per minute)
    wpm = word_count / duration_minutes if duration_minutes > 0 else 0
    
    # Filler word detection
    filler_words = ["um", "uh", "like", "you know", "so", "well", "actually", "basically", "literally", "totally"]
    filler_count = sum(1 for word in words if word.lower().strip('.,!?') in filler_words)
    filler_rate = filler_count / word_count if word_count > 0 else 0
    
    # Sentence clarity (simple metrics)
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    
    # Clarity score (arbitrary, higher is better)
    clarity_score = min(10, max(0, 10 - (avg_words_per_sentence - 15) * 0.5 - filler_rate * 20))
    
    return {
        "word_count": word_count,
        "duration_seconds": duration_seconds,
        "speaking_pace_wpm": round(wpm, 1),
        "filler_word_count": filler_count,
        "filler_rate": round(filler_rate, 3),
        "sentence_count": sentence_count,
        "avg_words_per_sentence": round(avg_words_per_sentence, 1),
        "sentence_clarity_score": round(clarity_score, 1)
    }

# ===========================================================================
# AUTH HELPERS
# ===========================================================================
def get_current_user(request: Request, db: Session):
    username = request.cookies.get("user")
    if not username:
        return None
    return db.query(User).filter(User.username == username).first()

def hash_password(password: str):
    password = password[:72]  # bcrypt limit
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str):
    try:
        return pwd_context.verify(plain[:72], hashed)
    except Exception:
        return False

# ===========================================================================
# PAGE ROUTES
# ===========================================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
def signup(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        return templates.TemplateResponse("signup.html", {"request": request, "message": "User already exists"})
    hashed_password = hash_password(password)
    user = User(username=username, password=hashed_password)
    db.add(user)
    db.commit()
    return RedirectResponse("/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()

    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "message": "Invalid credentials"}
        )

    if not verify_password(password, user.password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "message": "Invalid credentials"}
        )

    resp = RedirectResponse("/index", status_code=303)
    resp.set_cookie(key="user", value=username, httponly=True)
    return resp

@app.get("/logout")
def logout():
    resp = RedirectResponse("/")
    resp.delete_cookie("user")
    return resp

@app.get("/index", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login")
    return templates.TemplateResponse("index.html", {"request": request, "username": user.username})

@app.get("/progress", response_class=HTMLResponse)
def progress_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login")
    skills = db.query(SkillProgress).filter(SkillProgress.user_id == user.id).all()
    return templates.TemplateResponse("progress.html", {"request": request, "username": user.username, "skills": skills})

# ===========================================================================
# RESUME UPLOAD
# ===========================================================================
@app.post("/api/upload_resume")
async def upload_resume(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    content = await file.read()
    filename = (file.filename or "").lower()
    text = ""

    if filename.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif filename.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(content))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from the file")

    resume_store[user.username] = text
    print(f"[resume] Stored {len(text)} chars for user {user.username}")
    return {"status": "ok", "length": len(text), "preview": text[:200]}

# ===========================================================================
# INTERVIEW — START (returns page for voice interview)
# ===========================================================================
@app.post("/start_interview/", response_class=HTMLResponse)
def start_interview_page(request: Request, role: str = Form(...), level: str = Form(...), db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    return templates.TemplateResponse("interview.html", {
        "request": request,
        "username": user.username,
        "role": role,
        "level": level,
    })

# ===========================================================================
# INTERVIEW API — Generate questions (called by JS after page loads)
# ===========================================================================
@app.get("/api/interview/questions")
async def api_interview_questions(request: Request, role: str, level: str, count: int = 5, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    resume_text = resume_store.get(user.username, "")
    if resume_text:
        prompt = interview_questions_with_resume_prompt(role, level, resume_text, count)
    else:
        prompt = interview_questions_prompt(role, level, count)
    raw = await generate_content(prompt, use_cache=False, json_mode=True)

    try:
        questions = extract_json(raw)
        if not isinstance(questions, list):
            raise ValueError("Expected a list")
    except Exception:
        # Fallback: split by newlines and clean up
        questions = [q.strip().lstrip("0123456789.)- ") for q in raw.strip().split("\n") if q.strip()]
        questions = [q for q in questions if q.endswith("?")][:count]

    if not questions:
        questions = [
            "Tell me about yourself and your relevant experience.",
            "What is your greatest professional achievement?",
            "How do you handle tight deadlines?",
            "Describe a challenging problem you solved recently.",
            "Where do you see yourself in five years?",
        ]

    # Store session
    interview_sessions[user.username] = {
        "role": role,
        "level": level,
        "questions": questions,
        "answers": [],
    }

    return {"questions": questions}

# ===========================================================================
# INTERVIEW API — Batch evaluate all answers after interview ends
# ===========================================================================
@app.post("/api/interview/evaluate")
async def api_interview_evaluate(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    body = await request.json()
    
    if "questions_answers" in body:
        # ---------------------------------------------------------------
        # Full analysis pipeline (Features 1-8)
        # ---------------------------------------------------------------
        questions_answers = body.get("questions_answers", [])
        if not questions_answers:
            raise HTTPException(status_code=400, detail="No questions_answers provided")

        session = interview_sessions.get(user.username, {})
        role = body.get("role", session.get("role", "Software Developer"))
        level = body.get("level", session.get("level", "mid"))
        n = len(questions_answers)

        # -- FEATURE 2: Speech Delivery Analysis -----------------------
        speech_analyses: list[dict] = []
        for qa in questions_answers:
            # Compute duration from timestamps when available
            duration = float(qa.get("duration", 60))
            start_time = qa.get("start_time", "")
            end_time = qa.get("end_time", "")
            if start_time and end_time:
                try:
                    st = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    et = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                    duration = max((et - st).total_seconds(), 1.0)
                except Exception:
                    pass
            speech_analyses.append(
                analyze_speech_delivery(qa.get("answer", ""), duration)
            )

        # Aggregate voice metrics
        avg_pace = sum(a["speaking_pace_wpm"] for a in speech_analyses) / max(n, 1)
        total_fillers = sum(a["filler_word_count"] for a in speech_analyses)
        avg_clarity = sum(a["clarity_score"] for a in speech_analyses) / max(n, 1)
        avg_engagement = sum(a["engagement_score"] for a in speech_analyses) / max(n, 1)

        # -- FEATURE 4: Confidence Score -------------------------------
        confidence = compute_confidence_score(speech_analyses)

        # -- FEATURE 3: Content Analysis (LLM) ------------------------
        content_result = await evaluate_content(role, level, questions_answers)
        content_answers = content_result.get("answers", [])
        aggregate = content_result.get("aggregate", {})

        content_scores = [a.get("score", 50) for a in content_answers]
        content_avg = sum(content_scores) / max(len(content_scores), 1)

        # -- FEATURE 6: Overall Score & Verdict ------------------------
        overall = compute_overall_score(content_avg, avg_clarity, avg_engagement)
        verdict = compute_recruiter_verdict(overall, role)

        # -- FEATURE 4 (cont.): Session Metadata -----------------------
        total_duration = sum(a["duration_seconds"] for a in speech_analyses)

        # -- FEATURE 1: Candidate Profile ------------------------------
        candidate_profile = {
            "role": role,
            "level": level,
            "interview_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "total_questions": n,
        }

        # -- FEATURE 7: Detailed Answer Report -------------------------
        detailed_answers: list[dict] = []
        for i, qa in enumerate(questions_answers):
            ca = content_answers[i] if i < len(content_answers) else {}
            sa = speech_analyses[i] if i < len(speech_analyses) else {}
        detailed_answers.append({
                "question": qa.get("question", ""),
                "transcript": qa.get("answer", ""),
                "score": ca.get("score", 50),
                "feedback": ca.get("feedback", ""),
                "voice_metrics": {
                    "speaking_pace_wpm": sa.get("speaking_pace_wpm", 0),
                    "filler_count": sa.get("filler_word_count", 0),
                    "filler_words_found": sa.get("filler_words_found", []),
                    "clarity": sa.get("clarity_score", 0),
                    "engagement": sa.get("engagement_score", 0),
                    "word_count": sa.get("word_count", 0),
                },
            })

        # -- FEATURE 8: Assemble Report --------------------------------
        report = {
            "candidate_profile": candidate_profile,
            "overall_score": overall,
            "verdict": verdict,
            "performance_summary": "",  # filled below by LLM
            "voice_analysis": {
                "speaking_pace_wpm": round(avg_pace, 1),
                "total_filler_words": total_fillers,
                "clarity_score": round(avg_clarity),
                "engagement_score": round(avg_engagement),
                "confidence_score": confidence,
                "per_answer": speech_analyses,
            },
            "content_analysis": {
                "average_score": round(content_avg),
                "relevance_score": aggregate.get("relevance_score", round(content_avg)),
                "depth_score": aggregate.get("depth_score", round(content_avg * 0.9)),
                "star_method_score": aggregate.get("star_method_score", round(content_avg * 0.7)),
            },
            "detailed_answers": detailed_answers,
            "session_metadata": {
                "total_duration_seconds": round(total_duration, 1),
                "average_speaking_pace_wpm": round(avg_pace, 1),
                "total_filler_words": total_fillers,
                "confidence_score": confidence,
            },
        }

        # -- FEATURE 5: Performance Summary ----------------------------
        # Use overall_feedback from content evaluation to avoid a second LLM call.
        # Fall back to a separate LLM call only when the evaluator didn't provide one.
        overall_feedback = content_result.get("overall_feedback", "")
        if overall_feedback and overall_feedback.strip():
            report["performance_summary"] = overall_feedback.strip()
        else:
            summary = await generate_performance_summary(report)
            report["performance_summary"] = summary

        # Store report for the /report page
        report_store[user.username] = report

        # Persist answers to DB
        for i, qa in enumerate(questions_answers):
            fb = detailed_answers[i].get("feedback", "") if i < len(detailed_answers) else ""
            attempt = InterviewAttempt(
                user_id=user.id,
                role=role,
                topic="voice-interview",
                difficulty=level,
                answer=qa.get("answer", ""),
                feedback=fb,
            )
            db.add(attempt)

        # Update skill progress
        skill = (
            db.query(SkillProgress)
            .filter(SkillProgress.user_id == user.id, SkillProgress.skill == role)
            .first()
        )
        if not skill:
            skill = SkillProgress(
                user_id=user.id, skill=role, attempts=1, weak=overall < 50
            )
            db.add(skill)
        else:
            skill.attempts += 1
            skill.weak = overall < 50

        db.commit()
        return report
    
    else:
        # Old format
        role = body.get("role", "")
        answers = body.get("answers", [])  # [{question, answer}, ...]

        if not answers:
            raise HTTPException(status_code=400, detail="No answers provided")

        # Store answers in session
        session = interview_sessions.get(user.username, {})
        session["answers"] = answers

        # Generate batch evaluation
        prompt = batch_evaluation_prompt(role, answers)
        raw = await generate_content(prompt, use_cache=False, json_mode=True)

        print(f"[evaluate] raw AI response length: {len(raw)}")
        print(f"[evaluate] raw AI response (first 500 chars): {raw[:500]}")

        evaluation = None
        try:
            evaluation = extract_json(raw)
            print(f"[evaluate] parsed JSON keys: {list(evaluation.keys()) if isinstance(evaluation, dict) else type(evaluation)}")
        except Exception as exc:
            print(f"[evaluate] JSON parse error: {exc}")

        # Normalize top-level keys (AI may use variant names)
        if isinstance(evaluation, dict):
            for alt in ("evaluations", "feedback", "responses", "results"):
                if alt in evaluation and "feedback_per_question" not in evaluation:
                    evaluation["feedback_per_question"] = evaluation.pop(alt)
                    break
            for alt in ("tips", "suggestions", "general_tips"):
                if alt in evaluation and "improvement_tips" not in evaluation:
                    evaluation["improvement_tips"] = evaluation.pop(alt)
                    break
            for alt in ("resources", "recommended_resources", "study_resources"):
                if alt in evaluation and "learning_resources" not in evaluation:
                    evaluation["learning_resources"] = evaluation.pop(alt)
                    break

        # Validate and build fallback if parsing failed or structure is wrong
        if not isinstance(evaluation, dict) or "feedback_per_question" not in evaluation:
            evaluation = {
                "feedback_per_question": [
                    {
                        "question": qa.get("question", ""),
                        "candidate_answer": qa.get("answer", ""),
                        "feedback": "AI evaluation could not be parsed. Please try again.",
                        "improved_answer": "",
                    }
                    for qa in answers
                ],
                "improvement_tips": [
                    "Practice structuring your answers with concrete examples.",
                    "Use the STAR method (Situation, Task, Action, Result) for behavioral questions.",
                    "Research the company and role thoroughly before interviews.",
                ],
                "learning_resources": [{"topic": "Interview Preparation", "resource": "Practice common behavioral and technical questions for your role."}],
            }

        # Normalize per-question field names (AI may use variants)
        for i, fb in enumerate(evaluation.get("feedback_per_question", [])):
            qa = answers[i] if i < len(answers) else {}
            for src, dst in [
                ("answer", "candidate_answer"),
                ("user_answer", "candidate_answer"),
                ("evaluation", "feedback"),
                ("comment", "feedback"),
                ("better_answer", "improved_answer"),
                ("suggested_answer", "improved_answer"),
                ("sample_answer", "improved_answer"),
                ("ideal_answer", "improved_answer"),
            ]:
                if src in fb and dst not in fb:
                    fb[dst] = fb.pop(src)
            fb.setdefault("question", qa.get("question", ""))
            fb.setdefault("candidate_answer", qa.get("answer", ""))
            fb.setdefault("feedback", "")
            fb.setdefault("improved_answer", "")

        # Normalize learning_resources to always be list of {topic, resource}
        raw_resources = evaluation.get("learning_resources", [])
        normalized = []
        for r in raw_resources:
            if isinstance(r, dict):
                topic = r.get("topic") or r.get("name") or r.get("title") or ""
                resource = r.get("resource") or r.get("description") or r.get("link") or r.get("url") or ""
                normalized.append({"topic": topic, "resource": resource})
            elif isinstance(r, str):
                normalized.append({"topic": r, "resource": ""})
        evaluation["learning_resources"] = normalized

        # Persist each answer to DB
        for i, qa in enumerate(answers):
            fb_list = evaluation.get("feedback_per_question", [])
            fb_text = fb_list[i].get("feedback", "") if i < len(fb_list) else ""
            attempt = InterviewAttempt(
                user_id=user.id,
                role=role,
                topic="voice-interview",
                difficulty="adaptive",
                answer=qa.get("answer", ""),
                feedback=fb_text,
            )
            db.add(attempt)

        # Update skill progress
        skill = db.query(SkillProgress).filter(SkillProgress.user_id == user.id, SkillProgress.skill == role).first()
        if not skill:
            skill = SkillProgress(user_id=user.id, skill=role, attempts=1, weak=False)
            db.add(skill)
        else:
            skill.attempts += 1

        db.commit()

        return evaluation
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    return templates.TemplateResponse("course.html", {
        "request": request,
        "role": role,
        "level": level,
    })

# ===========================================================================
# REPORT PAGE
# ===========================================================================
@app.get("/report", response_class=HTMLResponse)
def report_page(request: Request, db: Session = Depends(get_db)):
    """Render the full interview performance report."""
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login")
    report = report_store.get(user.username)
    if not report:
        return RedirectResponse("/index")
    return templates.TemplateResponse("report.html", {
        "request": request,
        "username": user.username,
        "report": report,
    })

@app.get("/api/course/stream")
async def api_course_stream(request: Request, role: str, level: str, db: Session = Depends(get_db)):
    """Server-Sent Events endpoint for staged course generation."""
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    async def event_generator():
        # Stage 1: Outline
        yield {"event": "status", "data": "Generating course outline..."}

        outline_raw = await generate_content(course_outline_prompt(role, level), json_mode=True)
        outline = None
        try:
            outline = extract_json(outline_raw)
        except Exception as exc:
            print(f"[course] outline parse error: {exc}")

        if not isinstance(outline, dict) or not outline.get("modules"):
            outline = {
                "course_title": f"{level.title()} {role.title()} Course",
                "course_description": f"A comprehensive course for {level} {role} professionals.",
                "modules": [
                    {"title": f"{role.title()} Fundamentals", "description": f"Core foundational concepts for {role}."},
                    {"title": f"Intermediate {role.title()} Skills", "description": f"Building on the basics for {role}."},
                    {"title": f"Advanced {role.title()} Topics", "description": f"Deep-dive into complex {role} areas."},
                    {"title": f"{role.title()} Projects & Practice", "description": f"Hands-on application for {role}."},
                ],
            }

        # Normalize: the prompt asks for course_title/course_description
        # but the frontend also accepts title/description
        outline.setdefault("title", outline.get("course_title", ""))
        outline.setdefault("description", outline.get("course_description", ""))

        yield {"event": "outline", "data": json.dumps(outline)}

        # Stage 2 & 3: Module details (one at a time)
        modules = outline.get("modules", [])
        for i, mod in enumerate(modules):
            if await request.is_disconnected():
                return

            yield {"event": "status", "data": f"Generating module {i + 1}/{len(modules)}: {mod['title']}..."}

            detail_raw = await generate_content(
                course_module_detail_prompt(role, level, mod["title"]),
                json_mode=True,
            )
            detail = None
            try:
                detail = extract_json(detail_raw)
            except Exception as exc:
                print(f"[course] module {i} parse error: {exc}")

            if not isinstance(detail, dict) or "module_title" not in detail:
                detail = {
                    "module_title": mod["title"],
                    "overview": detail_raw[:500] if detail_raw else "Content generation in progress.",
                    "concepts": [],
                    "lessons": [],
                    "exercises": [],
                    "quiz": [],
                    "project": None,
                    "resources": [],
                }

            yield {"event": "module_detail", "data": json.dumps({"index": i, **detail})}

        yield {"event": "done", "data": "Course generation complete!"}

    return EventSourceResponse(event_generator())
