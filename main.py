# main.py — Refactored async FastAPI application

import io
import json
import re
import time
from pathlib import Path
import models
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, Form, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from ai_service import transcribe_audio
from sse_starlette.sse import EventSourceResponse
from database import Base, engine
Base.metadata.create_all(bind=engine)
import os
import shutil
UPLOAD_DIR = "uploads"
from database import SessionLocal
from models import (
    User,
    InterviewAttempt,
    SkillProgress,
    UserProfile,
    Interview,
    UserSkillProfileRow,
)
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
from prompts import resume_skill_profile_prompt
from user_skill_profile import (
    BasicUserInfo,
    ResumeData,
    UserSkillProfile,
    UserSkillVector,
    InterviewRecord,
    ScoreBreakdown,
    create_user_profile,
    detect_weaknesses,
    recommend_micro_courses,
    update_skill_score,
    update_skill_vector,
    calculate_overall_score,
    record_interview_result,
)

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def categorize_weak_topics(weak_topics: list[str]) -> dict[str, list[str]]:

    technical = []
    communication = []

    for topic in weak_topics:
        t = topic.lower()

        if any(k in t for k in [
            "sql","database","python","algorithm","data structure",
            "machine learning","statistics","api","system design"
        ]):
            technical.append(topic)

        elif any(k in t for k in [
            "explain","clarity","structure","communication",
            "example","confidence","detail","depth"
        ]):
            communication.append(topic)

        else:
            # fallback
            technical.append(topic)

    return {
        "Technical Skills": list(set(technical)),
        "Communication & Answer Quality": list(set(communication))
    }

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


def get_or_create_user_profile(db: Session, user: User) -> UserProfile:
    """
    Ensure a UserProfile row exists for the given auth user.

    This keeps creation logic in one place and can be reused from
    login, resume upload, and interview flows.
    """
    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    if profile:
        return profile

    profile = UserProfile(
        user_id=user.id,
        email=user.username,  # treat username as email by default
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


# ---------------------------------------------------------------------------
# User Skill Profile helpers (DB-backed skill vector)
# ---------------------------------------------------------------------------

def get_skill_profile(db: Session, user_id: int):
    """Fetch the user skill profile row, or None if it doesn't exist."""
    return db.query(UserSkillProfileRow).filter(UserSkillProfileRow.user_id == user_id).first()


def create_skill_profile(db: Session, user_id: int) -> UserSkillProfileRow:
    """Create a new skill profile row for a user (no-op if already exists)."""
    profile = get_skill_profile(db, user_id)
    if profile:
        return profile

    profile = UserSkillProfileRow(
        user_id=user_id,
        technical_skills={},
        interview_skills={},
        communication_skills={},
        overall_score=0.0,
        interview_count=0,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def update_skill_profile(db: Session, user_id: int, skill_data: dict) -> UserSkillProfileRow:
    """Update an existing skill profile row with the provided skill data."""
    profile = get_skill_profile(db, user_id)
    if not profile:
        profile = create_skill_profile(db, user_id)

    # Update fields if provided
    if "technical_skills" in skill_data:
        profile.technical_skills = skill_data["technical_skills"]
    if "interview_skills" in skill_data:
        profile.interview_skills = skill_data["interview_skills"]
    if "communication_skills" in skill_data:
        profile.communication_skills = skill_data["communication_skills"]
    if "overall_score" in skill_data:
        profile.overall_score = float(skill_data["overall_score"])
    if "interview_count" in skill_data:
        profile.interview_count = int(skill_data["interview_count"])

    profile.last_updated = datetime.utcnow()

    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile

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

    # Ensure a profile row exists for this user.
    get_or_create_user_profile(db, user)

    resp = RedirectResponse("/index", status_code=303)
    resp.set_cookie(key="user", value=username, httponly=True)
    return resp

@app.get("/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    resp = RedirectResponse("/")
    resp.delete_cookie("user")
    return resp

@app.post("/update-resume")
def update_resume(request: Request, file: UploadFile, db: Session = Depends(get_db)):

    user = get_current_user(request, db)

    if not user:
        return RedirectResponse("/login", status_code=303)

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_location = os.path.join(UPLOAD_DIR, f"{user.id}_{file.filename}")

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()

    if profile:
        profile.resume_file_path = file_location
        db.commit()

    return RedirectResponse("/profile", status_code=303)

@app.get("/index", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):

    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login")

    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()

    role = profile.role_applied_for if profile else ""
    level = profile.current_designation if profile else ""

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "username": user.username,
            "saved_role": role,
            "saved_level": level,
        },
    )

@app.get("/progress", response_class=HTMLResponse)
def progress_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login")
    skills = db.query(SkillProgress).filter(SkillProgress.user_id == user.id).all()
    return templates.TemplateResponse("progress.html", {"request": request, "username": user.username, "skills": skills})


@app.get("/progress/", response_class=HTMLResponse)
def progress_page_slash(request: Request, db: Session = Depends(get_db)):
    """Support both /progress and /progress/ URLs."""
    return progress_page(request, db)

# ===========================================================================
# RESUME UPLOAD
# ===========================================================================
@app.post("/api/upload_resume")
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    role: str = Form(...),
    level: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    content = await file.read()
    original_name = file.filename or "resume"
    filename = original_name.lower()
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

    # Persist raw text in memory for question generation
    resume_store[user.username] = text

    # Persist file to disk for later download from profile
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"user_{user.id}_{int(time.time())}_{original_name}"
    file_path = upload_dir / safe_name
    with open(file_path, "wb") as f:
        f.write(content)

    # Create / update structured skill profile using the resume
    profile_row = get_or_create_user_profile(db, user)

    # Ask LLM for compact skill profile
    try:
        from ai_service import generate_content, extract_json

        prompt = resume_skill_profile_prompt(text, role, level)
        raw = await generate_content(prompt, use_cache=False, json_mode=True)
        parsed = extract_json(raw)
    except Exception as exc:
        print("[resume] skill profile generation error:", exc)
        parsed = {}

    skills = parsed.get("skills") if isinstance(parsed, dict) else None
    if not isinstance(skills, list):
        skills = []
    skills = [str(s).strip() for s in skills if str(s).strip()]

    skill_gaps = parsed.get("skill_gaps") if isinstance(parsed, dict) else None
    if not isinstance(skill_gaps, list):
        skill_gaps = []
    skill_gaps = [str(s).strip() for s in skill_gaps if str(s).strip()]

    improvement_suggestions = (
        parsed.get("improvement_suggestions") if isinstance(parsed, dict) else ""
    ) or ""

    strength_pct = 0.0
    if isinstance(parsed, dict):
        try:
            strength_pct = float(parsed.get("strength_percentage", 0.0))
        except (TypeError, ValueError):
            strength_pct = 0.0

    # Build Pydantic UserSkillProfile
    basic_info = BasicUserInfo(
        user_id=str(user.id),
        name=user.username,
        target_role=role,
        experience_level=level,
        resume_file_path=str(file_path),
    )
    resume_data = ResumeData(skills=skills)
    profile_obj: UserSkillProfile = create_user_profile(basic_info, resume_data)

    # Detect weaknesses based on initial profile (may be empty at this stage)
    detect_weaknesses(profile_obj)

    profile_row.role_applied_for = role
    profile_row.current_designation = level
    profile_row.resume_file_path = str(file_path)
    profile_row.extracted_skills = json.dumps(skills)
    profile_row.skill_strength_percentage = strength_pct
    profile_row.skill_gaps = json.dumps(skill_gaps)
    profile_row.improvement_suggestions = improvement_suggestions
    profile_row.profile_json = profile_obj.json()
    profile_row.updated_at = datetime.utcnow()

    db.add(profile_row)
    db.commit()

    print(f"[resume] Stored {len(text)} chars for user {user.username}")
    return {
        "status": "ok",
        "length": len(text),
        "preview": text[:200],
        "skills": skills,
        "skill_gaps": skill_gaps,
        "strength_percentage": strength_pct,
    }

# ===========================================================================
# AUDIO TRANSCRIPTION (Whisper)
# ===========================================================================
@app.post("/api/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """
    Receive recorded audio from the frontend and convert it to text using Whisper.
    """

    audio_bytes = await file.read()

    temp_path = "temp_audio_" + str(datetime.now().timestamp()) + ".wav"

    # Save audio temporarily
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    # Run Whisper transcription
    text = transcribe_audio(temp_path)

    return {"transcript": text}
# ===========================================================================
# INTERVIEW — START (returns page for voice interview)
# ===========================================================================
@app.post("/start_interview/", response_class=HTMLResponse)
def start_interview_page(
    request: Request,
    role: str = Form(...),
    level: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    return templates.TemplateResponse(
        "interview.html",
        {
            "request": request,
            "username": user.username,
            "role": role,
            "level": level,
        },
    )
# ===========================================================================
# INTERVIEW API — Generate questions (called by JS after page loads)
# ===========================================================================
@app.get("/api/interview/questions")
async def api_interview_questions(
    request: Request,
    role: str,
    level: str,
    count: int = 5,
    db: Session = Depends(get_db)
):

    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    resume_text = resume_store.get(user.username, "")

    if resume_text:
        prompt = interview_questions_with_resume_prompt(role, level, resume_text, count)
    else:
        prompt = interview_questions_prompt(role, level, count)

    raw = await generate_content(prompt, use_cache=False, json_mode=True)

    raw = raw.strip()

    print(f"[questions] Raw response length: {len(raw) if raw else 0}")
    if raw:
        print(f"[questions] Raw preview: {raw[:300]}")

    # Fix truncated JSON arrays
    if raw.startswith("[") and not raw.endswith("]"):
        raw += "]"

    questions = []

    if not raw:
        print("[questions] Empty AI response — using fallback")

    else:
        try:
            questions_data = extract_json(raw)
            print(f"[questions] Extracted type: {type(questions_data).__name__}")

            if isinstance(questions_data, list):
                questions = questions_data

            elif isinstance(questions_data, dict):

                if "questions" in questions_data:
                    questions = questions_data["questions"]

                elif "question" in questions_data:
                    questions = questions_data["question"]

                else:
                    for key in questions_data:
                        if isinstance(questions_data[key], list):
                            questions = questions_data[key]
                            break

            cleaned_questions = []

            if isinstance(questions, list):

                for q in questions:

                    if isinstance(q, str):
                        cleaned_questions.append(q)

                    elif isinstance(q, dict):

                        if "question" in q:
                            cleaned_questions.append(str(q["question"]))

                        elif "text" in q:
                            cleaned_questions.append(str(q["text"]))

            questions = cleaned_questions[:count]

            print(f"[questions] Extracted {len(questions)} questions")

        except Exception as e:

            print(f"[questions] JSON extraction failed: {e}")

            matches = re.findall(r'"question"\s*:\s*"([^"]+)"', raw)

            if matches:
                questions = matches[:count]
                print(f"[questions] Extracted {len(questions)} questions using regex fallback")

            else:
                questions = [
                    line.strip().lstrip("0123456789.)- ")
                    for line in raw.split("\n")
                    if line.strip().endswith("?")
                ][:count]

    if not questions:
        questions = [
            "Tell me about yourself and your relevant experience.",
            "What is your greatest professional achievement?",
            "How do you handle tight deadlines?",
            "Describe a challenging problem you solved recently.",
            "Where do you see yourself in five years?",
        ][:count]

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

        print(f"[evaluate] Received {len(questions_answers)} question-answer pairs")
        for i, qa in enumerate(questions_answers):
            print(f"[evaluate] Q{i+1}: {qa.get('question', 'MISSING')[:50]}...")

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
        print(f"[evaluate] content_result keys: {list(content_result.keys())}")
        content_answers = content_result.get("answers", [])
        print(f"[evaluate] content_answers count: {len(content_answers)}")
        aggregate = content_result.get("aggregate", {})

        content_scores = [a.get("score", 50) for a in content_answers]
        content_avg = sum(content_scores) / max(len(content_scores), 1)

        # -- FEATURE 6: Overall Score & Verdict ------------------------
        overall = compute_overall_score(content_avg, avg_clarity, avg_engagement,questions_answers)
        verdict = compute_recruiter_verdict(overall, role)

        # -- FEATURE 4 (cont.): Session Metadata -----------------------
        total_duration = sum(a["duration_seconds"] for a in speech_analyses)

        # -- FEATURE 1: Candidate Profile ------------------------------
        candidate_profile = {
            "role": role,
            "level": level,
            "interview_date": datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M IST"),
            "total_questions": n,
        }
        # -- FEATURE 7: Detailed Answer Report -------------------------

        detailed_answers: list[dict] = []

        for i, qa in enumerate(questions_answers):

            ca = content_answers[i] if i < len(content_answers) else {}
            sa = speech_analyses[i] if i < len(speech_analyses) else {}

            # Ensure question text is from input
            question_text = qa.get("question", "")
            answer_text = qa.get("answer", "")

            print(f"[evaluate] Question {i+1}: {question_text[:50]}...")

            # Normalize strengths
            strengths = ca.get("strengths", [])

            if isinstance(strengths, str):
                strengths = [strengths]

            if not strengths:
                strengths = ["Answer attempted."]

            # Normalize weaknesses
            weaknesses = ca.get("weaknesses", [])

            if isinstance(weaknesses, str):
                weaknesses = [weaknesses]

            if not weaknesses:
                weaknesses = ["Needs improvement."]

            detailed_answers.append({
                "question": question_text,
                "transcript": answer_text,
                "score": ca.get("score", 50),
                "strengths": strengths,
                "weaknesses": weaknesses,
                "ideal_answer": ca.get("ideal_answer", "No ideal answer generated."),
                "weak_topics": ca.get("weak_topics", []),
               "voice_metrics": {
    "speaking_pace_wpm": sa.get("speaking_pace_wpm", 0),
    "filler_word_count": sa.get("filler_word_count", 0),
    "clarity_score": sa.get("clarity_score", 0),
    "engagement_score": sa.get("engagement_score", 0),
    "word_count": sa.get("word_count", 0),
}
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
                "total_duration_seconds": round(total_duration, 1),
                "per_answer": speech_analyses,
            },
            "content_analysis": {
                "average_score": round(content_avg),

                # values used by report.html
                "relevance_score": aggregate.get("technical_score", round(content_avg)),
                "depth_score": aggregate.get("communication_score", round(content_avg * 0.85)),
                "star_method_score": aggregate.get("overall_score", round(content_avg)),
             },
            "detailed_answers": detailed_answers,
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

        # Expose weak topics at the top level for report.html
        report["weak_topics"] = categorize_weak_topics(content_result.get("weak_topics", []))

        # Store report for the /report page (latest only)
        report_store[user.username] = report

        # Persist high-level interview record with full report JSON
        interview_row = Interview(
            user_id=user.id,
            role=role,
            date=datetime.utcnow(),
            score=overall,
            report_json=json.dumps(report),
        )
        db.add(interview_row)

        # Update rich UserSkillProfile based on this interview
        profile_row = get_or_create_user_profile(db, user)
        profile_obj: UserSkillProfile
        try:
            if profile_row.profile_json:
                profile_obj = UserSkillProfile.model_validate_json(profile_row.profile_json)
            else:
                # Minimal fallback profile if none existed yet
                basic_info = BasicUserInfo(
                    user_id=str(user.id),
                    name=user.username,
                    target_role=role,
                    experience_level=level,
                    resume_file_path=profile_row.resume_file_path or "",
                )
                resume_data = ResumeData(skills=[])
                profile_obj = create_user_profile(basic_info, resume_data)
        except Exception as exc:
            print("[profile] error loading existing profile:", exc)
            basic_info = BasicUserInfo(
                user_id=str(user.id),
                name=user.username,
                target_role=role,
                experience_level=level,
                resume_file_path=profile_row.resume_file_path or "",
            )
            resume_data = ResumeData(skills=[])
            profile_obj = create_user_profile(basic_info, resume_data)

        # Record interview per question into profile
        for i, qa in enumerate(questions_answers):
            ca = content_answers[i] if i < len(content_answers) else {}
            topic = role  # Treat role as the main topic/skill for now

            sb = ScoreBreakdown(
                correctness=float(ca.get("score", 50)),
                conceptual_depth=float(ca.get("score", 50)),
                clarity=float(avg_clarity),
                feedback="",
            )
            rec = InterviewRecord(
                question=qa.get("question", ""),
                topic=topic,
                answer_transcript=qa.get("answer", ""),
                evaluation_score=float(ca.get("score", 50)),
                score_breakdown=sb,
            )
            record_interview_result(profile_obj, rec)

        # Sync weak topics from LLM result
        weak_from_llm = []
        for topics in content_result.get("weak_topics", {}).values():
            weak_from_llm.extend(topics)
        for w in weak_from_llm:
            w_norm = w.strip().lower()
            if w_norm and w_norm not in {t.lower() for t in profile_obj.weak_topics}:
                profile_obj.weak_topics.append(w)

        # Recompute weaknesses & recommend micro-courses
        detect_weaknesses(profile_obj)
        recommend_micro_courses(profile_obj)

        # --------- Update skill vector based on interview metrics ---------
        def _int(v, default=0):
            try:
                return int(round(float(v)))
            except Exception:
                return default

        voice = report.get("voice_analysis", {})
        content = report.get("content_analysis", {})

        interview_metrics = {
            "relevance": _int(content.get("relevance_score", 0)),
            "explanation_depth": _int(content.get("depth_score", 0)),
            "problem_solving": _int(content.get("average_score", 0)),
            "structured_thinking": _int(content.get("average_score", 0)),
            "star_method": _int(content.get("star_method_score", 0)),

            "clarity": _int(voice.get("clarity_score", 0)),
            "confidence": _int(voice.get("confidence_score", 0)),
            "engagement": _int(voice.get("engagement_score", 0)),
            "speaking_pace": _int(voice.get("speaking_pace_wpm", 0)),
            "filler_control": 100 - min(100, _int(voice.get("total_filler_words", 0))),
}

        # Update the stored skill vector using the interview metrics
        skill_vector = UserSkillVector(
            technical_skills=profile_obj.technical_skills,
            interview_skills=profile_obj.interview_skills,
            communication_skills=profile_obj.communication_skills,
        )
        skill_vector = update_skill_vector(skill_vector, interview_metrics)

        profile_obj.technical_skills = skill_vector.technical_skills
        profile_obj.interview_skills = skill_vector.interview_skills
        profile_obj.communication_skills = skill_vector.communication_skills
        profile_obj.overall_score = calculate_overall_score(skill_vector)

        # Aggregate simple fields for quick display in profile page
        skills_list = [node.skill_name for node in profile_obj.skill_graph.values()]
        if profile_obj.skill_graph:
            avg_score = sum(n.score for n in profile_obj.skill_graph.values()) / len(
                profile_obj.skill_graph
            )
        else:
            avg_score = overall

        profile_row.role_applied_for = role
        profile_row.current_designation = level
        profile_row.extracted_skills = json.dumps(skills_list)
        profile_row.skill_strength_percentage = float(profile_obj.overall_score)
        profile_row.skill_gaps = json.dumps(profile_obj.weak_topics)
        # Use performance summary as latest improvement suggestions snapshot
        profile_row.improvement_suggestions = report.get("performance_summary", "")
        profile_row.profile_json = profile_obj.json()
        profile_row.updated_at = datetime.utcnow()

        # Persist a compact skill vector record for quick access / querying.
        update_skill_profile(
            db,
            user.id,
            {
                "technical_skills": profile_obj.technical_skills.dict(),
                "interview_skills": profile_obj.interview_skills.dict(),
                "communication_skills": profile_obj.communication_skills.dict(),
                "overall_score": profile_obj.overall_score,
                "interview_count": profile_obj.interview_count,
            },
        )

        # Return the updated skill profile as part of the report payload.
        report["skill_profile"] = profile_obj.dict()

        db.add(profile_row)

        # Persist answers to DB
        for i, qa in enumerate(questions_answers):
            attempt = InterviewAttempt(
                user_id=user.id,
                role=role,
                topic="voice-interview",
                difficulty=level,
                answer=qa.get("answer", ""),
                feedback="",
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
                        "feedback": "",
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
            attempt = InterviewAttempt(
                user_id=user.id,
                role=role,
                topic="voice-interview",
                difficulty="adaptive",
                answer=qa.get("answer", ""),
                feedback="",
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
    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "username": user.username,
            "report": report,
        },
    )


@app.get("/interview-report/{interview_id}", response_class=HTMLResponse)
def interview_report_page(
    request: Request,
    interview_id: int,
    db: Session = Depends(get_db),
):
    """
    View a past interview report stored in the `interviews` table.
    """
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login")

    interview = (
        db.query(Interview)
        .filter(Interview.id == interview_id, Interview.user_id == user.id)
        .first()
    )
    if not interview:
        return RedirectResponse("/profile")

    try:
        report = json.loads(interview.report_json)
    except Exception:
        report = {}

    return templates.TemplateResponse(
        "report.html",
        {"request": request, "username": user.username, "report": report},
    )


@app.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request, db: Session = Depends(get_db)):

    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login")

    profile_row = get_or_create_user_profile(db, user)

    # Default skill profile
    profile_data = {
        "technical_skills": {
            "dsa": 50,
            "dbms": 50,
            "operating_systems": 50,
            "system_design": 50,
        },
        "interview_skills": {
            "relevance": 50,
            "explanation_depth": 50,
            "structured_thinking": 50,
            "problem_solving": 50,
            "star_method": 50,
        },
        "communication_skills": {
            "clarity": 50,
            "confidence": 50,
            "engagement": 50,
            "speaking_pace": 50,
            "filler_control": 50,
        },
        "overall_score": 50
    }

    # Load stored skill profile
    if profile_row.profile_json:
        try:
            loaded = json.loads(profile_row.profile_json)
            profile_data.update(loaded)
        except Exception:
            pass

    # Extracted skills
    extracted_skills = []
    if profile_row.extracted_skills:
        try:
            extracted_skills = json.loads(profile_row.extracted_skills)
        except:
            extracted_skills = []

    # Skill gaps
    skill_gaps = []
    if profile_row.skill_gaps:
        try:
            skill_gaps = json.loads(profile_row.skill_gaps)
        except:
            skill_gaps = []

    # Interview history
    interviews = (
        db.query(Interview)
        .filter(Interview.user_id == user.id)
        .order_by(Interview.date.asc())
        .all()
    )

    history_by_role = {}
    for iv in interviews:
        history_by_role.setdefault(iv.role, []).append(iv)

    timeline_by_role = {}
    for role, ivs in history_by_role.items():
        timeline_by_role[role] = [
            {
                "label": iv.date.strftime("%Y-%m-%d"),
                "score": iv.score
            }
            for iv in ivs
        ]

    return templates.TemplateResponse(
        "profile.html",
        {
            "request": request,
            "username": user.username,
            "user_profile": profile_row,
            "skill_profile": profile_data,
            "extracted_skills": extracted_skills,
            "skill_gaps": skill_gaps,
            "interview_history_by_role": history_by_role,
            "timeline_by_role": timeline_by_role,
        },
    )

@app.get("/interview-history", response_class=HTMLResponse)
def interview_history_redirect(request: Request):
    """
    Simple semantic route that redirects to the profile page where
    interview history is rendered.
    """
    return RedirectResponse("/profile")


@app.post("/generate_course/", response_class=HTMLResponse)
def generate_course_page(
    request: Request,
    role: str = Form(...),
    level: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login")

    return templates.TemplateResponse(
        "course.html",
        {
            "request": request,
            "username": user.username,
            "role": role,
            "level": level,
        },
    )

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