# main.py

from fastapi import FastAPI, Request, Form, Depends, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from dotenv import load_dotenv
import requests
import os
import google.genai as genai

# avoid importing requests twice; it's already imported above
from database import SessionLocal
from models import User, InterviewAttempt, SkillProgress
import random
import whisper
import subprocess
import uuid

# ---------- SPEECH ANALYSIS HELPER  ----------
def analyze_speech(text: str, duration_sec: float):
    words = text.strip().split()
    word_count = len(words)

    minutes = duration_sec / 60 if duration_sec > 0 else 1
    wpm = int(word_count / minutes)

    fillers = ["um", "uh", "like", "you know", "actually"]
    filler_count = sum(text.lower().count(f) for f in fillers)

    analysis = {
        "word_count": word_count,
        "wpm": wpm,
        "filler_count": filler_count
    }

    return analysis
#------speech feedback from analysis------
def speech_feedback_from_analysis(analysis: dict):
    feedback = []

    wpm = analysis["wpm"]
    fillers = analysis["filler_count"]

 # Speaking speed
    if wpm < 90:
        feedback.append("You are speaking a bit slowly. Try to sound more confident and fluent.")
    elif wpm > 160:
        feedback.append("You are speaking too fast. Slow down slightly to improve clarity.")
    else:
        feedback.append("Your speaking speed is well balanced.")

    # Filler words
    if fillers > 5:
        feedback.append("You used several filler words. Try to reduce them to sound more professional.")
    else:
        feedback.append("Good control over filler words.")

    return " ".join(feedback)

# =======================
# ENV CONFIG
# =======================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# =======================
# APP INIT
# =======================
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# =======================
# DATABASE DEPENDENCY
# =======================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =======================
# SIMPLE SESSION STORE
# (Better → Redis / DB)
# =======================
interview_sessions = {}


# ---------------- SPEECH MODEL ----------------
whisper_model = whisper.load_model("base")

# =======================
# AUTH HELPERS
# =======================
def get_current_user(request: Request, db: Session):
    username = request.cookies.get("user")
    if not username:
        return None
    return db.query(User).filter(User.username == username).first()

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

# =======================
# AI HELPERS
# =======================
def generate_question(role: str, level: str, difficulty: str) -> str:
    prompt = f"""
You are a mock interviewer.
Role: {role}
Level: {level}
Difficulty: {difficulty}
Ask ONE realistic interview question.
Return ONLY the question.
""".strip()

    # Try Ollama
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=20
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print("Ollama error:", e)

    # Try Gemini
    try:
        if GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY)
            result = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            return result.text.strip()
    except Exception as e:
        print("Gemini error:", e)

    return "Tell me about yourself."

def generate_feedback(answer: str) -> str:
    prompt = f"""
Give constructive feedback for this answer:

"{answer}"

Also provide an improved answer in 3-4 lines.
""".strip()

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=20
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print("Feedback error:", e)

    return "Try structuring your answer clearly and include concrete examples."


def generate_course(role: str, level: str) -> str:
    """Generate an AI-created course roadmap for a given role and level."""
    prompt = f"""
You are an AI curriculum designer. Create a clean, structured course roadmap
for someone aiming to become a {role} at the {level} level. Include the
following sections:

Career Overview
Required Skills
Course Modules (with topics inside each module)
Tools Required
3 Beginner Projects
2 Intermediate Projects
Recommended Resources

Format the output clearly with headings and bullet points where appropriate.
""".strip()

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=20,
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print("Generate course error:", e)

    return "Unable to generate course at this time."

# =======================
# ROUTES
# =======================

@app.post("/generate_course/")
def generate_course_route(
    request: Request,
    role: str = Form(...),
    level: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    content = generate_course(role, level)
    return templates.TemplateResponse(
        "course.html",
        {"request": request, "role": role, "level": level, "content": content},
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# =======================
# AUTH ROUTES
# =======================

@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
def signup(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.username == username).first():
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "message": "User already exists"}
        )

    new_user = User(username=username, password=hash_password(password))
    db.add(new_user)
    db.commit()

    return RedirectResponse("/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()

    if user and verify_password(password, user.password):
        response = RedirectResponse("/index", status_code=303)
        response.set_cookie(key="user", value=username, httponly=True)
        return response

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "message": "Invalid credentials"}
    )

@app.get("/logout")
def logout():
    response = RedirectResponse("/")
    response.delete_cookie("user")
    return response

# =======================
# DASHBOARD
# =======================

@app.get("/index", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login")

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "username": user.username}
    )

# =======================
# INTERVIEW FLOW
# =======================

@app.post("/start_interview/")
def start_interview(
    request: Request,
    role: str = Form(...),
    level: str = Form(...),
    db: Session = Depends(get_db)
):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    interview_sessions[user.username] = {
        "role": role,
        "level": level,
        "difficulty": "easy",
        "round": 1,
        "history": [],
        "last_speech_analysis": None
    }

    question = generate_question(role, level, "easy")

    interview_sessions[user.username]["history"].append({
        "question": question,
        "answer": ""
    })

    return templates.TemplateResponse(
        "interview.html",
        {
            "request": request,
            "username": user.username,
            "role": role,
            "level": level,
            "question": question,
            "round": 1,
            "difficulty": "easy"
        }
    )

@app.post("/submit_answer/")
def submit_answer(
    request: Request,
    answer: str = Form(...),
    db: Session = Depends(get_db)
):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    session = interview_sessions.get(user.username)
    if not session:
        raise HTTPException(status_code=400, detail="Interview not started")

    session["history"][-1]["answer"] = answer

    feedback = generate_feedback(answer)

    # merge speech analysis if available
    speech_feedback = ""
    session = interview_sessions.get(user.username)
    if session and session.get("last_speech_analysis"):
        speech_feedback = speech_feedback_from_analysis(session["last_speech_analysis"])

    final_feedback = feedback
    if speech_feedback:
        final_feedback += "\n\n🗣️ Communication Feedback:\n" + speech_feedback

    # adapt difficulty based on feedback text
    low = final_feedback.lower()
    if "incorrect" in low or "wrong" in low or "not correct" in low:
        session["difficulty"] = "easy"
    else:
        if session['difficulty'] == "easy":
            session['difficulty'] = "medium"
        elif session['difficulty'] == "medium":
            session['difficulty'] = "hard"

    next_question = generate_question(
        session["role"],
        session["level"],
        session["difficulty"]
    )

    session["history"].append({
        "question": next_question,
        "answer": ""
    })

    session["round"] += 1

    # Save attempt
    attempt = InterviewAttempt(
        user_id=user.id,
        role=session["role"],
        topic="auto",
        difficulty=session["difficulty"],
        answer=answer,
        feedback=feedback
    )
    
     # Optional: update skill progress under "auto"
    skill = db.query(SkillProgress).filter(
        SkillProgress.user_id == user.id,
        SkillProgress.skill == "auto"
    ).first()

    if not skill:
        skill = SkillProgress(
            user_id=user.id,
            skill="auto",
            attempts=1,
            weak=False
        )
        db.add(skill)
    else:
        skill.attempts += 1

    if "incorrect" in low or "wrong" in low or "not correct" in low:
        skill.weak = True
    
    db.add(attempt)
    db.commit()

    # return the finalized feedback (including any speech comments)
    return {
        "feedback": final_feedback,
        "next_question": next_question
    }

# =======================
# AUDIO / SPEECH ROUTES
# =======================


@app.post("/speech_to_text/")
async def speech_to_text(
    request: Request,
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Accept a small audio clip, transcribe with Whisper, and store analysis in the
    current interview session. Returns the raw transcript and computed analysis.
    """
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    content = await audio.read()
    tmp_name = f"upload_{uuid.uuid4().hex}.wav"
    with open(tmp_name, "wb") as f:
        f.write(content)

    try:
        result = whisper_model.transcribe(tmp_name)
        text = result.get("text", "").strip()
        duration = 0.0
        if "segments" in result and result["segments"]:
            # end time of final segment is a good proxy for duration
            duration = result["segments"][-1].get("end", 0.0)

        analysis = analyze_speech(text, duration)
        session = interview_sessions.get(user.username)
        if session:
            session["last_speech_analysis"] = analysis

        return {"text": text, "analysis": analysis}
    finally:
        try:
            os.remove(tmp_name)
        except OSError:
            pass

# =======================
# PROGRESS PAGE
# =======================

@app.get("/progress", response_class=HTMLResponse)
def progress_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login")

    skills = db.query(SkillProgress).filter(
        SkillProgress.user_id == user.id
    ).all()

    return templates.TemplateResponse(
        "progress.html",
        {
            "request": request,
            "username": user.username,
            "skills": skills
        }
    )
