from database import SessionLocal
from models import User, InterviewAttempt, SkillProgress

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File

from dotenv import load_dotenv
#import google.genai as genai
import os
import requests
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


# ---------------- DB ----------------
def get_db():
    return SessionLocal()


# ---------------- ENV ----------------
load_dotenv()
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # you have llama3:latest

# ---------------- SPEECH MODEL ----------------
whisper_model = whisper.load_model("base")

# ---------------- APP ----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------- DEFAULT QUESTIONS (Final Fallback) ----------------
DEFAULT_QUESTIONS = {
    "intern": [
        "Tell me about yourself.",
        "What is the difference between a process and a thread?",
        "What is normalization in DBMS?",
        "Explain OOP concepts in simple words.",
        "What is an API and why do we use it?"
    ],
    "junior": [
        "Tell me about a project you built and what challenges you faced.",
        "Explain SQL JOINs with an example.",
        "What is deadlock and how can it be prevented?",
        "Explain polymorphism and give a real-world example.",
        "What is HTTP and how is it different from HTTPS?"
    ],
    "mid": [
        "Explain indexing in databases and when it helps.",
        "How do you handle concurrency in an application?",
        "Explain SOLID principles briefly.",
        "What happens when you type a URL in a browser?",
        "How do you optimize a slow SQL query?"
    ],
    "senior": [
        "How would you design a scalable interview preparation platform?",
        "Explain CAP theorem and real-world tradeoffs.",
        "How do you approach performance optimization end-to-end?",
        "How do you handle system failures and retries in production?",
        "Explain database sharding and when you would use it."
    ]
}


# ---------------- INTERVIEW SESSION STORE (in-memory) ----------------
interview_sessions = {}


# ---------------- FALLBACK FUNCTIONS ----------------
def generate_question_with_fallback(role: str, level: str, difficulty: str = "easy") -> str:
    prompt = f"""
You are a mock interviewer.
Role: {role}
Level: {level}
Difficulty: {difficulty}

Ask ONE realistic interview question.
Rules:
- Keep it short and practical.
- Don't mention "difficulty" or "category".
Return ONLY the question text.
""".strip()

    # 1) OLLAMA (Primary)
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": 
            {
                "num_predict": 80
             }
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=20)
        if r.status_code == 200:
            data = r.json()
            text = data.get("response", "").strip()
            if text:
                return text
    except Exception as e:
        print("Ollama question failed:", str(e))

    # 2) GEMINI (Secondary)
    
    # try:
    #     client = genai.Client(api_key=GEMINI_API_KEY)
    #     response = client.models.generate_content(
    #         model="gemini-2.0-flash",
    #         contents=prompt
    #     )
    #     text = response.text.strip() if hasattr(response, "text") else ""
    #     if text:
    #         return text
    # except Exception as e:
    #     print("Gemini question failed:", str(e)) 
    
    # 3) DEFAULT (Final)
    level_key = level.lower().strip()
    if level_key not in DEFAULT_QUESTIONS:
        level_key = "intern"
    return random.choice(DEFAULT_QUESTIONS[level_key])


def generate_feedback_with_fallback(role: str, level: str, answer: str) -> str:
    prompt = f"""
You are an interviewer for {role} ({level}).
Respond STRICTLY in the following format and DO NOT stop early:

Strength:
- (1–2 lines)

Improvement:
- (2–3 lines)

Improved Answer:
- (4–6 complete sentences)

Candidate answer:
{answer}

Finish all sections fully before stopping.
Keep it beginner-friendly.
""".strip()

    # 1) OLLAMA
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 450,
                "temperature": 0.7
          }
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=90)
        if r.status_code == 200:
            data = r.json()
            print("OLLAMA FEEDBACK RAW:", data)
            text = data.get("response", "").strip()
            if len(text.split()) < 60:
               print("⚠️ Feedback too short, using fallback expansion")
               return (
                   "Strength:\nYour answer shows effort and basic understanding.\n\n"
                   "Improvement:\nAdd clearer structure and concrete examples.\n\n"
                   "Improved Answer:\n"
                   "A strong response would begin by clearly defining the concept, "
                   "followed by an example and a brief explanation of why it matters in practice."
               )

            if text:
                return text
    except Exception as e:
        print("Ollama feedback failed:", str(e))

    # 2) GEMINI
    # try:
    #     client = genai.Client(api_key=GEMINI_API_KEY)
    #     response = client.models.generate_content(
    #         model="gemini-1.5-flash",
    #         contents=prompt
    #     )
    #     text = response.text.strip() if hasattr(response, "text") else ""
    #     if text:
    #         return text
    # except Exception as e:
    #     print("Gemini feedback failed:", str(e))

    # 3) DEFAULT
    return "Feedback unavailable right now. Improve clarity, add an example, and explain step-by-step."


# ---------------- ROUTES ----------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/signup", response_class=HTMLResponse)
def signup(request: Request, username: str = Form(...), password: str = Form(...)):
    db = get_db()

    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "message": "User already exists!"}
        )

    new_user = User(username=username, password=password)
    db.add(new_user)
    db.commit()

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "message": "Signup successful! Please login."}
    )


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    db = get_db()

    user = db.query(User).filter(
        User.username == username,
        User.password == password
    ).first()

    if user:
        response = RedirectResponse(url="/index", status_code=303)
        response.set_cookie(key="user", value=username)
        return response

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "message": "Invalid credentials."}
    )


@app.get("/index", response_class=HTMLResponse)
def interview_page(request: Request):
    username = request.cookies.get("user")
    if not username:
        return RedirectResponse(url="/login")

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "username": username}
    )


@app.get("/logout")
def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("user")
    return response


# ---------------- NEW INTERVIEW FLOW ----------------

@app.post("/start_interview/")
async def start_interview(request: Request, role: str = Form(...), level: str = Form(...)):
    username = request.cookies.get("user")
    if not username:
        return {"error": "Not logged in"}

    interview_sessions[username] = {
        "role": role,
        "level": level,
        "round": 1,
        "difficulty": "easy",
        "history": [],
        "last_speech_analysis": None
    }

    question_text = generate_question_with_fallback(role, level, "easy")
    interview_sessions[username]["history"].append({"question": question_text, "answer": ""})

    return {"question": question_text}


@app.post("/submit_answer/")
async def submit_answer(request: Request, answer: str = Form(...)):
    if not answer.strip():
        return {
            "feedback": "Please provide an answer so I can evaluate it.",
            "next_question": interview_sessions.get(
                request.cookies.get("user"), {}
            ).get("history", [{}])[-1].get("question", "")
        }
    username = request.cookies.get("user")
    if not username:
        return {"error": "Not logged in"}

    if username not in interview_sessions:
        return {"error": "Interview not started. Please start again."}

    session = interview_sessions[username]
    role = session["role"]
    level = session["level"]
    difficulty = session["difficulty"]

    # Save answer
    session["history"][-1]["answer"] = answer

    # Feedback
    feedback_text = generate_feedback_with_fallback(role, level, answer)

    # If speech analysis exists, merge it
    speech_feedback = ""
    if username in interview_sessions:
        session = interview_sessions[username]
        if "last_speech_analysis" in session and session["last_speech_analysis"]:
            speech_feedback = speech_feedback_from_analysis(
                session["last_speech_analysis"]
            )
    final_feedback = feedback_text

    if speech_feedback:
        final_feedback += "\n\n🗣️ Communication Feedback:\n" + speech_feedback
    # Difficulty adaptation (simple)
    low = final_feedback.lower()
    if "incorrect" in low or "wrong" in low or "not correct" in low:
        session["difficulty"] = "easy"
    else:
        if difficulty == "easy":
            session["difficulty"] = "medium"
        elif difficulty == "medium":
            session["difficulty"] = "hard"

    # Next question
    next_question = generate_question_with_fallback(role, level, session["difficulty"])

    session["history"].append({"question": next_question, "answer": ""})
    session["round"] += 1

    # Save attempt in DB
    db = get_db()
    user = db.query(User).filter(User.username == username).first()

    attempt = InterviewAttempt(
        user_id=user.id,
        role=role,
        topic="auto",
        difficulty=session["difficulty"],
        answer=answer,
        feedback=final_feedback
    )
    db.add(attempt)

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

    db.commit()

    return {"feedback": final_feedback, "next_question": next_question}


@app.get("/progress", response_class=HTMLResponse)
def progress_page(request: Request):
    db = get_db()
    username = request.cookies.get("user")

    user = db.query(User).filter(User.username == username).first()
    skills = db.query(SkillProgress).filter(SkillProgress.user_id == user.id).all()

    return templates.TemplateResponse(
        "progress.html",
        {"request": request, "username": username, "skills": skills}
    )
@app.post("/speech_to_text/")
async def speech_to_text(request: Request,audio: UploadFile = File(...)):
    temp_id = str(uuid.uuid4())

    webm_path = f"temp_{temp_id}.webm"
    wav_path = f"temp_{temp_id}.wav"

    # Save uploaded file
    with open(webm_path, "wb") as f:
        f.write(await audio.read())

    # Convert to WAV for Whisper
    subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Transcribe
    result = whisper_model.transcribe(wav_path,language="en",word_timestamps=True)
    
    segments = result.get("segments", [])
    if segments:
        duration = segments[-1]["end"]
    else:
        duration = 0.0

    analysis = analyze_speech(result["text"], duration)
# ✅ STORE ANALYSIS FOR SUBMIT_ANSWER
    username = request.cookies.get("user")
    if username and username in interview_sessions:
        interview_sessions[username]["last_speech_analysis"] = analysis
    # Cleanup
    os.remove(webm_path)
    os.remove(wav_path)
    from fastapi import Request
    return {"text": result["text"], "analysis": analysis}

