from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, Text
from database import Base
from datetime import datetime


# =========================
# USERS
# =========================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)


class UserTarget(Base):
    __tablename__ = "user_targets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)

    role = Column(String, nullable=False)
    level = Column(String, nullable=False)  # Intern/Junior/Senior
    updated_at = Column(DateTime, default=datetime.utcnow)


# =========================
# SKILL TRACKING
# =========================
class SkillProgress(Base):
    __tablename__ = "skill_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    skill = Column(String)
    attempts = Column(Integer, default=0)
    weak = Column(Boolean, default=False)


# =========================
# INTERVIEW
# =========================
class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    role = Column(String, nullable=False)
    level = Column(String, nullable=False)

    status = Column(String, default="active")  # active/completed
    created_at = Column(DateTime, default=datetime.utcnow)


class InterviewAttempt(Base):
    __tablename__ = "interview_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # ✅ FIXED

    role = Column(String)
    topic = Column(String)
    difficulty = Column(String)
    answer = Column(Text)       # better than String for long answers
    feedback = Column(Text)     # better than String for long feedback
    timestamp = Column(DateTime, default=datetime.utcnow)


# =========================
# COURSE BUILDER
# =========================
class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    level = Column(String, default="beginner")     # beginner/intermediate/advanced
    status = Column(String, default="draft")       # draft/generated


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"))

    title = Column(String, nullable=False)
    order_index = Column(Integer, default=0)


class Unit(Base):
    __tablename__ = "units"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"))

    title = Column(String, nullable=False)
    order_index = Column(Integer, default=0)

    content = Column(Text, nullable=True)
    status = Column(String, default="pending")
    estimated_minutes = Column(Integer, default=10)


# =========================
# QUIZ
# =========================
class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    unit_id = Column(Integer, ForeignKey("units.id"))
    score = Column(Integer, default=0)

    wrong_concepts = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
