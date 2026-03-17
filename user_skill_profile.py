# user_skill_profile.py
# Pydantic models used for user skill profiling and interview tracking

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================
# BASIC USER INFO
# ============================================================

class BasicUserInfo(BaseModel):

    user_id: str
    name: str
    target_role: str
    experience_level: str
    resume_file_path: str

    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# RESUME STRUCTURE
# ============================================================

class Project(BaseModel):

    title: str
    description: str


class Education(BaseModel):

    degree: str
    university: str
    graduation_year: int


class WorkExperience(BaseModel):

    company: str
    role: str
    duration: str


class ResumeData(BaseModel):

    skills: List[str] = Field(default_factory=list)

    projects: List[Project] = Field(default_factory=list)

    education: Optional[Education] = None

    work_experience: List[WorkExperience] = Field(default_factory=list)


# ============================================================
# SKILL GRAPH NODE
# ============================================================

class SkillNode(BaseModel):

    skill_name: str

    proficiency_level: str = "beginner"

    score: float = Field(0, ge=0, le=100)

    last_updated: datetime = Field(default_factory=datetime.utcnow)

    times_tested: int = 0

    @field_validator("proficiency_level")
    def validate_level(cls, v):

        allowed = {"beginner", "intermediate", "advanced"}

        if v not in allowed:
            raise ValueError("Invalid proficiency level")

        return v


# ============================================================
# TECHNICAL SKILLS
# ============================================================

class TechnicalSkillVector(BaseModel):

    dsa: int = Field(50, ge=0, le=100)

    dbms: int = Field(50, ge=0, le=100)

    operating_systems: int = Field(50, ge=0, le=100)

    computer_networks: int = Field(50, ge=0, le=100)

    system_design: int = Field(50, ge=0, le=100)


# ============================================================
# INTERVIEW SKILLS
# ============================================================

class InterviewSkillVector(BaseModel):

    relevance: int = Field(50, ge=0, le=100)

    explanation_depth: int = Field(50, ge=0, le=100)

    structured_thinking: int = Field(50, ge=0, le=100)

    problem_solving: int = Field(50, ge=0, le=100)

    star_method: int = Field(50, ge=0, le=100)


# ============================================================
# COMMUNICATION SKILLS
# ============================================================

class CommunicationSkillVector(BaseModel):

    clarity: int = Field(50, ge=0, le=100)

    confidence: int = Field(50, ge=0, le=100)

    engagement: int = Field(50, ge=0, le=100)

    speaking_pace: int = Field(50, ge=0, le=100)

    filler_control: int = Field(50, ge=0, le=100)


# ============================================================
# COMPLETE SKILL VECTOR
# ============================================================

class UserSkillVector(BaseModel):

    technical_skills: TechnicalSkillVector = Field(default_factory=TechnicalSkillVector)

    interview_skills: InterviewSkillVector = Field(default_factory=InterviewSkillVector)

    communication_skills: CommunicationSkillVector = Field(default_factory=CommunicationSkillVector)


# ============================================================
# INTERVIEW SCORING
# ============================================================

class ScoreBreakdown(BaseModel):

    correctness: float = Field(ge=0, le=100)

    conceptual_depth: float = Field(ge=0, le=100)

    clarity: float = Field(ge=0, le=100)

    feedback: str

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InterviewRecord(BaseModel):

    question: str

    topic: str

    answer_transcript: str

    evaluation_score: float = Field(ge=0, le=100)

    score_breakdown: ScoreBreakdown


# ============================================================
# COURSE TRACKING
# ============================================================

class CourseProgress(BaseModel):

    course_name: str

    completion_percentage: float = Field(0, ge=0, le=100)

    quizzes_completed: int = 0

    last_accessed: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# MAIN USER SKILL PROFILE
# ============================================================

class UserSkillProfile(BaseModel):

    user_id: str

    basic_info: BasicUserInfo

    resume_data: ResumeData

    technical_skills: TechnicalSkillVector = Field(default_factory=TechnicalSkillVector)

    interview_skills: InterviewSkillVector = Field(default_factory=InterviewSkillVector)

    communication_skills: CommunicationSkillVector = Field(default_factory=CommunicationSkillVector)

    overall_score: float = Field(50, ge=0, le=100)

    interview_count: int = 0

    interview_history: List[InterviewRecord] = Field(default_factory=list)

    courses: List[CourseProgress] = Field(default_factory=list)

    last_updated: datetime = Field(default_factory=datetime.utcnow)


    # ============================================================
# PROFILE HELPERS (required by main.py)
# ============================================================

def create_user_profile(basic_info: BasicUserInfo, resume_data: ResumeData) -> UserSkillProfile:
    """
    Initialize a new user skill profile.
    """
    return UserSkillProfile(
        user_id=basic_info.user_id,
        basic_info=basic_info,
        resume_data=resume_data,
    )


def detect_weaknesses(skill_vector: UserSkillVector) -> list[str]:
    """
    Detect weak skills below threshold.
    """
    weaknesses = []

    for name, value in skill_vector.technical_skills.model_dump().items():
        if value < 40:
            weaknesses.append(name)

    for name, value in skill_vector.interview_skills.model_dump().items():
        if value < 40:
            weaknesses.append(name)

    for name, value in skill_vector.communication_skills.model_dump().items():
        if value < 40:
            weaknesses.append(name)

    return weaknesses


def recommend_micro_courses(weak_topics: list[str]) -> list[dict]:
    """
    Generate micro-learning suggestions.
    """
    return [
        {
            "topic": topic,
            "course": f"Practice and revise {topic}",
        }
        for topic in weak_topics
    ]


def update_skill_score(current_score: float, new_score: float) -> float:
    """
    Update skill score using weighted averaging.
    """
    return round((current_score * 0.7) + (new_score * 0.3), 2)


def update_skill_vector(skill_vector: UserSkillVector, updates: dict) -> UserSkillVector:

    for key, value in updates.items():

        if hasattr(skill_vector.technical_skills, key):
            current = getattr(skill_vector.technical_skills, key)
            setattr(
                skill_vector.technical_skills,
                key,
                update_skill_score(current, value)
            )

        elif hasattr(skill_vector.interview_skills, key):
            current = getattr(skill_vector.interview_skills, key)
            setattr(
                skill_vector.interview_skills,
                key,
                update_skill_score(current, value)
            )

        elif hasattr(skill_vector.communication_skills, key):
            current = getattr(skill_vector.communication_skills, key)
            setattr(
                skill_vector.communication_skills,
                key,
                update_skill_score(current, value)
            )

    return skill_vector

def calculate_overall_score(skill_vector: UserSkillVector) -> float:
    """
    Compute overall skill score.
    """

    tech = sum(skill_vector.technical_skills.model_dump().values()) / 5
    interview = sum(skill_vector.interview_skills.model_dump().values()) / 5
    comm = sum(skill_vector.communication_skills.model_dump().values()) / 5

    return round((tech * 0.5) + (interview * 0.3) + (comm * 0.2), 2)


def record_interview_result(profile: UserSkillProfile, interview: InterviewRecord):
    """
    Store interview record and update stats.
    """

    profile.interview_history.append(interview)
    profile.interview_count += 1
    profile.last_updated = datetime.utcnow()

    return profile