# prompts.py
# Centralized prompt templates for the interview system


# ============================================================
# QUESTION GENERATION
# ============================================================

def interview_questions_prompt(role: str, level: str, count: int = 5) -> str:

    return f"""
You are an experienced technical interviewer.

Generate EXACTLY {count} interview questions for a {level} {role} candidate.

Question types:
1. Resume/experience question
2. Core technical concept
3. Behavioral question
4. Coding problem
5. Logical reasoning question

Rules:
- Avoid generic questions.
- Questions should test real understanding.
- Coding question should be similar to LeetCode Easy difficulty.

Return ONLY valid JSON array.

Example format:

[
 {{"type":"resume","question":"..."}},
 {{"type":"technical","question":"..."}},
 {{"type":"behavioral","question":"..."}},
 {{"type":"coding","question":"..."}},
 {{"type":"logic","question":"..."}}
]
"""


# ============================================================
# QUESTION GENERATION WITH RESUME
# ============================================================

def interview_questions_with_resume_prompt(
    role: str,
    level: str,
    resume_text: str,
    count: int = 5
) -> str:

    resume_preview = resume_text[:600]

    return f"""
You are an experienced interviewer.

Generate EXACTLY {count} interview questions for a {level} {role} candidate.

Use the resume to personalize ONE question.

Resume snippet:
{resume_preview}

Question distribution:

1 resume-based question
1 technical concept
1 behavioral
1 coding question
1 logical reasoning question

Coding question must be easy algorithmic difficulty.

Return ONLY valid JSON array.

[
 {{"type":"resume","question":"..."}},
 {{"type":"technical","question":"..."}},
 {{"type":"behavioral","question":"..."}},
 {{"type":"coding","question":"..."}},
 {{"type":"logic","question":"..."}}
]
"""


# ============================================================
# RESUME → SKILL PROFILE
# ============================================================

def resume_skill_profile_prompt(resume_text: str, role: str, designation: str) -> str:

    resume_preview = resume_text[:800]

    return f"""
You are analyzing a candidate resume.

Target role: {role}
Experience level: {designation}

Resume snippet:
{resume_preview}

Return ONLY JSON:

{{
 "skills": ["skill1","skill2","skill3"],
 "skill_gaps": ["topic1","topic2"],
 "improvement_suggestions": "short paragraph of learning advice",
 "strength_percentage": number between 0 and 100
}}
"""


# ============================================================
# BASIC EVALUATION (LEGACY MODE)
# ============================================================

def batch_evaluation_prompt(role: str, answers: list[dict]) -> str:

    qa_block = ""

    for i, item in enumerate(answers, 1):

        q = item["question"][:120]
        a = item["answer"][:800]

        qa_block += f"\nQ{i}: {q}\nA{i}: {a}\n"

    return f"""
You are evaluating interview answers for the role of {role}.

Answers are speech transcriptions and may contain grammar mistakes.

Evaluate meaning, not grammar.

Return JSON:

{{
 "feedback_per_question":[
  {{
   "question":"question",
   "candidate_answer":"answer",
   "feedback":"constructive explanation",
   "improved_answer":"better answer"
  }}
 ],
 "improvement_tips":[
  "tip1",
  "tip2",
  "tip3"
 ],
 "learning_resources":[
  {{
   "topic":"topic",
   "resource":"learning suggestion"
  }}
 ]
}}

Interview responses:
{qa_block}
"""


# ============================================================
# MAIN INTERVIEW EVALUATION
# ============================================================

def interview_evaluation_prompt(role: str, questions_answers: list[dict]) -> str:

    qa_block = ""

    for i, qa in enumerate(questions_answers, 1):

        q = qa["question"][:120]
        a = qa["answer"][:800]

        qa_block += f"\nQ{i}: {q}\nA{i}: {a}\n"

    return f"""
You are a technical interviewer evaluating answers for a {role} interview.

Important:
Answers come from speech transcription. Ignore grammar errors.
Ignore minor factual inaccuracies if overall understanding is clear.
ignore irrelevant rambling if core answer is good.


Evaluate every answer.

Scoring guide:
90-100 → Excellent
70-89 → Good with minor gaps
50-69 → Partial understanding
30-49 → Weak but attempted
0-29 → Incorrect or empty

Return ONLY JSON.

{{
 "answers":[
  {{
   "score":70,
   "strengths":["specific strength"],
   "weaknesses":["specific improvement"],
   "ideal_answer":"short model answer about how the user should've answered(maximum 3 sentences)",
   "weak_topics":["topic to study"]
  }}
 ],
 "weak_topics":["overall topic1","overall topic2"]
}}

Interview responses:
{qa_block}
"""


# ============================================================
# COURSE OUTLINE
# ============================================================

def course_outline_prompt(role: str, level: str) -> str:

    return f"""
You are an expert course designer.

Create a learning roadmap for a {level} {role}.

Return ONLY JSON.

{{
 "course_title":"title",
 "course_description":"short description",
 "modules":[
  {{
   "title":"module title",
   "description":"module description"
  }}
 ]
}}
"""


# ============================================================
# COURSE MODULE DETAIL
# ============================================================

def course_module_detail_prompt(role: str, level: str, module_title: str) -> str:

    return f"""
Generate detailed educational content.

Course: {level} {role}
Module: {module_title}

Return ONLY JSON.

{{
 "module_title":"{module_title}",
 "overview":"overview text",
 "concepts":["concept1","concept2"],
 "lessons":[
  {{
   "title":"lesson title",
   "theory":"concept explanation",
   "example":"practical example"
  }}
 ],
 "exercises":["exercise1","exercise2"],
 "quiz":[
  {{
   "question":"quiz question",
   "options":["A","B","C","D"],
   "answer":"correct option"
  }}
 ],
 "project":{{
  "title":"project title",
  "description":"project description",
  "skills_practiced":["skill1","skill2"]
 }},
 "resources":[
  {{
   "title":"resource",
   "link":"URL"
  }}
 ]
}}
"""


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def performance_summary_prompt(report_data: dict) -> str:

    profile = report_data.get("candidate_profile", {})
    voice = report_data.get("voice_analysis", {})
    content = report_data.get("content_analysis", {})
    verdict = report_data.get("verdict", {})
    answers = report_data.get("detailed_answers", [])

    role = profile.get("role")
    score = report_data.get("overall_score")

    clarity = voice.get("clarity_score")
    engagement = voice.get("engagement_score")

    relevance = content.get("relevance_score")
    depth = content.get("depth_score")

    recommendation = verdict.get("recommendation")

    # Summarize answers for the prompt
    answer_summary = []

    for i, ans in enumerate(answers, 1):

        q = ans.get("question", "")[:120]
        a_score = ans.get("score", 0)

        strengths = ", ".join(ans.get("strengths", []))
        weaknesses = ", ".join(ans.get("weaknesses", []))

        answer_summary.append(
            f"Q{i} Score: {a_score}\n"
            f"Strengths: {strengths}\n"
            f"Weaknesses: {weaknesses}"
        )

    answer_block = "\n\n".join(answer_summary)

    return f"""
You are summarizing an AI interview evaluation.

Role: {role}
Overall Score: {score}
Technical Relevance: {relevance}
Depth of Explanation: {depth}
Clarity Score: {clarity}
Engagement Score: {engagement}
Verdict: {recommendation}

Question Evaluations:
{answer_block}

Give a concise 4 sentence performance summary.

The summary must include:
1. Candidate's main strengths
2. Key technical weaknesses
3. Communication quality
4. One actionable recommendation for improvement

Be specific and refer to the evaluation results.
Avoid generic statements.
"""