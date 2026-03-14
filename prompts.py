# prompts.py — Centralized prompt templates


# ============================================================
# QUESTION GENERATION
# ============================================================

def interview_questions_prompt(role: str, level: str, count: int = 5) -> str:
    return f"""
You are an experienced technical interviewer.

Generate EXACTLY {count} interview questions for a {level} {role} candidate.

Questions must include:

• technical concepts
• behavioral questions
• one coding question
• one logical reasoning problem

Coding question rules:
• Easy level (similar to LeetCode Easy)
• Test problem-solving ability

Return ONLY a JSON array of questions.

Example:

[
"Question 1?",
"Question 2?",
"Question 3?",
"Question 4?",
"Question 5?"
]
"""


def interview_questions_with_resume_prompt(role: str, level: str, resume_text: str, count: int = 5) -> str:

    truncated = resume_text[:600]

    return f"""
You are an experienced technical interviewer.

Generate EXACTLY {count} interview questions for a {level} {role} candidate.

Question distribution:

1. Resume-based question
2. Core technical concept
3. Behavioral or experience question
4. Coding problem
5. Logical reasoning problem

Important rules:

• Do NOT rely only on the resume.
• Avoid generic questions like "Tell me about yourself".
• At least ONE coding problem must be included.
• Prefer a programming language mentioned in the resume.
• If none detected, use pseudocode.

Coding difficulty:
Easy algorithmic problem similar to LeetCode Easy.

Resume:
{truncated}

Return ONLY JSON array.

[
"Question 1?",
"Question 2?",
"Question 3?",
"Question 4?",
"Question 5?"
]
"""


# ============================================================
# BASIC EVALUATION (LEGACY MODE)
# ============================================================

def batch_evaluation_prompt(role: str, answers: list[dict]) -> str:

    qa_block = ""
    for i, item in enumerate(answers, 1):
        qa_block += f"\nQ{i}: {item['question']}\nA{i}: {item['answer']}\n"

    return f"""
You are an expert interview evaluator for the role of {role}.

Evaluate the answers below.

The answers come from speech-to-text transcription and may contain:
• grammar mistakes
• punctuation issues
• repeated words

Evaluate the **intended meaning**, not grammar.

For EACH answer return:

• feedback
• improved_answer

Also return:

• 3 improvement tips
• 3 learning resources

Interview responses:
{qa_block}

Return ONLY JSON.

{{
 "feedback_per_question":[
  {{
   "question":"question",
   "candidate_answer":"answer",
   "feedback":"feedback",
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
   "resource":"resource"
  }}
 ]
}}
"""


# ============================================================
# MAIN INTERVIEW EVALUATION
# ============================================================

def interview_evaluation_prompt(role: str, questions_answers: list[dict]) -> str:

    qa_block = ""
    for i, qa in enumerate(questions_answers, 1):
        q = qa["question"][:120]
        a = qa["answer"][:600]
        qa_block += f"\nQ{i}: {q}\nA{i}: {a}\n"

    return f"""
You are a technical interviewer evaluating answers for a {role} interview.

IMPORTANT:
Answers come from speech transcription and may contain grammar errors.
Evaluate the intended meaning, not exact wording.

Scoring guide:
70-100 → concept mostly correct
50-69 → partial understanding
35-49 → weak but attempted
0-34 → incorrect

For EACH answer return:
score
feedback (2 sentences)
strengths (2 points)
weaknesses (2 points)
ideal_answer (short strong answer)

Also return weak_topics (concepts to study).

Interview:
{qa_block}

Return ONLY JSON:

{{
 "answers":[
  {{
   "score":70,
   "feedback":"short explanation",
   "strengths":["point","point"],
   "weaknesses":["issue","issue"],
   "ideal_answer":"example answer"
  }}
 ],
 "weak_topics":["topic1","topic2"],
 "overall_feedback":"2 sentence summary"
}}
"""


# ============================================================
# COURSE GENERATION
# ============================================================

def course_outline_prompt(role: str, level: str) -> str:

    return f"""
Create a structured course outline for a {level} {role}.

The course must prepare the student with all skills needed for the role.

Include technologies, frameworks, and concepts employers expect.

Return ONLY JSON.

{{
 "course_title":"Course title",
 "course_description":"Short description",
 "modules":[
  {{"title":"Module title","description":"Short description"}}
 ]
}}
"""


def course_module_detail_prompt(role: str, level: str, module_title: str) -> str:

    return f"""
Generate educational content for the module "{module_title}" in a {level} {role} course.

Return ONLY JSON.

{{
 "module_title":"{module_title}",
 "overview":"overview text",
 "concepts":["concept1","concept2"],
 "lessons":[
  {{
   "title":"Lesson title",
   "theory":"theory explanation",
   "example":"example"
  }}
 ],
 "exercises":["exercise1","exercise2"],
 "quiz":[
  {{
   "question":"question",
   "options":["A","B","C","D"],
   "answer":"correct"
  }}
 ],
 "project":{{
  "title":"project",
  "description":"description",
  "skills_practiced":["skill1"]
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

    return f"""
Write a concise 4 sentence interview performance summary.

Role: {profile.get('role')}
Level: {profile.get('level')}
Score: {report_data.get('overall_score')}/100
Verdict: {verdict.get('recommendation')}

Voice metrics:
Pace {voice.get('speaking_pace_wpm')} WPM,
Fillers {voice.get('total_filler_words')}

Content metrics:
Average score {content.get('average_score')}

Discuss:

• overall performance
• communication
• technical understanding
• improvement areas

Return ONLY the paragraph.
"""