# prompts.py — Centralized prompt templates for all AI interactions


def interview_questions_prompt(role: str, level: str, count: int = 5) -> str:
    return f"""Generate {count} interview questions for a {level} {role}. Mix behavioral, technical, and situational.
Return ONLY a JSON array of strings:
["Question 1?", "Question 2?"]"""


def interview_questions_with_resume_prompt(role: str, level: str, resume_text: str, count: int = 5) -> str:
    truncated = resume_text[:800]
    return f"""Generate {count} personalized interview questions for a {level} {role} based on their resume.
Mix: resume-specific, technical, behavioral, and gap-probing questions.

Resume:
{truncated}

Return ONLY a JSON array of strings:
["Question 1?", "Question 2?"]"""


def batch_evaluation_prompt(role: str, answers: list[dict]) -> str:
    qa_block = ""
    for i, item in enumerate(answers, 1):
        qa_block += f"\nQ{i}: {item['question']}\nA{i}: {item['answer']}\n"

    return f"""You are an expert interview evaluator for the role of {role}.
Evaluate each question-answer pair below. Be specific and constructive.

For EACH pair you MUST provide:
- feedback: 2-3 sentences of specific, constructive feedback about the answer
- improved_answer: a stronger 3-4 sentence example answer the candidate could give instead

Also provide:
- 3 general improvement tips as an array of strings
- 3 learning resources with topic name and description

Interview responses:
{qa_block}

IMPORTANT: Return ONLY valid JSON. Do NOT include any text before or after the JSON.
Do NOT wrap the JSON in markdown code fences.

{{
  "feedback_per_question": [
    {{
      "question": "the exact question text",
      "candidate_answer": "the candidate's answer",
      "feedback": "2-3 sentences of specific constructive feedback",
      "improved_answer": "a stronger 3-4 sentence example answer"
    }}
  ],
  "improvement_tips": [
    "specific actionable tip 1",
    "specific actionable tip 2",
    "specific actionable tip 3"
  ],
  "learning_resources": [
    {{
      "topic": "topic name",
      "resource": "description or URL"
    }}
  ]
}}"""


def course_outline_prompt(role: str, level: str) -> str:
    return f"""Create a structured course outline for a {level} {role}.

The course must prepare the student with ALL the core skills required to work as a {role}.
Include role-specific technologies, tools, frameworks, and concepts that employers expect.

The course MUST have exactly 4 modules:
1. Fundamentals — core foundational knowledge every {role} needs
2. Intermediate Skills — practical tools and techniques used daily
3. Advanced Topics — complex concepts that separate good from great
4. Projects & Practice — real-world projects to build a portfolio

IMPORTANT: Return ONLY valid JSON, no text before or after.
{{
  "course_title": "Course title here",
  "course_description": "A 1-2 sentence course description",
  "modules": [
    {{"title": "Module title", "description": "Brief 1-sentence description of this module"}}
  ]
}}"""


def course_module_detail_prompt(role: str, level: str, module_title: str) -> str:
    return f"""You are building a professional online course for a {level} {role}.
Generate COMPLETE educational content for the module: "{module_title}".

Include ALL sections below with detailed, real educational content relevant to the {role} role.

IMPORTANT: Return ONLY valid JSON. No text before or after the JSON.

{{
  "module_title": "{module_title}",
  "overview": "3-4 sentence overview of this module",
  "concepts": ["key concept 1", "key concept 2", "key concept 3", "key concept 4"],
  "lessons": [
    {{
      "title": "Lesson title",
      "theory": "Detailed multi-paragraph theory explanation (at least 3-4 sentences)",
      "example": "A concrete code example, formula, or real-world scenario"
    }},
    {{
      "title": "Another lesson",
      "theory": "Detailed explanation",
      "example": "Another example"
    }},
    {{
      "title": "Third lesson",
      "theory": "Detailed explanation",
      "example": "Another example"
    }}
  ],
  "exercises": [
    "Practical exercise 1 with clear instructions",
    "Practical exercise 2 with clear instructions",
    "Practical exercise 3 with clear instructions"
  ],
  "quiz": [
    {{
      "question": "Multiple choice question",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "Correct option text"
    }},
    {{
      "question": "Another question",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "Correct option text"
    }}
  ],
  "project": {{
    "title": "Mini project title",
    "description": "Detailed project description with requirements",
    "skills_practiced": ["skill1", "skill2", "skill3"]
  }},
  "resources": [
    {{
      "title": "Resource name",
      "link": "URL or description"
    }}
  ]
}}"""


def content_analysis_prompt(role: str, level: str, questions_answers: list[dict]) -> str:
    """Prompt for detailed content evaluation of interview answers."""
    qa_block = ""
    for i, item in enumerate(questions_answers, 1):
        q = item['question'][:150]
        a = item['answer'][:800]
        qa_block += f"\nQ{i}: {q}\nA{i}: {a}\n"

    return f"""You are a fair and experienced technical interviewer evaluating {len(questions_answers)} answers for a {level} {role} position.

SCORING RUBRIC (use the FULL range):
- 90-100: Excellent — accurate, comprehensive, demonstrates deep understanding
- 70-89: Mostly correct — covers key concepts with minor gaps
- 50-69: Partial — some correct ideas but incomplete or vague
- 30-49: Weak — shows minimal understanding, major gaps
- 0-29: Incorrect — wrong, off-topic, or empty answer

IMPORTANT RULES:
- Score based on CONCEPTUAL correctness, not exact wording. An answer phrased differently but conveying the right idea should score high.
- Accept alternative correct explanations. There are many valid ways to explain a concept.
- A brief but accurate answer deserves 60-75. A detailed and accurate answer deserves 80-100.
- An empty or irrelevant answer must score 0-15.
- Each answer MUST have at least 1 item in "strengths" and at least 1 item in "improvements".
{qa_block}
Return ONLY valid JSON, no extra text:
{{"answers":[{{"score":75,"feedback":"2-3 sentence evaluation.","strengths":["specific strength 1"],"improvements":["specific actionable improvement 1"]}}],"overall_feedback":"2-3 sentence summary of candidate performance.","aggregate":{{"technical_score":70,"communication_score":65,"overall_score":68}}}}"""


def performance_summary_prompt(report_data: dict) -> str:
    """Prompt for generating an executive performance summary."""
    profile = report_data.get("candidate_profile", {})
    voice = report_data.get("voice_analysis", {})
    content = report_data.get("content_analysis", {})
    meta = report_data.get("session_metadata", {})
    verdict = report_data.get("verdict", {})

    return f"""Write a 4-sentence professional interview performance summary.
Role: {profile.get('role', 'Unknown')} ({profile.get('level', 'Unknown')}), Score: {report_data.get('overall_score', 0)}/100, Verdict: {verdict.get('recommendation', 'N/A')}
Voice: {voice.get('speaking_pace_wpm', 0)} WPM, {voice.get('total_filler_words', 0)} fillers, clarity {voice.get('clarity_score', 0)}/100
Content: avg {content.get('average_score', 0)}/100, technical {content.get('technical_score', 0)}, communication {content.get('communication_score', 0)}
Cover: overall assessment, communication, technical knowledge, areas for improvement.
Return ONLY the plain-text paragraph."""


def interview_evaluation_prompt(role: str, questions_answers: list[dict]) -> str:
    qa_block = ""
    for i, item in enumerate(questions_answers, 1):
        qa_block += f"\nQ{i}: {item['question']}\nA{i}: {item['answer']}\n"

    return f"""You are an expert interview evaluator for the role of {role}.
Evaluate each question-answer pair below. Provide a score from 0-100, an ideal answer, strengths, weaknesses, and feedback for each.

{qa_block}

IMPORTANT: Return ONLY valid JSON. Do NOT include any text before or after the JSON.
Do NOT wrap the JSON in markdown code fences.

{{
  "evaluations": [
    {{
      "question": "the exact question text",
      "candidate_answer": "the candidate's answer",
      "score": 85,
      "ideal_answer": "A strong example answer that demonstrates key competencies",
      "strengths": ["Specific strength 1", "Specific strength 2"],
      "weaknesses": ["Specific weakness 1", "Specific weakness 2"],
      "feedback": "2-3 sentences of constructive feedback"
    }}
  ]
}}"""