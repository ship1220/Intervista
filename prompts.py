# prompts.py — Centralized prompt templates for all AI interactions


def interview_questions_prompt(role: str, level: str, count: int = 5) -> str:
    return f"""You are an expert interviewer for the role of {role} at the {level} level.
Generate exactly {count} realistic interview questions that would be asked in a real interview.
Mix behavioral, technical, and situational questions appropriate for this role and level.

Return ONLY a JSON array of strings, no extra text. Example format:
["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]"""


def interview_questions_with_resume_prompt(role: str, level: str, resume_text: str, count: int = 5) -> str:
    truncated = resume_text[:3000]
    return f"""You are an expert interviewer for the role of {role} at the {level} level.

The candidate has submitted the following resume:
---
{truncated}
---

Based on the candidate's resume, skills, projects, and experience, generate exactly {count} personalized interview questions.

Mix these types:
- Questions about specific projects or experiences from their resume
- Technical questions relevant to their listed skills
- Behavioral questions appropriate for the {level} level
- At least one question probing gaps or growth areas

Return ONLY a JSON array of strings, no extra text:
["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]"""


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
    """Prompt for detailed content evaluation of interview answers.

    Returns a prompt that asks the LLM for a structured evaluation.
    """
    qa_block = ""
    for i, item in enumerate(questions_answers, 1):
        qa_block += f"\nQ{i}: {item['question']}\nA{i}: {item['answer']}\n"

    return f"""You are a senior technical recruiter evaluating a candidate for {role} at the {level} level.

Evaluate EACH of the {len(questions_answers)} question-answer pairs below. You MUST provide feedback for EVERY single question.

For EACH answer provide:
- "score": integer 0-100 based on correctness, depth, relevance, and structure
- "feedback": detailed paragraph with strengths, weaknesses, and improved approach
- "strengths": list of specific strengths in the answer
- "improvements": list of specific areas for improvement

Also provide:
- "overall_feedback": 4-6 sentence professional summary
- "aggregate": technical_score, communication_score, overall_score (each 0-100)

Interview responses:
{qa_block}

IMPORTANT: Return ONLY valid JSON. No markdown fences. No text before or after.

{{
  "answers": [
    {{
      "score": 75,
      "feedback": "The answer demonstrated good understanding...",
      "strengths": ["Specific strength 1", "Specific strength 2"],
      "improvements": ["Could add more detail about X", "Consider Y"]
    }}
  ],
  "overall_feedback": "The candidate demonstrated... Overall performance was...",
  "aggregate": {{
    "technical_score": 70,
    "communication_score": 80,
    "overall_score": 75
  }}
}}"""


def performance_summary_prompt(report_data: dict) -> str:
    """Prompt for generating an executive performance summary."""
    profile = report_data.get("candidate_profile", {})
    voice = report_data.get("voice_analysis", {})
    content = report_data.get("content_analysis", {})
    meta = report_data.get("session_metadata", {})
    verdict = report_data.get("verdict", {})

    return f"""You are a senior career coach writing an executive performance summary for an interview candidate.

Interview Performance Data:
- Role: {profile.get('role', 'Unknown')}
- Level: {profile.get('level', 'Unknown')}
- Overall Score: {report_data.get('overall_score', 0)}/100
- Verdict: {verdict.get('recommendation', 'N/A')}

Voice Analysis:
- Speaking Pace: {voice.get('speaking_pace_wpm', 0)} WPM
- Total Filler Words: {voice.get('total_filler_words', 0)}
- Clarity Score: {voice.get('clarity_score', 0)}/100
- Engagement Score: {voice.get('engagement_score', 0)}/100
- Confidence Score: {meta.get('confidence_score', 0)}/100

Content Analysis:
- Average Answer Score: {content.get('average_score', 0)}/100
- Relevance: {content.get('relevance_score', 0)}/100
- Depth: {content.get('depth_score', 0)}/100
- STAR Method: {content.get('star_method_score', 0)}/100

Write a 4-6 sentence professional summary covering:
1. Overall candidate assessment
2. Key communication strengths or issues
3. Technical knowledge evaluation
4. Specific areas for improvement

Write in third person. Be specific and constructive. Do NOT use markdown formatting or bullet points.
Return ONLY the plain-text summary paragraph, nothing else."""


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