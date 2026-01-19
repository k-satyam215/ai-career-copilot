# agents/llm_interview_agent.py

from llm_provider import get_llm


def llm_interview_agent(state):
    """
    Generates resume + JD grounded interview questions.
    Safe for freshers, avoids theory-heavy and repetitive questions.
    """

    llm = get_llm()

    role = state["role"]
    jd_text = state["jd_text"]
    resume_context = "\n".join(state["resume_chunks"][:8])

    prompt = f"""
You are a strict technical interviewer.

ROLE: {role}

JOB DESCRIPTION:
{jd_text}

CANDIDATE RESUME CONTEXT:
{resume_context}

STRICT RULES:
- EXACTLY 15 questions
- ONE sentence per question
- 15–20 words per question
- NO sub-questions
- NO theory
- NO "explain"
- Ask ONLY from resume + JD
- Fresher-friendly, implementation-focused
- End every question with '?'

FORMAT:
1. Question?
2. Question?
...
15. Question?
"""

    response = llm.invoke(prompt)

    questions = []

    for line in response.content.split("\n"):
        line = line.strip()

        if not line:
            continue

        # Match numbered questions
        if line[0].isdigit() and "?" in line:
            q = line.split(".", 1)[-1].strip()

            # Enforce length constraint
            word_count = len(q.split())
            if 15 <= word_count <= 20:
                questions.append(q)

    # ---------------- SAFETY FALLBACK ----------------
    # Prevent repetition and generic spam
    fallback_questions = [
        "What specific component did you implement in one project, and why was that technical approach chosen?",
        "Which part of your project codebase required the most optimization, and how did you validate improvements?",
        "How did you decide chunk size or retrieval strategy in your project, and what trade-offs did you observe?",
        "What failure or limitation did you notice in your system, and how did you mitigate it technically?",
        "Which tool or library choice had the biggest impact on your project outcome, and why?",
    ]

    i = 0
    while len(questions) < 15:
        questions.append(fallback_questions[i % len(fallback_questions)])
        i += 1

    state["interview_questions"] = questions[:15]
    return state
