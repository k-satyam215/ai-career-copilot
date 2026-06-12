from llm_provider import get_llm

NUM_QUESTIONS = 15
MIN_WORDS = 5
MAX_WORDS = 25
RESUME_CONTEXT_CHUNKS = 8

FALLBACK_QUESTIONS = [
    "What specific component did you implement in one project, and why was that technical approach chosen?",
    "Which part of your project codebase required the most optimization, and how did you validate improvements?",
    "How did you decide chunk size or retrieval strategy in your project, and what trade-offs did you observe?",
    "What failure or limitation did you notice in your system, and how did you mitigate it technically?",
    "Which tool or library choice had the biggest impact on your project outcome, and why?",
]


def build_prompt(role: str, jd_text: str, resume_context: str) -> str:
    return f"""You are a strict technical interviewer.

ROLE: {role}

JOB DESCRIPTION:
{jd_text}

CANDIDATE RESUME CONTEXT:
{resume_context}

STRICT RULES:
- EXACTLY {NUM_QUESTIONS} questions
- ONE sentence per question
- {MIN_WORDS}-{MAX_WORDS} words per question
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
{NUM_QUESTIONS}. Question?
"""


def parse_questions(raw_text: str) -> list:
    """Extract numbered questions matching the word-count constraint."""
    questions = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line or "?" not in line:
            continue
        if not line[0].isdigit():
            continue

        question = line.split(".", 1)[-1].strip()
        word_count = len(question.split())
        if MIN_WORDS <= word_count <= MAX_WORDS:
            questions.append(question)

    return questions


def fill_with_fallback(questions: list) -> list:
    """Pad the question list up to NUM_QUESTIONS using fallback questions."""
    i = 0
    while len(questions) < NUM_QUESTIONS:
        questions.append(FALLBACK_QUESTIONS[i % len(FALLBACK_QUESTIONS)])
        i += 1
    return questions[:NUM_QUESTIONS]


def llm_interview_agent(state):
    """Generates resume + JD grounded interview questions."""
    llm = get_llm()

    role = state["role"]
    jd_text = state["jd_text"]
    resume_context = "\n".join(state["resume_chunks"][:RESUME_CONTEXT_CHUNKS])

    response = llm.invoke(build_prompt(role, jd_text, resume_context))
    questions = parse_questions(response.content)
    questions = fill_with_fallback(questions)

    state["interview_questions"] = questions
    return state
