from llm_provider import get_llm

MAX_WORDS = 80
MAX_SENTENCES = 4
EVIDENCE_CONTEXT_CHUNKS = 8

FALLBACK_ANSWER = (
    "I built this component as part of my project, focusing on reliability and correctness."
)


def get_relevant_chunks(question: str, all_chunks: list, top_k: int = 6) -> list:
    q_words = set(question.lower().split())
    scored = [(len(set(c.lower().split()) & q_words), c) for c in all_chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def build_batch_prompt(questions: list, all_chunks: list, evidence: list) -> str:
    # Build full resume context once
    context = "\n\n".join(all_chunks[:12])

    numbered_questions = "\n".join(
        f"Q{i+1}. {q}" for i, q in enumerate(questions)
    )

    return f"""You are answering in a real technical interview as the candidate who BUILT these projects.

RESUME CONTEXT (use ONLY information from here):
{context}

RULES:
- Answer ONLY using tools/technologies explicitly in the RESUME CONTEXT
- Each answer: 3-4 sentences, 60-80 words, conversational, first person
- Be specific — mention actual tools, numbers, architecture decisions from resume
- If a question asks about something not in context, describe what you DID build instead
- Do NOT say "I didn't use X" or mention gaps
- Format: Answer each question as "A1.", "A2.", etc. on separate lines

QUESTIONS:
{numbered_questions}

Provide all answers now:
"""


def enforce_constraints(raw_text: str) -> str:
    sentences = [s.strip() for s in raw_text.replace("\n", " ").split(".") if s.strip()]
    sentences = sentences[:MAX_SENTENCES]
    text = ". ".join(sentences) + "." if sentences else "."
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS]) + "."
    return text


def parse_batch_response(response_text: str, num_questions: int) -> list:
    """Parse A1. A2. ... format from batch LLM response."""
    import re
    answers = []
    for i in range(1, num_questions + 1):
        pattern = rf"A{i}\.\s*(.*?)(?=A{i+1}\.|$)"
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            raw = match.group(1).strip()
            answers.append(enforce_constraints(raw))
        else:
            answers.append(FALLBACK_ANSWER)
    return answers


def llm_answer_agent(state):
    """Generates grounded interview answers in a single LLM batch call."""
    questions = state.get("interview_questions", [])
    all_chunks = state.get("full_resume_chunks") or state.get("resume_chunks", [])
    evidence = state.get("evidence_chunks", [])

    if not questions:
        state["interview_answers"] = []
        return state

    llm = get_llm()

    try:
        prompt = build_batch_prompt(questions, all_chunks, evidence)
        response = llm.invoke(prompt)
        answers = parse_batch_response(response.content.strip(), len(questions))
    except Exception:
        answers = [FALLBACK_ANSWER] * len(questions)

    state["interview_answers"] = answers
    return state
