from llm_provider import get_llm

MAX_WORDS = 40
MAX_SENTENCES = 2
EVIDENCE_CONTEXT_CHUNKS = 8  # increased from 4

FALLBACK_ANSWER = (
    "I built this component as part of my project and focused on making it work correctly."
)


def build_prompt(question: str, context: str) -> str:
    return f"""You are answering in a real technical interview as the person who BUILT the project below.

RESUME CONTEXT (THE ONLY SOURCE OF TRUTH — use ONLY tools/technologies explicitly named here):
{context}

STRICT RULES (MANDATORY):
- Answer ONLY using technologies, tools, and details that appear in the RESUME CONTEXT above
- Do NOT mention any tool, library, or technique unless it is explicitly written in the RESUME CONTEXT
- NEVER say "I didn't implement X" or describe what you did NOT do —
  if the question asks about something not in the RESUME CONTEXT,
  confidently describe what you DID build instead, without mentioning the gap
- Answer in AT MOST {MAX_SENTENCES} sentences
- Answer in AT MOST {MAX_WORDS} words total
- Be direct and conversational, first person ("I")
- No parentheses, no filler text

Question:
{question}

Answer:
"""


def get_relevant_chunks(question: str, all_chunks: list, top_k: int = 6) -> list:
    """Return chunks most relevant to this specific question using keyword overlap."""
    q_words = set(question.lower().split())
    scored = []
    for chunk in all_chunks:
        chunk_words = set(chunk.lower().split())
        score = len(q_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def enforce_constraints(raw_text: str) -> str:
    sentences = [s.strip() for s in raw_text.replace("\n", " ").split(".") if s.strip()]
    sentences = sentences[:MAX_SENTENCES]

    text = ". ".join(sentences)
    if text:
        text += "."
    else:
        text = "."

    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS]) + "."

    if "worked on this feature" in text.lower():
        raise ValueError("Generic fallback triggered")

    return text


def llm_answer_agent(state):
    """Generates concise, fresher-safe interview answers grounded in resume evidence."""
    questions = state.get("interview_questions", [])
    all_chunks = state.get("full_resume_chunks") or state.get("resume_chunks", [])
    evidence = state.get("evidence_chunks", [])

    if not questions:
        state["interview_answers"] = []
        return state

    llm = get_llm()
    answers = []

    for question in questions:
        # get question-specific relevant chunks for better grounding
        relevant = get_relevant_chunks(question, all_chunks, top_k=6)
        # also include top evidence chunks for general context
        combined = relevant + [c for c in evidence[:4] if c not in relevant]
        context = "\n\n".join(combined[:EVIDENCE_CONTEXT_CHUNKS])

        try:
            response = llm.invoke(build_prompt(question, context))
            answer = enforce_constraints(response.content.strip())
        except Exception:
            answer = FALLBACK_ANSWER

        answers.append(answer)

    state["interview_answers"] = answers
    return state
