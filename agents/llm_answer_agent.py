from llm_provider import get_llm

MAX_WORDS = 35
MAX_SENTENCES = 2
EVIDENCE_CONTEXT_CHUNKS = 1

FALLBACK_ANSWER = (
    "I built this component as part of my project and focused on making it work correctly."
)


def build_prompt(question: str, context: str) -> str:
    return f"""You are answering in a real technical interview.

RESUME CONTEXT (ONLY FACTS YOU MAY USE):
{context}

STRICT RULES (MANDATORY):
- Answer in AT MOST {MAX_SENTENCES} sentences
- Answer in AT MOST {MAX_WORDS} words total
- Be direct and conversational
- Do NOT explain everything
- Do NOT add examples unless necessary
- First person ("I")
- No parentheses, no filler text

Question:
{question}

Answer:
"""


def enforce_constraints(raw_text: str) -> str:
    """Trim model output to the sentence/word limits, raising on generic filler."""
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
    evidence = state.get("evidence_chunks", [])

    if not questions or not evidence:
        state["interview_answers"] = []
        return state

    context = "\n".join(evidence[:EVIDENCE_CONTEXT_CHUNKS])
    llm = get_llm()

    answers = []
    for question in questions:
        try:
            response = llm.invoke(build_prompt(question, context))
            answer = enforce_constraints(response.content.strip())
        except Exception:
            answer = FALLBACK_ANSWER

        answers.append(answer)

    state["interview_answers"] = answers
    return state
