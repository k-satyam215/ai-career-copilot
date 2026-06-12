from llm_provider import get_llm

MAX_WORDS = 40
MAX_SENTENCES = 2
EVIDENCE_CONTEXT_CHUNKS = 4

FALLBACK_ANSWER = (
    "I built this component as part of my project and focused on making it work correctly."
)


def build_prompt(question: str, context: str) -> str:
    return f"""You are answering in a real technical interview as the person who BUILT the project below.

RESUME CONTEXT (THE ONLY SOURCE OF TRUTH — use ONLY tools/technologies explicitly named here):
{context}

STRICT RULES (MANDATORY):
- Answer ONLY using technologies, tools, and details that appear in the RESUME CONTEXT above
- Do NOT mention any tool, library, or technique (e.g. ChromaDB, AWS Lambda, LoRA, ROUGE/BLEU, Pinecone)
  unless it is explicitly written in the RESUME CONTEXT
- NEVER say "I didn't implement X" or describe what you did NOT do —
  if the question asks about something not in the RESUME CONTEXT,
  confidently describe what you DID build instead, without mentioning the gap
- Answer in AT MOST {MAX_SENTENCES} sentences
- Answer in AT MOST {MAX_WORDS} words total
- Be direct and conversational
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
