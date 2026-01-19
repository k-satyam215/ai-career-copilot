# agents/llm_answer_agent.py

from llm_provider import get_llm


MAX_WORDS = 35
MAX_SENTENCES = 2


def llm_answer_agent(state):
    """
    FINAL GUARANTEED SHORT ANSWERS
    - Max 2 sentences
    - Max 35 words
    - Natural interview tone
    - Fresher-safe
    """

    questions = state.get("interview_questions", [])
    evidence = state.get("evidence_chunks", [])

    if not questions or not evidence:
        state["interview_answers"] = []
        return state

    # Minimal context to avoid long reasoning
    context = "\n".join(evidence[:1])
    llm = get_llm()

    answers = []

    for q in questions:
        prompt = f"""
You are answering in a real technical interview.

RESUME CONTEXT (ONLY FACTS YOU MAY USE):
{context}

STRICT RULES (MANDATORY):
- Answer in AT MOST 2 sentences
- Answer in AT MOST 35 words total
- Be direct and conversational
- Do NOT explain everything
- Do NOT add examples unless necessary
- First person ("I")
- No parentheses, no filler text

Question:
{q}

Answer:
"""

        try:
            resp = llm.invoke(prompt)
            text = resp.content.strip()

            # -------- HARD ENFORCEMENT --------
            # Split sentences
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
            sentences = sentences[:MAX_SENTENCES]

            # Rebuild text
            text = ". ".join(sentences)
            if text:
                text += "."

            # Enforce word limit
            words = text.split()
            if len(words) > MAX_WORDS:
                text = " ".join(words[:MAX_WORDS]) + "."

            # Safety generic filter
            if "worked on this feature" in text.lower():
                raise ValueError("Generic fallback")

        except Exception:
            # Ultra-short safe fallback
            text = "I built this component as part of my project and focused on making it work correctly."

        answers.append(text)

    state["interview_answers"] = answers
    return state
