# agents/improvement_agent.py

from llm_provider import get_llm


def improvement_agent(state):
    """
    Generates resume-ready improvement bullets
    (NOT instructions, actual rewritten bullets)
    """

    resume_chunks = state.get("resume_chunks", [])
    jd_text = state.get("jd_text", "")
    llm = get_llm()

    suggestions = []

    for chunk in resume_chunks:
        text = chunk.strip()

        # Skip very short or broken lines
        if len(text.split()) < 8:
            continue

        # Focus only on project / system lines
        keywords = ["project", "system", "pipeline", "built", "implemented", "designed"]
        if not any(k in text.lower() for k in keywords):
            continue

        prompt = f"""
You are helping improve a FRESHER's resume.

ORIGINAL RESUME LINE:
{text}

JOB DESCRIPTION CONTEXT:
{jd_text[:300]}

TASK:
Rewrite this line as a STRONG resume bullet.

RULES:
- Do NOT give instructions
- Output ONLY the improved bullet
- Mention what was built, how, and outcome
- No fake metrics or percentages
- Fresher-safe, realistic impact
- One concise sentence only

Improved Resume Bullet:
"""

        try:
            resp = llm.invoke(prompt)
            bullet = resp.content.strip()

            # Safety filters
            if len(bullet.split()) < 8:
                continue
            if "add" in bullet.lower() or "should" in bullet.lower():
                continue

            suggestions.append(bullet)

        except Exception:
            continue

        if len(suggestions) >= 3:
            break

    state["improvement_suggestions"] = suggestions
    return state
