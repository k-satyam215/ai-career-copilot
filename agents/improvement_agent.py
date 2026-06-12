from llm_provider import get_llm

FOCUS_KEYWORDS = [
    "project", "system", "pipeline", "built", "implement", "design",
    "develop", "architecture", "module", "api", "application",
]
MIN_LINE_WORDS = 7
MIN_BULLET_WORDS = 8
MAX_SUGGESTIONS = 3
JD_CONTEXT_CHARS = 300


def is_eligible_line(text: str) -> bool:
    """A resume line is eligible for rewriting if it's substantial and
    references project/system-building work."""
    words = text.split()
    if len(words) < MIN_LINE_WORDS:
        return False
    return any(keyword in text.lower() for keyword in FOCUS_KEYWORDS)


def build_prompt(resume_line: str, jd_text: str) -> str:
    return f"""You are helping improve a FRESHER's resume.

ORIGINAL RESUME LINE:
{resume_line}

JOB DESCRIPTION CONTEXT:
{jd_text[:JD_CONTEXT_CHARS]}

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


def is_valid_bullet(bullet: str) -> bool:
    if len(bullet.split()) < MIN_BULLET_WORDS:
        return False
    lowered = bullet.lower()
    return "add" not in lowered and "should" not in lowered


def improvement_agent(state):
    """Generates resume-ready improvement bullets (not instructions)."""
    resume_chunks = state.get("resume_chunks", [])
    jd_text = state.get("jd_text", "")
    llm = get_llm()

    suggestions = []

    for chunk in resume_chunks:
        text = chunk.strip()

        if not is_eligible_line(text):
            continue

        try:
            response = llm.invoke(build_prompt(text, jd_text))
            bullet = response.content.strip()
        except Exception:
            continue

        if is_valid_bullet(bullet):
            suggestions.append(bullet)

        if len(suggestions) >= MAX_SUGGESTIONS:
            break

    state["improvement_suggestions"] = suggestions
    return state
