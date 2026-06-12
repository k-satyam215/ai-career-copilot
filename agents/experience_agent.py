EXPERIENCE_INDICATORS = [
    "project",
    "built",
    "implemented",
    "developed",
    "designed",
    "system",
    "pipeline",
    "application",
]

POINTS_PER_INDICATOR = 8
MAX_EXPERIENCE_SCORE = 100


def experience_agent(state):
    """Deterministic experience-score agent based on action-verb density."""
    text = " ".join(state["resume_chunks"]).lower()

    score = sum(
        POINTS_PER_INDICATOR for indicator in EXPERIENCE_INDICATORS if indicator in text
    )

    state["experience_score"] = min(score, MAX_EXPERIENCE_SCORE)
    return state
