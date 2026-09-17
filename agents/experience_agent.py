EXPERIENCE_INDICATORS = [
    "project",
    "built",
    "build",
    "implement",
    "develop",
    "design",
    "system",
    "pipeline",
    "application",
    "deploy",
    "integrat",
    "optimiz",
    "deliver",
    "engineer",
    "automat",
    "reduc",
    "improv",
    "architect",
    "harden",
    "orchestrat",
    "scal",
]

POINTS_PER_INDICATOR = 6
MAX_EXPERIENCE_SCORE = 100


def experience_agent(state):
    """Deterministic experience-score agent based on action-verb density."""
    text = " ".join(state["resume_chunks"]).lower()

    score = sum(
        POINTS_PER_INDICATOR for indicator in EXPERIENCE_INDICATORS if indicator in text
    )

    state["experience_score"] = min(score, MAX_EXPERIENCE_SCORE)
    return state
