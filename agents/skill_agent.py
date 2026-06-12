SKILL_GROUPS = {
    "python": ["python"],
    "genai": ["llm", "rag", "langchain", "prompt"],
    "vector": ["embedding", "vector", "faiss", "chroma"],
    "backend": ["api", "backend", "flask", "django", "fastapi"],
    "projects": ["project", "built", "developed"],
}

POINTS_PER_GROUP = 15
MAX_SKILL_SCORE = 70


def skill_agent(state):
    """Deterministic skill-score agent.

    Scores resume content against predefined skill domains.
    Score is capped to keep results realistic for fresher candidates.
    """
    text = " ".join(state["resume_chunks"]).lower()

    score = 0
    for group_keywords in SKILL_GROUPS.values():
        if any(keyword in text for keyword in group_keywords):
            score += POINTS_PER_GROUP

    state["skill_score"] = min(score, MAX_SKILL_SCORE)
    return state
