SKILL_GROUPS = {
    "python": ["python"],
    "genai": ["llm", "rag", "langchain", "langgraph", "prompt", "agent", "react pattern"],
    "vector": ["embedding", "vector", "faiss", "chroma", "pinecone", "bm25"],
    "backend": ["api", "backend", "flask", "django", "fastapi", "microservice"],
    "projects": ["project", "built", "developed", "delivered", "shipped"],
    "devops": ["docker", "kubernetes", "ci/cd", "github actions", "deployment"],
    "testing": ["pytest", "test", "coverage", "tests"],
    "data": ["sql", "database", "postgresql", "redis", "etl"],
    "ml": ["machine learning", "deep learning", "pytorch", "transformers", "nlp"],
    "security": ["security", "guardrails", "redaction", "injection", "authentication"],
}

POINTS_PER_GROUP = 15
MAX_SKILL_SCORE = 100


def skill_agent(state):
    """Deterministic skill-score agent.

    Scores resume content against predefined skill domains.
    Score is capped to keep results realistic for fresher candidates.
    """
    text = " ".join(state["resume_chunks"]).lower()

    matched_groups = [
        group_name
        for group_name, group_keywords in SKILL_GROUPS.items()
        if any(keyword in text for keyword in group_keywords)
    ]
    missing_groups = [g for g in SKILL_GROUPS if g not in matched_groups]

    score = len(matched_groups) * POINTS_PER_GROUP

    state["skill_score"] = min(score, MAX_SKILL_SCORE)
    state["skill_breakdown"] = {
        "matched": matched_groups,
        "missing": missing_groups,
    }
    return state
