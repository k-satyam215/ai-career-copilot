def skill_agent(state):
    text = " ".join(state["resume_chunks"]).lower()

    score = 0

    skill_groups = {
        "python": ["python"],
        "genai": ["llm", "rag", "langchain", "prompt"],
        "vector": ["embedding", "vector", "faiss", "chroma"],
        "backend": ["api", "backend", "flask", "django"],
        "projects": ["project", "built", "developed"]
    }

    for group in skill_groups.values():
        if any(k in text for k in group):
            score += 15   # controlled increment

    # Fresher cap
    state["skill_score"] = min(score, 70)
    return state
