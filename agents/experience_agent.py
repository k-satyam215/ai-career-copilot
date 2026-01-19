def experience_agent(state):
    text = " ".join(state["resume_chunks"]).lower()

    score = 0

    indicators = [
        "project", "built", "implemented",
        "developed", "designed", "system",
        "pipeline", "application"
    ]

    for i in indicators:
        if i in text:
            score += 8

    state["experience_score"] = min(score, 100)
    return state
