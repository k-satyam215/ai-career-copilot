# agents/interview_agent.py

def interview_question_agent(state):
    """
    Basic Interview Question Agent (Rule-based)
    """

    role = state["role"]
    resume_text = " ".join(state["resume_chunks"]).lower()

    questions = []

    # -------- ROLE-SPECIFIC QUESTIONS --------
    if role == "ai":
        questions.extend([
            "Explain how you would design a RAG pipeline end-to-end.",
            "What challenges have you faced while working with LLMs?",
            "How do you evaluate the quality of retrieved context?"
        ])

    elif role == "backend":
        questions.extend([
            "How do you design scalable REST APIs?",
            "How do you handle database performance issues?",
            "What strategies do you use for backend optimization?"
        ])

    else:
        questions.extend([
            "Explain one complex project you worked on.",
            "How do you approach unfamiliar technical problems?"
        ])

    # -------- GAP-BASED QUESTIONS --------
    if "testing" not in resume_text:
        questions.append(
            "How do you ensure testing and reliability in your projects?"
        )

    if "deployment" not in resume_text:
        questions.append(
            "Can you explain how you deploy your applications?"
        )

    state["interview_questions"] = questions[:8]
    return state
