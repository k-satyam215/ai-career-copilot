# agents/judge_agent.py

from roles_config import EVALUATION_THRESHOLDS


def judge_agent(state: dict) -> dict:
    """
    FINAL judge agent
    - Supports hybrid roles (genai + backend)
    - Uses generic thresholds
    - No ROLE_CONFIG dependency
    """

    skill_score = state.get("skill_score", 0)
    experience_score = state.get("experience_score", 0)

    skill_threshold = EVALUATION_THRESHOLDS["skill_score"]
    experience_threshold = EVALUATION_THRESHOLDS["experience_score"]

    if skill_score >= skill_threshold and experience_score >= experience_threshold:
        verdict = "Interview Ready (Fresher)"
    else:
        verdict = "Needs Improvement"

    state["verdict"] = verdict
    return state
