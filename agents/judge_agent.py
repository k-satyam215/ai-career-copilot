from roles_config import EVALUATION_THRESHOLDS


def judge_agent(state):
    """Final verdict agent based on deterministic score thresholds."""
    skill_score = state.get("skill_score", 0)
    experience_score = state.get("experience_score", 0)

    skill_ok = skill_score >= EVALUATION_THRESHOLDS["skill_score"]
    experience_ok = experience_score >= EVALUATION_THRESHOLDS["experience_score"]

    state["verdict"] = (
        "Interview Ready (Fresher)" if skill_ok and experience_ok else "Needs Improvement"
    )

    gaps = []
    if not skill_ok:
        gap = EVALUATION_THRESHOLDS["skill_score"] - skill_score
        gaps.append(f"skill score is {gap:.0f} points below the {EVALUATION_THRESHOLDS['skill_score']} threshold")
    if not experience_ok:
        gap = EVALUATION_THRESHOLDS["experience_score"] - experience_score
        gaps.append(
            f"experience score is {gap:.0f} points below the {EVALUATION_THRESHOLDS['experience_score']} threshold"
        )
    state["verdict_reason"] = (
        "Meets both skill and experience thresholds for this role."
        if not gaps
        else "; ".join(gaps).capitalize() + "."
    )
    return state
