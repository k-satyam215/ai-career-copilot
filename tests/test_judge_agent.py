"""Tests for judge_agent."""
from agents.judge_agent import judge_agent
from roles_config import EVALUATION_THRESHOLDS


def test_both_pass_interview_ready():
    state = {
        "skill_score": EVALUATION_THRESHOLDS["skill_score"],
        "experience_score": EVALUATION_THRESHOLDS["experience_score"],
    }
    result = judge_agent(state)
    assert result["verdict"] == "Interview Ready (Fresher)"


def test_skill_fail_needs_improvement():
    state = {
        "skill_score": EVALUATION_THRESHOLDS["skill_score"] - 1,
        "experience_score": EVALUATION_THRESHOLDS["experience_score"],
    }
    result = judge_agent(state)
    assert result["verdict"] == "Needs Improvement"


def test_experience_fail_needs_improvement():
    state = {
        "skill_score": EVALUATION_THRESHOLDS["skill_score"],
        "experience_score": EVALUATION_THRESHOLDS["experience_score"] - 1,
    }
    result = judge_agent(state)
    assert result["verdict"] == "Needs Improvement"


def test_both_fail_needs_improvement():
    state = {"skill_score": 0, "experience_score": 0}
    result = judge_agent(state)
    assert result["verdict"] == "Needs Improvement"


def test_high_scores_interview_ready():
    state = {"skill_score": 100, "experience_score": 100}
    result = judge_agent(state)
    assert result["verdict"] == "Interview Ready (Fresher)"


def test_missing_scores_default_zero():
    state = {}
    result = judge_agent(state)
    assert result["verdict"] == "Needs Improvement"


def test_state_returned():
    state = {"skill_score": 70, "experience_score": 70}
    result = judge_agent(state)
    assert result is state


def test_verdict_string_type():
    state = {"skill_score": 70, "experience_score": 70}
    result = judge_agent(state)
    assert isinstance(result["verdict"], str)


def test_exact_threshold_passes():
    state = {
        "skill_score": float(EVALUATION_THRESHOLDS["skill_score"]),
        "experience_score": float(EVALUATION_THRESHOLDS["experience_score"]),
    }
    result = judge_agent(state)
    assert result["verdict"] == "Interview Ready (Fresher)"
