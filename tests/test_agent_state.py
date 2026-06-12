"""Tests for agents state schema."""
import pytest
from agents.state import AgentState


def test_state_is_typed_dict():
    state: AgentState = {}
    assert isinstance(state, dict)


def test_state_accepts_all_fields():
    state: AgentState = {
        "resume_chunks": ["chunk1"],
        "full_resume_chunks": ["chunk1"],
        "evidence_chunks": ["evidence1"],
        "jd_text": "some jd",
        "role": "genai",
        "skill_score": 70.0,
        "experience_score": 64.0,
        "ats_issues": [],
        "improvement_suggestions": ["improved bullet"],
        "interview_questions": ["question?"],
        "interview_answers": ["answer"],
        "verdict": "Interview Ready (Fresher)",
    }
    assert state["verdict"] == "Interview Ready (Fresher)"
    assert state["skill_score"] == 70.0


def test_state_partial_is_ok():
    state: AgentState = {"jd_text": "AI engineer role", "role": "genai"}
    assert state["jd_text"] == "AI engineer role"


def test_state_resume_chunks_list():
    state: AgentState = {"resume_chunks": ["a", "b", "c"]}
    assert len(state["resume_chunks"]) == 3
