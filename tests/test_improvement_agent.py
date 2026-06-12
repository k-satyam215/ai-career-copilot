"""Tests for improvement_agent (LLM mocked)."""
import pytest
from unittest.mock import MagicMock, patch
from agents.improvement_agent import (
    improvement_agent,
    is_eligible_line,
    build_prompt,
    is_valid_bullet,
)


# ---- Pure function tests ---------------------------------------------------

def test_eligible_line_with_project():
    assert is_eligible_line("Built a RAG pipeline for document retrieval system")


def test_eligible_line_with_implemented():
    assert is_eligible_line("Implemented the authentication module using JWT tokens")


def test_eligible_line_with_designed():
    assert is_eligible_line("Designed the microservices architecture for production system")


def test_ineligible_short_line():
    assert not is_eligible_line("Python developer")


def test_ineligible_no_keyword():
    assert not is_eligible_line("I am a motivated software engineering graduate student")


def test_build_prompt_contains_resume_line():
    prompt = build_prompt("Built a pipeline for data", "AI engineer role")
    assert "Built a pipeline for data" in prompt


def test_build_prompt_contains_jd():
    prompt = build_prompt("Built a pipeline", "Looking for GenAI engineer")
    assert "Looking for GenAI engineer" in prompt


def test_build_prompt_truncates_jd():
    long_jd = "x" * 1000
    prompt = build_prompt("some line", long_jd)
    # jd should be truncated in prompt
    assert len(prompt) < len(long_jd) + 500


def test_valid_bullet_accepted():
    assert is_valid_bullet("Developed a scalable RAG pipeline using LangChain and FAISS for search")


def test_invalid_bullet_too_short():
    assert not is_valid_bullet("Good work")


def test_invalid_bullet_with_add():
    assert not is_valid_bullet("Add more experience to your resume for better results")


def test_invalid_bullet_with_should():
    assert not is_valid_bullet("You should improve your resume bullet point now")


# ---- Agent integration tests (LLM mocked) ---------------------------------

def _make_mock_llm(response_text):
    mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = response_text
    mock.invoke.return_value = mock_resp
    return mock


def test_agent_returns_suggestions(base_state):
    good_bullet = "Developed a RAG pipeline using LangChain and FAISS achieving semantic retrieval"
    with patch("agents.improvement_agent.get_llm", return_value=_make_mock_llm(good_bullet)):
        result = improvement_agent(base_state)
    assert "improvement_suggestions" in result
    assert isinstance(result["improvement_suggestions"], list)


def test_agent_filters_invalid_bullets(base_state):
    with patch("agents.improvement_agent.get_llm", return_value=_make_mock_llm("add more")):
        result = improvement_agent(base_state)
    # "add more" is invalid, so 0 suggestions
    assert result["improvement_suggestions"] == []


def test_agent_caps_at_max_suggestions(base_state):
    good = "Implemented a scalable FastAPI backend serving ten thousand daily requests"
    with patch("agents.improvement_agent.get_llm", return_value=_make_mock_llm(good)):
        result = improvement_agent(base_state)
    assert len(result["improvement_suggestions"]) <= 3


def test_agent_handles_llm_exception(base_state):
    mock = MagicMock()
    mock.invoke.side_effect = Exception("API error")
    with patch("agents.improvement_agent.get_llm", return_value=mock):
        result = improvement_agent(base_state)
    assert result["improvement_suggestions"] == []


def test_agent_empty_chunks():
    state = {"resume_chunks": [], "jd_text": "AI engineer", "full_resume_chunks": []}
    mock = _make_mock_llm("Built a robust system using LangChain for document retrieval")
    with patch("agents.improvement_agent.get_llm", return_value=mock):
        result = improvement_agent(state)
    assert result["improvement_suggestions"] == []
