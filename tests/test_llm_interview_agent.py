"""Tests for llm_interview_agent (LLM mocked)."""
import pytest
from unittest.mock import MagicMock, patch
from agents.llm_interview_agent import (
    llm_interview_agent,
    parse_questions,
    fill_with_fallback,
    build_prompt,
    NUM_QUESTIONS,
)


def _make_mock_llm(text):
    mock = MagicMock()
    r = MagicMock()
    r.content = text
    mock.invoke.return_value = r
    return mock


def _numbered_questions(n, words=16):
    return "\n".join(
        f"{i+1}. How did you implement component {i+1} in your RAG pipeline project?"
        for i in range(n)
    )


# ---- parse_questions -------------------------------------------------------

def test_parse_questions_basic():
    raw = _numbered_questions(5)
    result = parse_questions(raw)
    assert len(result) == 5


def test_parse_questions_skips_no_question_mark():
    raw = "1. This has no question mark\n2. This one does have a question mark?"
    result = parse_questions(raw)
    assert len(result) == 1


def test_parse_questions_skips_non_numbered():
    raw = "Some intro text\n1. How did you build your RAG pipeline for document retrieval?\nTrailing text"
    result = parse_questions(raw)
    assert len(result) >= 1


def test_parse_questions_empty():
    assert parse_questions("") == []


# ---- fill_with_fallback ----------------------------------------------------

def test_fill_pads_to_num_questions():
    result = fill_with_fallback([])
    assert len(result) == NUM_QUESTIONS


def test_fill_truncates_over_limit():
    long = [f"Q{i}?" for i in range(NUM_QUESTIONS + 10)]
    result = fill_with_fallback(long)
    assert len(result) == NUM_QUESTIONS


def test_fill_preserves_existing():
    existing = ["How did you implement the search component in your RAG pipeline project?"]
    result = fill_with_fallback(existing)
    assert result[0] == existing[0]


# ---- build_prompt ----------------------------------------------------------

def test_build_prompt_contains_role():
    p = build_prompt("GenAI Engineer", "AI role", "resume context")
    assert "GenAI Engineer" in p


def test_build_prompt_contains_jd():
    p = build_prompt("role", "needs LangChain experience", "resume")
    assert "LangChain" in p


# ---- agent integration -----------------------------------------------------

def test_agent_returns_15_questions(base_state):
    raw = _numbered_questions(15)
    with patch("agents.llm_interview_agent.get_llm", return_value=_make_mock_llm(raw)):
        result = llm_interview_agent(base_state)
    assert len(result["interview_questions"]) == NUM_QUESTIONS


def test_agent_pads_with_fallback(base_state):
    raw = _numbered_questions(3)  # only 3 valid
    with patch("agents.llm_interview_agent.get_llm", return_value=_make_mock_llm(raw)):
        result = llm_interview_agent(base_state)
    assert len(result["interview_questions"]) == NUM_QUESTIONS


def test_agent_state_returned(base_state):
    with patch("agents.llm_interview_agent.get_llm", return_value=_make_mock_llm(_numbered_questions(15))):
        result = llm_interview_agent(base_state)
    assert result is base_state
