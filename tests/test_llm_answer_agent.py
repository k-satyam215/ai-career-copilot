"""Tests for llm_answer_agent (LLM mocked)."""
from unittest.mock import MagicMock, patch

import pytest

from agents.llm_answer_agent import (
    FALLBACK_ANSWER,
    MAX_SENTENCES,
    MAX_WORDS,
    enforce_constraints,
    llm_answer_agent,
    get_relevant_chunks,
    parse_batch_response,
)


def _make_mock_llm(text):
    mock = MagicMock()
    r = MagicMock()
    r.content = text
    mock.invoke.return_value = r
    return mock


GOOD_ANSWER = "I built the retrieval component using FAISS and LangChain for document search."

BATCH_RESPONSE = """A1. I used LangGraph and FastAPI to build the pipeline with retry loops.
A2. The retry loop triggers on test failure and has a max of 3 cycles."""


# ---- enforce_constraints ---------------------------------------------------

def test_enforce_short_answer_passes():
    result = enforce_constraints(GOOD_ANSWER)
    assert len(result.split()) <= MAX_WORDS + 2


def test_enforce_trims_to_max_sentences():
    long = "Sentence one. Sentence two here. Sentence three here too. Sentence four also. Sentence five."
    result = enforce_constraints(long)
    assert result.count(".") <= MAX_SENTENCES + 1


def test_enforce_generic_fallback_raises():
    with pytest.raises(ValueError):
        enforce_constraints("I worked on this feature and it was great.")


def test_enforce_empty_string():
    result = enforce_constraints("")
    assert result == "."


# ---- get_relevant_chunks ---------------------------------------------------

def test_get_relevant_chunks_returns_top_k():
    chunks = ["LangGraph retry loop", "FastAPI endpoint", "unrelated content here", "LangGraph agent graph"]
    result = get_relevant_chunks("LangGraph retry", chunks, top_k=2)
    assert len(result) == 2


def test_get_relevant_chunks_empty():
    result = get_relevant_chunks("anything", [], top_k=3)
    assert result == []


# ---- parse_batch_response --------------------------------------------------

def test_parse_batch_response_parses_correctly():
    answers = parse_batch_response(BATCH_RESPONSE, 2)
    assert len(answers) == 2
    assert "LangGraph" in answers[0]


def test_parse_batch_response_fallback_on_missing():
    answers = parse_batch_response("A1. Only one answer here.", 3)
    assert len(answers) == 3
    assert answers[1] == FALLBACK_ANSWER
    assert answers[2] == FALLBACK_ANSWER


# ---- agent integration -----------------------------------------------------

def test_agent_returns_answers(state_with_questions):
    with patch("agents.llm_answer_agent.get_llm", return_value=_make_mock_llm(BATCH_RESPONSE)):
        result = llm_answer_agent(state_with_questions)
    assert "interview_answers" in result
    assert len(result["interview_answers"]) == len(state_with_questions["interview_questions"])


def test_agent_uses_fallback_on_llm_error(state_with_questions):
    mock = MagicMock()
    mock.invoke.side_effect = Exception("API down")
    with patch("agents.llm_answer_agent.get_llm", return_value=mock):
        result = llm_answer_agent(state_with_questions)
    assert all(a == FALLBACK_ANSWER for a in result["interview_answers"])


def test_agent_empty_questions(base_state):
    base_state["interview_questions"] = []
    with patch("agents.llm_answer_agent.get_llm", return_value=_make_mock_llm(BATCH_RESPONSE)):
        result = llm_answer_agent(base_state)
    assert result["interview_answers"] == []


def test_agent_empty_evidence(base_state):
    base_state["interview_questions"] = ["How did you implement the system?"]
    base_state["evidence_chunks"] = []
    base_state["resume_chunks"] = ["Some resume content about LangGraph and FastAPI."]
    with patch("agents.llm_answer_agent.get_llm", return_value=_make_mock_llm(BATCH_RESPONSE)):
        result = llm_answer_agent(base_state)
    assert "interview_answers" in result
    assert len(result["interview_answers"]) == 1


def test_agent_state_returned(state_with_questions):
    with patch("agents.llm_answer_agent.get_llm", return_value=_make_mock_llm(BATCH_RESPONSE)):
        result = llm_answer_agent(state_with_questions)
    assert result is state_with_questions
