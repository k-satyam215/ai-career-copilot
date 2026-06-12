"""Tests for llm_answer_agent (LLM mocked)."""
import pytest
from unittest.mock import MagicMock, patch
from agents.llm_answer_agent import (
    llm_answer_agent,
    build_prompt,
    enforce_constraints,
    FALLBACK_ANSWER,
    MAX_WORDS,
    MAX_SENTENCES,
)


def _make_mock_llm(text):
    mock = MagicMock()
    r = MagicMock()
    r.content = text
    mock.invoke.return_value = r
    return mock


GOOD_ANSWER = "I built the retrieval component using FAISS and LangChain for document search."


# ---- enforce_constraints ---------------------------------------------------

def test_enforce_short_answer_passes():
    result = enforce_constraints(GOOD_ANSWER)
    assert len(result.split()) <= MAX_WORDS + 2  # small tolerance for period


def test_enforce_trims_to_max_sentences():
    long = "Sentence one. Sentence two here. Sentence three here too. Sentence four also."
    result = enforce_constraints(long)
    # should be max 2 sentences
    assert result.count(".") <= MAX_SENTENCES + 1


def test_enforce_generic_fallback_raises():
    import pytest
    with pytest.raises(ValueError):
        enforce_constraints("I worked on this feature and it was great.")


def test_enforce_empty_string():
    result = enforce_constraints("")
    assert result == "."


# ---- build_prompt ----------------------------------------------------------

def test_build_prompt_contains_question():
    p = build_prompt("How did you build the RAG system?", "context text here")
    assert "How did you build the RAG system?" in p


def test_build_prompt_contains_context():
    p = build_prompt("question?", "LangChain and FAISS usage details")
    assert "LangChain and FAISS" in p


# ---- agent integration -----------------------------------------------------

def test_agent_returns_answers(state_with_questions):
    with patch("agents.llm_answer_agent.get_llm", return_value=_make_mock_llm(GOOD_ANSWER)):
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
    with patch("agents.llm_answer_agent.get_llm", return_value=_make_mock_llm(GOOD_ANSWER)):
        result = llm_answer_agent(base_state)
    assert result["interview_answers"] == []


def test_agent_empty_evidence(base_state):
    base_state["interview_questions"] = ["How did you implement the system?"]
    base_state["evidence_chunks"] = []
    with patch("agents.llm_answer_agent.get_llm", return_value=_make_mock_llm(GOOD_ANSWER)):
        result = llm_answer_agent(base_state)
    assert result["interview_answers"] == []


def test_agent_state_returned(state_with_questions):
    with patch("agents.llm_answer_agent.get_llm", return_value=_make_mock_llm(GOOD_ANSWER)):
        result = llm_answer_agent(state_with_questions)
    assert result is state_with_questions
