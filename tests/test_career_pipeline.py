"""Tests for career_pipeline module."""
import pytest
from unittest.mock import patch, MagicMock
from career_pipeline import (
    format_role_label,
    _extract_retrieval_keywords,
    run_agent_pipeline,
    evaluate_resume,
)


# ---- format_role_label -----------------------------------------------------

def test_genai_jd_backend_resume():
    result = format_role_label("genai", "backend")
    assert "Backend" in result and "GenAI" in result


def test_genai_jd_ml_resume():
    result = format_role_label("genai", "ml")
    assert "Machine Learning" in result or "ML" in result


def test_genai_jd_genai_resume():
    result = format_role_label("genai", "genai")
    assert "GenAI" in result


def test_backend_role_label():
    result = format_role_label("backend", "backend")
    assert "Backend" in result


def test_returns_string():
    result = format_role_label("data", "data")
    assert isinstance(result, str)
    assert len(result) > 0


# ---- _extract_retrieval_keywords -------------------------------------------

def test_extracts_genai_keywords():
    keywords = _extract_retrieval_keywords("genai")
    assert isinstance(keywords, list)
    assert len(keywords) > 0


def test_extracts_combined_role():
    keywords = _extract_retrieval_keywords("genai + backend")
    assert len(keywords) > 0


def test_unknown_role_returns_none():
    result = _extract_retrieval_keywords("unknownrole")
    assert result is None


# ---- run_agent_pipeline ----------------------------------------------------

def test_pipeline_runs_all_agents(base_state):
    # Mock LLM agents
    good_bullet = "Built a scalable RAG pipeline using LangChain and FAISS for search"
    good_q = "\n".join(
        f"{i+1}. How did you implement the RAG component {i+1} in your project?"
        for i in range(15)
    )
    good_answer = "I implemented the retrieval component using FAISS and LangChain directly."

    def make_mock(text):
        m = MagicMock()
        r = MagicMock()
        r.content = text
        m.invoke.return_value = r
        return m

    with patch("agents.improvement_agent.get_llm", return_value=make_mock(good_bullet)), \
         patch("agents.llm_interview_agent.get_llm", return_value=make_mock(good_q)), \
         patch("agents.llm_answer_agent.get_llm", return_value=make_mock(good_answer)):
        result = run_agent_pipeline(base_state)

    assert "skill_score" in result
    assert "experience_score" in result
    assert "ats_issues" in result
    assert "verdict" in result


def test_pipeline_state_has_verdict(base_state):
    good_q = "\n".join(
        f"{i+1}. How did you build the pipeline component {i+1} in your system?"
        for i in range(15)
    )
    mock_llm = MagicMock()
    r = MagicMock()
    r.content = good_q
    mock_llm.invoke.return_value = r

    with patch("agents.improvement_agent.get_llm", return_value=mock_llm), \
         patch("agents.llm_interview_agent.get_llm", return_value=mock_llm), \
         patch("agents.llm_answer_agent.get_llm", return_value=mock_llm):
        result = run_agent_pipeline(base_state)

    assert result["verdict"] in ["Interview Ready (Fresher)", "Needs Improvement"]


# ---- evaluate_resume (full end-to-end, file mocked) ------------------------

def test_evaluate_resume_returns_expected_keys():
    mock_resume_text = (
        "Satyam Kumar - GenAI Engineer\n"
        "Education: B.Tech, ABC University, CGPA 8.5\n"
        "Projects: Built RAG pipeline using LangChain and FAISS\n"
        "Implemented FastAPI backend for serving LLM inference\n"
        "Developed agent system using prompt engineering and vector embeddings\n"
        "Skills: Python, LangChain, FastAPI, FAISS, LLM, RAG, prompt engineering\n"
    )
    mock_chunks = [mock_resume_text[i:i+400] for i in range(0, len(mock_resume_text), 400)]

    good_q = "\n".join(
        f"{i+1}. How did you implement the RAG pipeline component {i+1}?"
        for i in range(15)
    )
    good_answer = "I built the component using LangChain and FAISS for document retrieval."
    good_bullet = "Developed a production-ready RAG pipeline using LangChain achieving semantic retrieval"

    mock_llm = MagicMock()
    r = MagicMock()
    mock_llm.invoke.return_value = r

    import itertools
    responses = itertools.cycle([good_bullet, good_q, good_answer])

    def side_effect(prompt):
        resp = MagicMock()
        resp.content = next(responses)
        return resp

    mock_llm.invoke.side_effect = side_effect

    with patch("career_pipeline.load_resume", return_value=mock_resume_text), \
         patch("career_pipeline.chunk_resume", return_value=mock_chunks), \
         patch("career_pipeline.retrieve_evidence", return_value=mock_chunks[:3]), \
         patch("agents.improvement_agent.get_llm", return_value=mock_llm), \
         patch("agents.llm_interview_agent.get_llm", return_value=mock_llm), \
         patch("agents.llm_answer_agent.get_llm", return_value=mock_llm):
        result = evaluate_resume("fake_path.pdf", "AI GenAI LLM RAG engineer role")

    assert "role" in result
    assert "skill_score" in result
    assert "experience_score" in result
    assert "ats_issues" in result
    assert "improvement_suggestions" in result
    assert "interview_questions" in result
    assert "interview_answers" in result
    assert "verdict" in result
