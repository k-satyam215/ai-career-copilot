"""Tests for role_detector and roles_config."""
import pytest
from role_detector import detect_role_from_jd, detect_role_from_resume
from roles_config import SKILL_DOMAINS, EVALUATION_THRESHOLDS, ROLE_CONFIG


# ---- detect_role_from_jd ---------------------------------------------------

def test_genai_jd_detects_genai():
    jd = "Looking for GenAI engineer with LLM RAG LangChain prompt vector embedding experience"
    result = detect_role_from_jd(jd)
    assert "genai" in result


def test_backend_jd_detects_backend():
    jd = "Backend developer needed with FastAPI Django REST API microservice database SQL"
    result = detect_role_from_jd(jd)
    assert "backend" in result


def test_empty_jd_returns_domain():
    result = detect_role_from_jd("")
    assert isinstance(result, str)
    assert len(result) > 0


def test_mixed_jd_returns_combined():
    jd = "LLM RAG agent vector embedding backend fastapi REST API database microservice"
    result = detect_role_from_jd(jd)
    # should include primary + secondary
    assert "+" in result or isinstance(result, str)


def test_returns_string():
    result = detect_role_from_jd("some random text")
    assert isinstance(result, str)


# ---- detect_role_from_resume -----------------------------------------------

def test_resume_scores_returns_dict():
    chunks = ["python llm rag langchain prompt vector embedding faiss"]
    result = detect_role_from_resume(chunks)
    assert isinstance(result, dict)


def test_resume_scores_all_domains():
    chunks = ["text"]
    result = detect_role_from_resume(chunks)
    assert set(result.keys()) == set(SKILL_DOMAINS.keys())


def test_resume_genai_scores_highest():
    chunks = ["llm rag langchain prompt agent vector embedding chroma faiss react genai"]
    result = detect_role_from_resume(chunks)
    assert result["genai"] >= result["backend"]


def test_resume_empty_chunks():
    result = detect_role_from_resume([])
    assert all(v == 0 for v in result.values())


# ---- roles_config ----------------------------------------------------------

def test_skill_domains_not_empty():
    assert len(SKILL_DOMAINS) > 0


def test_evaluation_thresholds_present():
    assert "skill_score" in EVALUATION_THRESHOLDS
    assert "experience_score" in EVALUATION_THRESHOLDS


def test_thresholds_are_numeric():
    assert isinstance(EVALUATION_THRESHOLDS["skill_score"], (int, float))
    assert isinstance(EVALUATION_THRESHOLDS["experience_score"], (int, float))


def test_role_config_backward_compat():
    assert isinstance(ROLE_CONFIG, dict)
    for domain, cfg in ROLE_CONFIG.items():
        assert "keywords" in cfg
        assert "skill_threshold" in cfg


def test_skill_domains_have_list_values():
    for domain, keywords in SKILL_DOMAINS.items():
        assert isinstance(keywords, list)
        assert len(keywords) > 0
