"""Tests for retrieval modules: chunking, hyde, hybrid."""
import pytest
from retrieval.chunking import chunk_resume, CHUNK_SIZE
from retrieval.hyde import generate_hypothetical_profile
from retrieval.hybrid import keyword_search, enhance_query


# ---- chunking.py -----------------------------------------------------------

def test_chunk_single_chunk():
    text = "short resume text"
    result = chunk_resume(text)
    assert result == ["short resume text"]


def test_chunk_splits_long_text():
    text = "a" * (CHUNK_SIZE * 3)
    result = chunk_resume(text)
    assert len(result) == 3


def test_chunk_empty_string():
    assert chunk_resume("") == []


def test_chunk_size_correct():
    text = "x" * (CHUNK_SIZE + 100)
    chunks = chunk_resume(text)
    assert len(chunks[0]) == CHUNK_SIZE
    assert len(chunks[1]) == 100


def test_chunk_preserves_all_text():
    text = "hello world " * 200
    chunks = chunk_resume(text)
    assert "".join(chunks) == text


# ---- hyde.py ---------------------------------------------------------------

def test_hyde_contains_jd():
    jd = "Looking for GenAI engineer with LLM experience"
    result = generate_hypothetical_profile(jd)
    assert jd in result


def test_hyde_returns_string():
    result = generate_hypothetical_profile("test jd")
    assert isinstance(result, str)


def test_hyde_non_empty():
    result = generate_hypothetical_profile("")
    assert len(result) > 0


def test_hyde_contains_candidate_language():
    result = generate_hypothetical_profile("jd text")
    assert "candidate" in result.lower() or "ideal" in result.lower()


# ---- hybrid.py -------------------------------------------------------------

def test_keyword_search_returns_top_k():
    chunks = [f"python fastapi llm chunk {i}" for i in range(10)]
    result = keyword_search("python fastapi", chunks, k=3)
    assert len(result) == 3


def test_keyword_search_ranks_by_overlap():
    chunks = ["python llm rag fastapi", "javascript node react", "python script tool"]
    result = keyword_search("python llm", chunks, k=2)
    assert result[0] == "python llm rag fastapi"


def test_keyword_search_empty_chunks():
    result = keyword_search("python", [], k=3)
    assert result == []


def test_enhance_query_returns_list():
    result = enhance_query("genai engineer", "needs LLM experience")
    assert isinstance(result, list)
    assert len(result) > 0


def test_enhance_query_contains_variants():
    result = enhance_query("query", "jd text")
    assert any("query" in r or "jd text" in r for r in result)


def test_keyword_search_k_larger_than_chunks():
    chunks = ["python llm", "fastapi backend"]
    result = keyword_search("python", chunks, k=10)
    assert len(result) == 2
