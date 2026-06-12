"""Tests for MCP server tools."""
import pytest
from unittest.mock import patch, MagicMock


MOCK_CHUNKS = [
    "Built a RAG pipeline using LangChain and FAISS for document retrieval",
    "Implemented FastAPI backend with LLM inference endpoint",
]

MOCK_RESULT = {
    "role": "GenAI Engineer",
    "skill_score": 70.0,
    "experience_score": 64.0,
    "ats_issues": [],
    "improvement_suggestions": ["Developed scalable RAG pipeline using LangChain."],
    "interview_questions": ["How did you implement the retrieval component?"],
    "interview_answers": ["I used LangChain and FAISS for vector search."],
    "verdict": "Interview Ready (Fresher)",
}


def test_parse_resume_tool():
    from mcp_server.server import parse_resume
    with patch("mcp_server.server.load_resume", return_value="resume text content"), \
         patch("mcp_server.server.chunk_resume", return_value=MOCK_CHUNKS):
        result = parse_resume("fake.pdf")
    assert result["chunk_count"] == 2
    assert result["chunks"] == MOCK_CHUNKS


def test_parse_resume_returns_dict():
    from mcp_server.server import parse_resume
    with patch("mcp_server.server.load_resume", return_value="text"), \
         patch("mcp_server.server.chunk_resume", return_value=MOCK_CHUNKS):
        result = parse_resume("fake.pdf")
    assert isinstance(result, dict)
    assert "chunk_count" in result
    assert "chunks" in result


def test_retrieve_resume_evidence_tool():
    from mcp_server.server import retrieve_resume_evidence
    with patch("mcp_server.server.load_resume", return_value="resume text"), \
         patch("mcp_server.server.chunk_resume", return_value=MOCK_CHUNKS), \
         patch("mcp_server.server.detect_role_from_jd", return_value="genai"), \
         patch("mcp_server.server._extract_retrieval_keywords", return_value=["llm", "rag"]), \
         patch("mcp_server.server.retrieve_evidence", return_value=MOCK_CHUNKS):
        result = retrieve_resume_evidence("fake.pdf", "AI engineer role")
    assert "role" in result
    assert "evidence_chunks" in result
    assert result["role"] == "genai"


def test_evaluate_resume_against_jd_tool():
    from mcp_server.server import evaluate_resume_against_jd
    with patch("mcp_server.server.evaluate_resume", return_value=MOCK_RESULT):
        result = evaluate_resume_against_jd("fake.pdf", "AI engineer role")
    assert result["verdict"] == "Interview Ready (Fresher)"
    assert result["skill_score"] == 70.0


def test_evaluate_resume_against_jd_returns_all_keys():
    from mcp_server.server import evaluate_resume_against_jd
    with patch("mcp_server.server.evaluate_resume", return_value=MOCK_RESULT):
        result = evaluate_resume_against_jd("fake.pdf", "role")
    for key in MOCK_RESULT:
        assert key in result
