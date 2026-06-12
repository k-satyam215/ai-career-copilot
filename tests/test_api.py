"""Tests for FastAPI endpoints."""
import io
from unittest.mock import patch

import pytest

MOCK_RESULT = {
    "role": "GenAI Engineer",
    "skill_score": 70.0,
    "experience_score": 64.0,
    "ats_issues": [],
    "improvement_suggestions": ["Built a scalable RAG pipeline using LangChain for semantic search."],
    "interview_questions": ["How did you implement the retrieval component in your RAG pipeline?"],
    "interview_answers": ["I built it using LangChain and FAISS for document retrieval."],
    "verdict": "Interview Ready (Fresher)",
}


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_evaluate_success(client):
    pdf_bytes = b"%PDF-1.4 fake pdf content for testing purposes only"
    with patch("api.main.evaluate_resume", return_value=MOCK_RESULT):
        resp = client.post(
            "/evaluate",
            files={"resume": ("resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"jd_text": "AI GenAI engineer with LLM RAG experience"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "GenAI Engineer"
    assert body["verdict"] == "Interview Ready (Fresher)"


def test_evaluate_wrong_content_type(client):
    with patch("api.main.evaluate_resume", return_value=MOCK_RESULT):
        resp = client.post(
            "/evaluate",
            files={"resume": ("resume.txt", io.BytesIO(b"plain text"), "text/plain")},
            data={"jd_text": "some jd"},
        )
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]


def test_evaluate_empty_jd(client):
    pdf_bytes = b"%PDF fake"
    with patch("api.main.evaluate_resume", return_value=MOCK_RESULT):
        resp = client.post(
            "/evaluate",
            files={"resume": ("resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"jd_text": "   "},
        )
    assert resp.status_code == 400
    assert "Job description" in resp.json()["detail"]


def test_evaluate_response_has_all_fields(client):
    pdf_bytes = b"%PDF fake content"
    with patch("api.main.evaluate_resume", return_value=MOCK_RESULT):
        resp = client.post(
            "/evaluate",
            files={"resume": ("resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"jd_text": "GenAI LLM RAG engineer role"},
        )
    body = resp.json()
    for key in MOCK_RESULT:
        assert key in body


def test_health_response_structure(client):
    resp = client.get("/health")
    assert "status" in resp.json()


def test_api_docs_accessible(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_evaluate_returns_skill_score(client):
    pdf_bytes = b"%PDF fake"
    with patch("api.main.evaluate_resume", return_value=MOCK_RESULT):
        resp = client.post(
            "/evaluate",
            files={"resume": ("cv.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"jd_text": "AI engineer with Python and LLM experience"},
        )
    assert resp.json()["skill_score"] == 70.0
