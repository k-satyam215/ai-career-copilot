"""Shared fixtures and mocks for the test suite."""
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub heavy optional dependencies BEFORE any app module imports them
# ---------------------------------------------------------------------------

def _make_sentence_transformers_stub():
    """Return a minimal sentence_transformers stub."""
    st = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return np.random.rand(len(texts), 384).astype("float32")

    st.SentenceTransformer = FakeSentenceTransformer
    return st


if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = _make_sentence_transformers_stub()

if "faiss" not in sys.modules:
    faiss_stub = types.ModuleType("faiss")
    sys.modules["faiss"] = faiss_stub

# ---------------------------------------------------------------------------
# Common state fixtures
# ---------------------------------------------------------------------------

SAMPLE_JD = (
    "Looking for an AI/GenAI engineer with experience in LLMs, RAG, "
    "LangChain, prompt engineering, vector databases, agents, and "
    "generative AI systems. FastAPI and Docker experience is a plus."
)

SAMPLE_CHUNKS = [
    "Developed a RAG pipeline using LangChain and FAISS for document retrieval",
    "Built a FastAPI backend for serving ML model predictions",
    "Implemented prompt engineering patterns for LLM applications",
    "Designed a vector database system with embedding-based search",
    "Created Python scripts for data processing and ETL pipelines",
    "Worked on machine learning model training and evaluation metrics",
    "Developed an agent-based system using LangChain ReAct pattern",
    "Built scalable backend APIs with authentication and rate limiting",
]


@pytest.fixture
def sample_jd():
    return SAMPLE_JD


@pytest.fixture
def sample_chunks():
    return SAMPLE_CHUNKS


@pytest.fixture
def base_state(sample_chunks, sample_jd):
    return {
        "resume_chunks": sample_chunks,
        "full_resume_chunks": sample_chunks,
        "evidence_chunks": sample_chunks[:4],
        "jd_text": sample_jd,
        "role": "genai",
    }


@pytest.fixture
def state_with_scores(base_state):
    state = base_state.copy()
    state.update({"skill_score": 70.0, "experience_score": 64.0})
    return state


@pytest.fixture
def state_with_questions(state_with_scores):
    state = state_with_scores.copy()
    state["interview_questions"] = [
        f"How did you implement the RAG pipeline component {i}?"
        for i in range(15)
    ]
    return state


@pytest.fixture
def mock_llm(mocker):
    """Return a mock LLM that returns a configurable string."""
    from unittest.mock import MagicMock
    fake = MagicMock()
    fake.content = "Built a retrieval system using FAISS and LangChain for semantic search."
    mock_resp = MagicMock()
    mock_resp.content = fake.content
    fake.invoke = MagicMock(return_value=mock_resp)
    return fake
