"""Tests for retrieval.mmr and retrieval.retriever (model mocked)."""
from unittest.mock import patch

import numpy as np

# ---- MMR tests -------------------------------------------------------------

def _make_mock_model(n_dims=8):
    """Return a fake SentenceTransformer model."""
    class FakeModel:
        def encode(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            np.random.seed(42)
            vecs = np.random.rand(len(texts), n_dims).astype("float32")
            return vecs
    return FakeModel()


def test_mmr_returns_top_k():
    from retrieval.mmr import mmr
    docs = [f"document about python llm rag {i}" for i in range(10)]
    with patch("retrieval.mmr.get_model", return_value=_make_mock_model()):
        result = mmr("query about python llm", docs, top_k=5)
    assert len(result) == 5


def test_mmr_empty_docs():
    from retrieval.mmr import mmr
    with patch("retrieval.mmr.get_model", return_value=_make_mock_model()):
        result = mmr("query", [], top_k=5)
    assert result == []


def test_mmr_top_k_larger_than_docs():
    from retrieval.mmr import mmr
    docs = ["doc one llm", "doc two rag"]
    with patch("retrieval.mmr.get_model", return_value=_make_mock_model()):
        result = mmr("query", docs, top_k=10)
    assert len(result) == 2


def test_mmr_all_docs_from_original():
    from retrieval.mmr import mmr
    docs = [f"chunk {i}" for i in range(6)]
    with patch("retrieval.mmr.get_model", return_value=_make_mock_model()):
        result = mmr("query", docs, top_k=6)
    assert all(r in docs for r in result)


def test_mmr_no_duplicates():
    from retrieval.mmr import mmr
    docs = [f"text about rag llm langchain {i}" for i in range(8)]
    with patch("retrieval.mmr.get_model", return_value=_make_mock_model()):
        result = mmr("query", docs, top_k=8)
    assert len(result) == len(set(result))


def test_normalize_zero_vector():
    from retrieval.mmr import normalize
    v = np.zeros(4)
    result = normalize(v)
    assert (result == 0).all()


def test_normalize_unit_vector():
    from retrieval.mmr import normalize
    v = np.array([3.0, 4.0])
    result = normalize(v)
    np.testing.assert_almost_equal(np.linalg.norm(result), 1.0)


# ---- retriever tests -------------------------------------------------------

def test_retriever_returns_list(sample_chunks, sample_jd):
    from retrieval.retriever import retrieve_evidence
    keywords = ["llm", "rag", "langchain"]
    with patch("retrieval.retriever.mmr") as mock_mmr:
        mock_mmr.return_value = sample_chunks[:4]
        result = retrieve_evidence(sample_chunks, sample_jd, keywords)
    assert isinstance(result, list)


def test_retriever_falls_back_when_few_candidates(sample_chunks, sample_jd):
    from retrieval.retriever import retrieve_evidence
    # keywords that match nothing -> fallback to all chunks
    keywords = ["xyznonexistent"]
    with patch("retrieval.retriever.mmr") as mock_mmr:
        mock_mmr.return_value = sample_chunks
        retrieve_evidence(sample_chunks, sample_jd, keywords)
        # mmr should have been called with all chunks
        call_args = mock_mmr.call_args
        assert len(call_args[0][1]) == len(sample_chunks)


def test_retriever_no_keywords_uses_all_chunks(sample_chunks, sample_jd):
    from retrieval.retriever import retrieve_evidence
    with patch("retrieval.retriever.mmr") as mock_mmr:
        mock_mmr.return_value = sample_chunks[:3]
        retrieve_evidence(sample_chunks, sample_jd, None)
        call_args = mock_mmr.call_args
        assert len(call_args[0][1]) == len(sample_chunks)
