"""Sentence-embedding utilities with Streamlit cache support."""

_model = None


def get_model():
    global _model
    if _model is None:
        try:
            import streamlit as st

            @st.cache_resource(show_spinner=False)
            def _load():
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer("all-MiniLM-L6-v2")

            _model = _load()
        except Exception:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts):
    return get_model().encode(texts)


def semantic_search(query, chunks, index=None, k=5):
    import numpy as np

    if not chunks:
        return []

    model = get_model()
    query_emb = model.encode([query])[0]
    chunk_embs = model.encode(chunks)

    scores = chunk_embs @ query_emb
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_indices]
