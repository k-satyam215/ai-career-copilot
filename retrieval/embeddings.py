"""Sentence-embedding utilities.

The SentenceTransformer model is loaded lazily so that modules importing
this file (and test collection) don't pay the model-download cost unless
embeddings are actually used.
"""

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts):
    return get_model().encode(texts)


def semantic_search(query, chunks, index=None, k=5):
    """Return the top-k chunks most semantically similar to the query."""
    import numpy as np

    if not chunks:
        return []

    model = get_model()
    query_emb = model.encode([query])[0]
    chunk_embs = model.encode(chunks)

    scores = chunk_embs @ query_emb
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_indices]
