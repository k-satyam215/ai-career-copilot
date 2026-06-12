import numpy as np

from retrieval.embeddings import get_model

DEFAULT_TOP_K = 8
DEFAULT_LAMBDA = 0.7


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def mmr(query: str, docs: list, top_k: int = DEFAULT_TOP_K, lambda_param: float = DEFAULT_LAMBDA) -> list:
    """Maximal Marginal Relevance selection for diversity-aware evidence retrieval."""
    if not docs:
        return []

    model = get_model()
    query_emb = normalize(model.encode([query])[0])
    doc_embs = np.array([normalize(e) for e in model.encode(docs)])

    selected_indices: list = []
    selected_embs: list = []

    for _ in range(min(top_k, len(docs))):
        best_score = None
        best_index = None

        for i, emb in enumerate(doc_embs):
            if i in selected_indices:
                continue

            relevance = float(np.dot(emb, query_emb))
            redundancy = max((float(np.dot(emb, s)) for s in selected_embs), default=0.0)

            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if best_score is None or score > best_score:
                best_score = score
                best_index = i

        selected_indices.append(best_index)
        selected_embs.append(doc_embs[best_index])

    return [docs[i] for i in selected_indices]
