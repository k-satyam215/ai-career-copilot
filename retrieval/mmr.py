import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(v):
    return v / np.linalg.norm(v)

def mmr(query, docs, top_k=8, lambda_param=0.7):
    if not docs:
        return []

    q_emb = normalize(model.encode([query])[0])
    d_embs = np.array([normalize(e) for e in model.encode(docs)])

    selected, selected_embs = [], []

    for _ in range(min(top_k, len(docs))):
        scores = []

        for i, emb in enumerate(d_embs):
            if i in selected:
                continue

            sim_q = np.dot(emb, q_emb)
            sim_sel = max(
                [np.dot(emb, s) for s in selected_embs],
                default=0
            )

            score = lambda_param * sim_q - (1 - lambda_param) * sim_sel
            scores.append((score, i))

        _, best = max(scores)
        selected.append(best)
        selected_embs.append(d_embs[best])

    return [docs[i] for i in selected]
