from retrieval.embeddings import semantic_search

def enhance_query(user_query, jd_text):
    return [
        f"{user_query} matching required skills",
        f"experience relevance for role: {jd_text[:200]}",
        "ATS optimized resume bullets"
    ]

def keyword_search(query, chunks, k=5):
    q = set(query.lower().split())
    scored = []
    for c in chunks:
        score = len(q & set(c.lower().split()))
        scored.append((score, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:k]]

def hybrid_search(query, chunks, index):
    dense = semantic_search(query, chunks, index)
    sparse = keyword_search(query, chunks)
    return list(dict.fromkeys(dense + sparse))
