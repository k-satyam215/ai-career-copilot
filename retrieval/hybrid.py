from retrieval.embeddings import semantic_search

SPARSE_TOP_K = 5


def enhance_query(user_query: str, jd_text: str) -> list:
    """Generate query variants for richer retrieval coverage."""
    return [
        f"{user_query} matching required skills",
        f"experience relevance for role: {jd_text[:200]}",
        "ATS optimized resume bullets",
    ]


def keyword_search(query: str, chunks: list, k: int = SPARSE_TOP_K) -> list:
    """Simple sparse retrieval based on keyword overlap."""
    query_terms = set(query.lower().split())

    scored = [
        (len(query_terms & set(chunk.lower().split())), chunk)
        for chunk in chunks
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [chunk for _, chunk in scored[:k]]


def hybrid_search(query: str, chunks: list, index=None) -> list:
    """Combine dense (semantic) and sparse (keyword) retrieval, deduplicated."""
    dense = semantic_search(query, chunks, index)
    sparse = keyword_search(query, chunks)
    return list(dict.fromkeys(dense + sparse))
