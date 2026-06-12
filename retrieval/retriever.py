from retrieval.hyde import generate_hypothetical_profile
from retrieval.mmr import mmr

MIN_CANDIDATES = 8
EVIDENCE_TOP_K = 10


def retrieve_evidence(resume_chunks: list, jd_text: str, role_keywords) -> list:
    """Retrieve diverse, role-relevant evidence chunks from the resume.

    Falls back to the full chunk set if role-keyword filtering removes
    too much signal.
    """
    hypothetical = generate_hypothetical_profile(jd_text)

    if role_keywords:
        candidates = [
            chunk for chunk in resume_chunks
            if any(keyword in chunk.lower() for keyword in role_keywords)
        ]
    else:
        candidates = []

    if len(candidates) < MIN_CANDIDATES:
        candidates = resume_chunks

    return mmr(hypothetical, candidates, top_k=EVIDENCE_TOP_K)
