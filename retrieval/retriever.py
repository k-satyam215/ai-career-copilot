from retrieval.hyde import generate_hypothetical_profile
from retrieval.mmr import mmr

def retrieve_evidence(resume_chunks, jd_text, role_keywords):
    hypothetical = generate_hypothetical_profile(jd_text)

    # role keyword filter
    candidates = [
        c for c in resume_chunks
        if any(k in c.lower() for k in role_keywords)
    ]

    # 🔥 CRITICAL FIX
    # If filtering removed too much signal, fallback
    if len(candidates) < 8:
        candidates = resume_chunks

    # diversity + relevance
    return mmr(hypothetical, candidates, top_k=10)
