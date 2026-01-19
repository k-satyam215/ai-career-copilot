# role_detector.py

from roles_config import SKILL_DOMAINS


def detect_role_from_jd(jd_text: str) -> str:
    """
    JD-driven role intent detection
    """
    jd = jd_text.lower()
    scores = {}

    for domain, keywords in SKILL_DOMAINS.items():
        scores[domain] = sum(1 for k in keywords if k in jd)

    # sort by relevance
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    primary = ranked[0][0]
    secondary = ranked[1][0] if ranked[1][1] > 0 else None

    if secondary:
        return f"{primary} + {secondary}"
    return primary


def detect_role_from_resume(resume_chunks: list) -> dict:
    """
    Resume-driven strength detection
    Returns domain → score mapping
    """
    text = " ".join(resume_chunks).lower()
    scores = {}

    for domain, keywords in SKILL_DOMAINS.items():
        scores[domain] = sum(1 for k in keywords if k in text)

    return scores
