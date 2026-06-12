from roles_config import SKILL_DOMAINS


def _score_domains(text: str) -> dict:
    return {
        domain: sum(1 for keyword in keywords if keyword in text)
        for domain, keywords in SKILL_DOMAINS.items()
    }


def detect_role_from_jd(jd_text: str) -> str:
    """Detect primary (and optional secondary) role intent from a job description."""
    scores = _score_domains(jd_text.lower())
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    primary = ranked[0][0]
    secondary = ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0 else None

    return f"{primary} + {secondary}" if secondary else primary


def detect_role_from_resume(resume_chunks: list) -> dict:
    """Return a domain -> keyword-match-count mapping for resume strength analysis."""
    text = " ".join(resume_chunks).lower()
    return _score_domains(text)
