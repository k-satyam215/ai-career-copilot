def generate_hypothetical_profile(jd_text: str) -> str:
    """Generate a hypothetical ideal-candidate profile for HYDE-style retrieval."""
    return (
        "Ideal candidate closely matches the following job description:\n"
        f"{jd_text}\n\n"
        "Demonstrates hands-on experience, ownership, "
        "system-level understanding, and measurable impact."
    )
