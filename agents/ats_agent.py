def ats_agent(state):
    """
    ATS Agent
    - Resume readability & structure validation
    - MUST run on FULL resume (not filtered evidence)
    """

    issues = []

    # 🔥 IMPORTANT FIX: use FULL resume
    text = " ".join(state["full_resume_chunks"]).lower()
    word_count = len(text.split())

    # ---------- HARD ATS CHECK ----------
    if word_count < 80:
        issues.append("Resume content too short for reliable ATS parsing")

    SECTION_ALIASES = {
        "experience": ["experience", "work", "employment", "professional"],
        "project": ["project", "projects", "research", "work"],
        "education": [
            "education", "academics", "qualification", "degree",
            "bachelor", "master", "masters", "mca",
            "b.tech", "btech", "m.tech", "mtech",
            "phd", "doctorate",
            "iit ", "nit ", "iiit ",
            "university", "college", "institute",
            "from ", "graduated", "cgpa", "gpa"
        ]
    }

    missing = []
    for section, aliases in SECTION_ALIASES.items():
        if not any(alias in text for alias in aliases):
            missing.append(section)

    if missing:
        issues.append(f"Missing core sections: {', '.join(missing)}")

    # ---------- SOFT WARNING ----------
    if 80 <= word_count < 120:
        issues.append("Resume may be overly compressed or design-heavy")

    state["ats_issues"] = issues
    return state
