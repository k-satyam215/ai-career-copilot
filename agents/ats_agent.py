MIN_WORD_COUNT = 80
SOFT_WARNING_UPPER_BOUND = 120

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
        "from ", "graduated", "cgpa", "gpa",
    ],
}


def ats_agent(state):
    """ATS readability and structure validation.

    MUST run on the FULL resume text (not the filtered evidence chunks),
    since structural checks (sections, length) need complete context.
    """
    issues = []

    text = " ".join(state["full_resume_chunks"]).lower()
    word_count = len(text.split())

    if word_count < MIN_WORD_COUNT:
        issues.append("Resume content too short for reliable ATS parsing")

    missing_sections = [
        section
        for section, aliases in SECTION_ALIASES.items()
        if not any(alias in text for alias in aliases)
    ]
    if missing_sections:
        issues.append(f"Missing core sections: {', '.join(missing_sections)}")

    if MIN_WORD_COUNT <= word_count < SOFT_WARNING_UPPER_BOUND:
        issues.append("Resume may be overly compressed or design-heavy")

    state["ats_issues"] = issues
    return state
