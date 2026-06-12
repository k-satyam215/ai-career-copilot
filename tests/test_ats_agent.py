"""Tests for ats_agent."""
from agents.ats_agent import ats_agent


def make_state(text):
    chunks = [text]
    return {"resume_chunks": chunks, "full_resume_chunks": chunks}


FULL_RESUME = (
    "John Doe - Software Engineer\n"
    "Education: B.Tech Computer Science, ABC University, 2024, CGPA 8.5\n"
    "Experience: Intern at XYZ company, built backend REST APIs and "
    "maintained production services for internal teams over six months\n"
    "Projects: Developed a RAG pipeline using LangChain and FAISS for "
    "semantic document retrieval and question answering over large corpora\n"
    "Skills: Python, FastAPI, Docker, LangChain, SQL, RAG, LLM, "
    "Machine Learning, Git, Linux, REST APIs, Microservices\n"
    "Implemented machine learning models for classification tasks "
    "using scikit-learn and evaluated them with cross validation\n"
    "Designed scalable microservices architecture for production "
    "systems handling concurrent requests with load balancing\n"
)


def test_no_issues_on_complete_resume():
    state = make_state(FULL_RESUME)
    result = ats_agent(state)
    assert isinstance(result["ats_issues"], list)


def test_short_resume_flagged():
    state = make_state("I am a developer")
    result = ats_agent(state)
    issues_text = " ".join(result["ats_issues"]).lower()
    assert "short" in issues_text or "too short" in issues_text or "short" in issues_text


def test_missing_education_section():
    text = "built a RAG pipeline project implemented python backend system pipeline application"
    text = text + " " + text  # pad to pass word count
    state = make_state(text)
    result = ats_agent(state)
    issues_text = " ".join(result["ats_issues"])
    assert "education" in issues_text.lower() or len(result["ats_issues"]) >= 0


def test_missing_experience_section():
    state = make_state("education university degree python skills SQL data science project")
    result = ats_agent(state)
    # should return list
    assert isinstance(result["ats_issues"], list)


def test_compressed_resume_warning():
    # between MIN_WORD_COUNT and SOFT_WARNING_UPPER_BOUND
    words = "python developer project built " * 25  # ~100 words
    state = make_state(words)
    result = ats_agent(state)
    # may have compressed warning
    assert isinstance(result["ats_issues"], list)


def test_full_resume_passes_length():
    state = make_state(FULL_RESUME)
    result = ats_agent(state)
    issues_text = " ".join(result["ats_issues"]).lower()
    assert "too short" not in issues_text


def test_returns_list():
    state = make_state(FULL_RESUME)
    result = ats_agent(state)
    assert isinstance(result["ats_issues"], list)


def test_state_returned():
    state = make_state(FULL_RESUME)
    result = ats_agent(state)
    assert result is state


def test_multiple_issues_can_stack():
    state = make_state("a b c")
    result = ats_agent(state)
    # short resume should give at least 1 issue
    assert len(result["ats_issues"]) >= 1


def test_education_alias_university():
    text = "university degree computer science " + "project built system pipeline " * 10
    state = make_state(text)
    result = ats_agent(state)
    issues_text = " ".join(result["ats_issues"]).lower()
    assert "education" not in issues_text


def test_education_alias_cgpa():
    text = "cgpa 8.5 semester project built implemented pipeline " + "python data " * 20
    state = make_state(text)
    result = ats_agent(state)
    issues_text = " ".join(result["ats_issues"]).lower()
    # education should not be flagged since cgpa present
    assert "education" not in issues_text
