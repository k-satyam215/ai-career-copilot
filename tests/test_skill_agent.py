"""Tests for skill_agent."""
from agents.skill_agent import MAX_SKILL_SCORE, POINTS_PER_GROUP, SKILL_GROUPS, skill_agent


def make_state(text):
    return {"resume_chunks": [text], "full_resume_chunks": [text]}


# ---- Basic scoring --------------------------------------------------------

def test_python_keyword_scores():
    state = make_state("python developer with 2 years experience")
    result = skill_agent(state)
    assert result["skill_score"] >= POINTS_PER_GROUP


def test_genai_keywords_score():
    state = make_state("built an LLM application with RAG and LangChain")
    result = skill_agent(state)
    assert result["skill_score"] >= POINTS_PER_GROUP


def test_vector_keywords_score():
    state = make_state("used FAISS embeddings for vector search")
    result = skill_agent(state)
    assert result["skill_score"] >= POINTS_PER_GROUP


def test_backend_keywords_score():
    state = make_state("built a FastAPI backend REST API service")
    result = skill_agent(state)
    assert result["skill_score"] >= POINTS_PER_GROUP


def test_projects_keywords_score():
    state = make_state("built and developed several projects")
    result = skill_agent(state)
    assert result["skill_score"] >= POINTS_PER_GROUP


def test_empty_resume_zero_score():
    state = make_state("")
    result = skill_agent(state)
    assert result["skill_score"] == 0


def test_irrelevant_text_zero_score():
    state = make_state("I like cooking and reading novels")
    result = skill_agent(state)
    assert result["skill_score"] == 0


def test_score_capped_at_max():
    text = " ".join([
        kw for group in SKILL_GROUPS.values() for kw in group
    ])
    state = make_state(text)
    result = skill_agent(state)
    assert result["skill_score"] <= MAX_SKILL_SCORE


def test_multiple_groups_accumulate():
    state = make_state("python llm rag fastapi project built embedding")
    result = skill_agent(state)
    assert result["skill_score"] > POINTS_PER_GROUP


def test_case_insensitive_matching():
    state = make_state("PYTHON LLM FASTAPI EMBEDDING PROJECT")
    result = skill_agent(state)
    assert result["skill_score"] > 0


def test_state_mutation_returns_state():
    state = make_state("python llm")
    result = skill_agent(state)
    assert "skill_score" in result
    assert result is state


def test_multiple_chunks():
    state = {
        "resume_chunks": ["python developer", "langchain rag pipeline", "fastapi project"],
        "full_resume_chunks": [],
    }
    result = skill_agent(state)
    assert result["skill_score"] >= POINTS_PER_GROUP * 3
