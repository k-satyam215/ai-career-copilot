from agents import (
    ats_agent,
    experience_agent,
    improvement_agent,
    judge_agent,
    llm_answer_agent,
    llm_interview_agent,
    skill_agent,
)
from retrieval.chunking import chunk_resume
from retrieval.loader import load_resume
from retrieval.retriever import retrieve_evidence
from role_detector import detect_role_from_jd, detect_role_from_resume
from roles_config import EVALUATION_THRESHOLDS, SKILL_DOMAINS

AGENT_PIPELINE = [
    skill_agent,
    experience_agent,
    ats_agent,
    improvement_agent,
    llm_interview_agent,
    llm_answer_agent,
    judge_agent,
]


def format_role_label(jd_role: str, resume_primary: str) -> str:
    """Produce a user-facing role label from JD intent and resume strengths."""
    jd_role = jd_role.lower()
    resume_primary = resume_primary.lower()

    if "genai" in jd_role and resume_primary == "backend":
        return "Backend Engineer (GenAI-Aligned)"
    if "genai" in jd_role and resume_primary == "ml":
        return "Machine Learning Engineer (GenAI-Aligned)"
    if "genai" in jd_role:
        return "GenAI Engineer"

    return resume_primary.replace("_", " ").title() + " Engineer"


def _extract_retrieval_keywords(role: str):
    domains = [d.strip() for d in role.split("+")]
    keywords = []
    for domain in domains:
        if domain in SKILL_DOMAINS:
            keywords.extend(SKILL_DOMAINS[domain])
    return keywords or None


def run_agent_pipeline(state: dict) -> dict:
    """Run all agents in sequence over the shared state."""
    for agent in AGENT_PIPELINE:
        state = agent(state)
    return state


def evaluate_resume(resume_path: str, jd_text: str) -> dict:
    """End-to-end resume evaluation against a job description.

    Loads and chunks the resume, detects role alignment, retrieves
    relevant evidence, then runs the full agent pipeline to produce
    scores, ATS issues, improvement suggestions, interview Q&A, and
    a final verdict.
    """
    resume_text = load_resume(resume_path)
    all_chunks = chunk_resume(resume_text)

    jd_role = detect_role_from_jd(jd_text)
    resume_scores = detect_role_from_resume(all_chunks)
    resume_primary = max(resume_scores, key=resume_scores.get)

    retrieval_keywords = _extract_retrieval_keywords(jd_role)
    evidence_chunks = retrieve_evidence(all_chunks, jd_text, retrieval_keywords)

    state = {
        "resume_chunks": all_chunks,
        "full_resume_chunks": all_chunks,
        "evidence_chunks": evidence_chunks,
        "jd_text": jd_text,
        "role": jd_role,
    }

    state = run_agent_pipeline(state)

    skill_ok = state.get("skill_score", 0) >= EVALUATION_THRESHOLDS["skill_score"]
    experience_ok = state.get("experience_score", 0) >= EVALUATION_THRESHOLDS["experience_score"]
    verdict = "Interview Ready (Fresher)" if skill_ok and experience_ok else "Needs Improvement"

    return {
        "role": format_role_label(jd_role, resume_primary),
        "skill_score": state.get("skill_score", 0),
        "experience_score": state.get("experience_score", 0),
        "ats_issues": state.get("ats_issues", []),
        "improvement_suggestions": state.get("improvement_suggestions", []),
        "interview_questions": state.get("interview_questions", []),
        "interview_answers": state.get("interview_answers", []),
        "verdict": state.get("verdict", verdict),
    }
