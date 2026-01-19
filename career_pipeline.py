# career_pipeline.py

from retrieval.loader import load_resume
from retrieval.chunking import chunk_resume
from retrieval.retriever import retrieve_evidence

from role_detector import (
    detect_role_from_jd,
    detect_role_from_resume
)

from roles_config import SKILL_DOMAINS, EVALUATION_THRESHOLDS

from agents.skill_agent import skill_agent
from agents.experience_agent import experience_agent
from agents.ats_agent import ats_agent
from agents.improvement_agent import improvement_agent
from agents.llm_interview_agent import llm_interview_agent
from agents.llm_answer_agent import llm_answer_agent
from agents.judge_agent import judge_agent


# --------------------------------------------------
# USER-FACING ROLE LABEL
# --------------------------------------------------
def format_role_label(jd_role: str, resume_primary: str) -> str:
    jd_role = jd_role.lower()
    resume_primary = resume_primary.lower()

    if "genai" in jd_role and resume_primary == "backend":
        return "Backend Engineer (GenAI-Aligned)"

    if "genai" in jd_role and resume_primary == "ml":
        return "Machine Learning Engineer (GenAI-Aligned)"

    if "genai" in jd_role:
        return "GenAI Engineer"

    return resume_primary.replace("_", " ").title() + " Engineer"


# --------------------------------------------------
# RETRIEVAL KEYWORDS
# --------------------------------------------------
def _extract_retrieval_keywords(role: str):
    domains = [r.strip() for r in role.split("+")]
    keywords = []

    for d in domains:
        if d in SKILL_DOMAINS:
            keywords.extend(SKILL_DOMAINS[d])

    return keywords if keywords else None


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def evaluate_resume(resume_path: str, jd_text: str) -> dict:

    # 1️⃣ Load & chunk resume
    resume_text = load_resume(resume_path)
    all_chunks = chunk_resume(resume_text)

    # 2️⃣ Detect roles
    jd_role = detect_role_from_jd(jd_text)
    resume_scores = detect_role_from_resume(all_chunks)

    resume_primary = max(resume_scores, key=resume_scores.get)

    # 3️⃣ Retrieval
    retrieval_keywords = _extract_retrieval_keywords(jd_role)
    evidence_chunks = retrieve_evidence(
        all_chunks,
        jd_text,
        retrieval_keywords
    )

    # 4️⃣ Shared agent state
    state = {
        "resume_chunks": all_chunks,
        "full_resume_chunks": all_chunks,
        "evidence_chunks": evidence_chunks,
        "jd_text": jd_text,
        "role": jd_role,
    }

    # 5️⃣ Agent pipeline
    for agent in [
        skill_agent,
        experience_agent,
        ats_agent,
        improvement_agent,
        llm_interview_agent,
        llm_answer_agent,
        judge_agent,
    ]:
        state = agent(state)

    # 6️⃣ Verdict
    skill_ok = state.get("skill_score", 0) >= EVALUATION_THRESHOLDS["skill_score"]
    exp_ok = state.get("experience_score", 0) >= EVALUATION_THRESHOLDS["experience_score"]

    verdict = (
        "Interview Ready (Fresher)"
        if skill_ok and exp_ok
        else "Needs Improvement"
    )

    # 7️⃣ FINAL OUTPUT
    return {
        "role": format_role_label(jd_role, resume_primary),
        "skill_score": state.get("skill_score", 0),
        "experience_score": state.get("experience_score", 0),
        "ats_issues": state.get("ats_issues", []),
        "improvement_suggestions": state.get("improvement_suggestions", []),
        "interview_questions": state.get("interview_questions", []),
        "interview_answers": state.get("interview_answers", []),
        "verdict": verdict,
    }
