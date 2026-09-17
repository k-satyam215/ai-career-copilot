from typing import TypedDict


class AgentState(TypedDict, total=False):
    resume_chunks: list[str]
    full_resume_chunks: list[str]
    evidence_chunks: list[str]
    jd_text: str
    role: str
    skill_score: float
    experience_score: float
    ats_issues: list[str]
    improvement_suggestions: list[str]
    interview_questions: list[str]
    interview_answers: list[str]
    verdict: str
    verdict_reason: str
    skill_breakdown: dict
    llm_errors: list[str]
