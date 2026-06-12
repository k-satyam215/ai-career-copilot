from typing import TypedDict, List


class AgentState(TypedDict, total=False):
    resume_chunks: List[str]
    full_resume_chunks: List[str]
    evidence_chunks: List[str]
    jd_text: str
    role: str
    skill_score: float
    experience_score: float
    ats_issues: List[str]
    improvement_suggestions: List[str]
    interview_questions: List[str]
    interview_answers: List[str]
    verdict: str
