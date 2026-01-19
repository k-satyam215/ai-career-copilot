from typing import TypedDict, List

class AgentState(TypedDict):
    resume_chunks: List[str]
    jd_text: str
    role: str
    skill_score: float
    experience_score: float
    ats_issues: List[str]
    verdict: str
