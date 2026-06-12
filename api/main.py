import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from career_pipeline import evaluate_resume

app = FastAPI(
    title="AI Career Copilot API",
    description="Agentic resume evaluation, ATS validation, and interview prep.",
    version="1.0.0",
)


class EvaluationResponse(BaseModel):
    role: str
    skill_score: float
    experience_score: float
    ats_issues: list[str]
    improvement_suggestions: list[str]
    interview_questions: list[str]
    interview_answers: list[str]
    verdict: str


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    resume: UploadFile = File(..., description="Resume file (PDF)"),
    jd_text: str = Form(..., description="Target job description"),
) -> dict:
    """Evaluate an uploaded resume PDF against a job description."""
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported")

    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description must not be empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(resume.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        return evaluate_resume(str(tmp_path), jd_text)
    finally:
        tmp_path.unlink(missing_ok=True)
