"""MCP (Model Context Protocol) tool layer for AI Career Copilot.

Exposes resume parsing, evidence retrieval, and full resume evaluation
as MCP tools so external agents/LLM hosts can drive the pipeline.
"""

from mcp.server.fastmcp import FastMCP

from career_pipeline import evaluate_resume, _extract_retrieval_keywords
from retrieval.chunking import chunk_resume
from retrieval.loader import load_resume
from retrieval.retriever import retrieve_evidence
from role_detector import detect_role_from_jd

mcp = FastMCP("ai-career-copilot")


@mcp.tool()
def parse_resume(resume_path: str) -> dict:
    """Load a resume PDF and split it into evidence chunks.

    Args:
        resume_path: Absolute path to a resume PDF file.

    Returns:
        A dict with the raw text length and the list of resume chunks.
    """
    text = load_resume(resume_path)
    chunks = chunk_resume(text)
    return {"chunk_count": len(chunks), "chunks": chunks}


@mcp.tool()
def retrieve_resume_evidence(resume_path: str, jd_text: str) -> dict:
    """Retrieve diversity-ranked evidence chunks relevant to a job description.

    Args:
        resume_path: Absolute path to a resume PDF file.
        jd_text: The target job description text.

    Returns:
        A dict containing the detected JD role and the retrieved evidence chunks.
    """
    text = load_resume(resume_path)
    chunks = chunk_resume(text)

    jd_role = detect_role_from_jd(jd_text)
    keywords = _extract_retrieval_keywords(jd_role)
    evidence = retrieve_evidence(chunks, jd_text, keywords)

    return {"role": jd_role, "evidence_chunks": evidence}


@mcp.tool()
def evaluate_resume_against_jd(resume_path: str, jd_text: str) -> dict:
    """Run the full agentic evaluation pipeline on a resume against a job description.

    Args:
        resume_path: Absolute path to a resume PDF file.
        jd_text: The target job description text.

    Returns:
        A dict with role label, skill/experience scores, ATS issues,
        improvement suggestions, interview Q&A, and a final verdict.
    """
    return evaluate_resume(resume_path, jd_text)


if __name__ == "__main__":
    mcp.run()
