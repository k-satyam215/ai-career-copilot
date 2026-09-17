import tempfile
import traceback

import streamlit as st

from career_pipeline import evaluate_resume

st.set_page_config(page_title="AI Career Copilot", layout="centered")

st.title("🧑‍💼 AI Career Copilot")
st.write("Upload your resume and select the target role.")

ROLE_JD_MAP = {
    "AI / GenAI Engineer": """
Looking for an AI / GenAI engineer with experience in LLMs, RAG,
LangChain, prompt engineering, vector databases, agents,
and generative AI systems.
""",
    "Backend Engineer": """
Backend engineer with experience in Python, REST APIs, databases,
system design, scalability, performance optimization,
and backend frameworks.
""",
    "Data Analyst": """
Data analyst with strong skills in SQL, Pandas, NumPy,
data visualization, dashboards, reporting,
and data-driven decision making.
""",
    "Machine Learning Engineer": """
Machine learning engineer with experience in model training,
feature engineering, evaluation, metrics,
scikit-learn, and ML pipelines.
""",
    "Frontend Engineer": """
Frontend engineer with experience in React, JavaScript,
HTML, CSS, UI/UX design, dashboards,
and frontend frameworks.
""",
    "DevOps Engineer": """
DevOps engineer with experience in Docker, CI/CD pipelines,
cloud platforms (AWS/GCP/Azure),
deployment, monitoring, and automation.
""",
    "Auto Detect (from JD)": "",
}

selected_role = st.selectbox("Select Target Role", list(ROLE_JD_MAP.keys()))

if selected_role == "Auto Detect (from JD)":
    jd_text = st.text_area(
        "Paste Job Description",
        height=180,
        placeholder="Paste the job description here...",
    )
else:
    jd_text = ROLE_JD_MAP[selected_role]

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if st.button("Evaluate Resume"):
    if not uploaded_file:
        st.error("Please upload a resume PDF.")
        st.stop()

    if not jd_text.strip():
        st.error("Please provide a job description.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        resume_path = f.name

    try:
        progress = st.progress(0, text="📄 Loading resume...")
        progress.progress(10, text="🔍 Detecting role...")

        from retrieval.chunking import chunk_resume
        from retrieval.loader import load_resume
        resume_text = load_resume(resume_path)
        all_chunks = chunk_resume(resume_text)
        progress.progress(25, text="🧠 Loading embedding model (first run may take ~30s)...")

        # warm up model now so progress bar shows
        from retrieval.embeddings import get_model
        get_model()
        progress.progress(50, text="🔎 Retrieving evidence chunks...")

        progress.progress(65, text="🤖 Running agent pipeline...")
        result = evaluate_resume(resume_path, jd_text)
        progress.progress(100, text="✅ Done!")
        progress.empty()

        if result.get("llm_errors"):
            with st.expander("⚠️ Some AI-generated sections used fallback content — click for details", expanded=True):
                st.error(
                    "The Groq LLM call failed, so improvement suggestions / "
                    "interview Q&A below are generic fallback text, not "
                    "personalized to this resume. Check GROQ_API_KEY in your "
                    "Streamlit secrets."
                )
                for err in result["llm_errors"]:
                    st.code(err)

        st.subheader("🧠 Evaluation Result")
        st.write("**Detected Role Profile:**", result["role"])

        c1, c2 = st.columns(2)
        c1.metric("Skill Score", result["skill_score"])
        c2.metric("Experience Score", result["experience_score"])

        breakdown = result.get("skill_breakdown", {})
        if breakdown.get("matched") or breakdown.get("missing"):
            with st.expander("📊 Skill area breakdown", expanded=False):
                if breakdown.get("matched"):
                    st.markdown(
                        "**Covered:** "
                        + " ".join(f"`{g}`" for g in breakdown["matched"])
                    )
                if breakdown.get("missing"):
                    st.markdown(
                        "**Not detected:** "
                        + " ".join(f"`{g}`" for g in breakdown["missing"])
                    )

        st.subheader("📄 ATS Check")
        if result["ats_issues"]:
            for issue in result["ats_issues"]:
                st.warning(issue)
        else:
            st.success("No ATS issues found")

        st.subheader("✍️ Resume Improvement Suggestions")
        for suggestion in result["improvement_suggestions"]:
            st.info(suggestion)

        st.subheader("🎯 Interview Questions & Answers")
        for i, (question, answer) in enumerate(
            zip(result["interview_questions"], result["interview_answers"]), 1
        ):
            st.markdown(f"**Q{i}. {question}**")
            st.markdown(f"🗣️ {answer}")

        st.subheader("✅ Final Verdict")
        st.success(result["verdict"])
        if result.get("verdict_reason"):
            st.caption(result["verdict_reason"])

        report_lines = [
            "AI Career Copilot — Evaluation Report",
            f"Role: {result['role']}",
            f"Skill Score: {result['skill_score']}",
            f"Experience Score: {result['experience_score']}",
            f"Verdict: {result['verdict']} — {result.get('verdict_reason', '')}",
            "",
            "ATS Issues:",
            *(result["ats_issues"] or ["None"]),
            "",
            "Improvement Suggestions:",
            *[f"- {s}" for s in result["improvement_suggestions"]],
            "",
            "Interview Questions & Answers:",
            *[
                f"Q{i}. {q}\nA{i}. {a}\n"
                for i, (q, a) in enumerate(
                    zip(result["interview_questions"], result["interview_answers"]), 1
                )
            ],
        ]
        st.download_button(
            "📥 Download Full Report (.txt)",
            data="\n".join(report_lines),
            file_name="resume_evaluation_report.txt",
            mime="text/plain",
        )

    except Exception as e:  # noqa: BLE001
        st.error(f"❌ Error: {e}")
        with st.expander("Show full traceback"):
            st.code(traceback.format_exc())
