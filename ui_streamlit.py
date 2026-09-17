import tempfile
import traceback

import streamlit as st

from career_pipeline import evaluate_resume

st.set_page_config(
    page_title="AI Career Copilot",
    page_icon="🧑‍💼",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Global styling — fonts, gradient hero, pill badges, score cards, verdict
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* ---- Hero ---- */
    .hero-wrap {
        text-align: center;
        padding: 2.2rem 1rem 1.6rem 1rem;
    }
    .hero-icon {
        font-size: 2.6rem;
        width: 84px;
        height: 84px;
        line-height: 84px;
        margin: 0 auto 1.1rem auto;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, rgba(139,92,246,0.35), rgba(34,211,238,0.12) 60%, transparent 75%);
        border: 1px solid rgba(139,92,246,0.45);
        box-shadow: 0 0 40px rgba(139,92,246,0.35);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #22d3ee 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        color: #93a0b8;
        font-size: 1.02rem;
        margin-top: 0.6rem;
    }
    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.32rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #b9c4d8;
        background: rgba(139,92,246,0.08);
        border: 1px solid rgba(139,92,246,0.35);
    }

    /* ---- Score cards ---- */
    .score-card {
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        background: linear-gradient(145deg, rgba(139,92,246,0.10), rgba(34,211,238,0.05));
        border: 1px solid rgba(139,92,246,0.30);
        text-align: center;
    }
    .score-card .label {
        color: #93a0b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .score-card .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #e6e9f0;
        margin-top: 0.15rem;
    }
    .score-card .value span {
        font-size: 1.1rem;
        color: #93a0b8;
        font-weight: 500;
    }

    /* ---- Verdict banner ---- */
    .verdict-banner {
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-top: 0.4rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
    }
    .verdict-pass {
        background: linear-gradient(90deg, rgba(34,211,238,0.14), rgba(139,92,246,0.14));
        border: 1px solid rgba(34,211,238,0.45);
        color: #7ee7f5;
    }
    .verdict-fail {
        background: linear-gradient(90deg, rgba(251,191,36,0.12), rgba(244,114,182,0.10));
        border: 1px solid rgba(251,191,36,0.4);
        color: #fbbf24;
    }
    .verdict-reason {
        color: #93a0b8;
        font-size: 0.88rem;
        font-weight: 400;
        margin-top: 0.35rem;
        font-family: 'Inter', sans-serif;
    }

    /* Native widget polish */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(139,92,246,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-icon">🧑‍💼</div>
        <div class="hero-title">AI Career Copilot</div>
        <div class="hero-subtitle">Upload your resume, pick a target role, and get an instant AI-graded evaluation.</div>
        <div class="hero-badges">
            <span class="pill">🧠 AI-Powered Analysis</span>
            <span class="pill">🛡️ ATS Optimized</span>
            <span class="pill">⚡ Instant Feedback</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

with st.container(border=True):
    selected_role = st.selectbox("🎯 Select Target Role", list(ROLE_JD_MAP.keys()))

    if selected_role == "Auto Detect (from JD)":
        jd_text = st.text_area(
            "Paste Job Description",
            height=180,
            placeholder="Paste the job description here...",
        )
    else:
        jd_text = ROLE_JD_MAP[selected_role]

    uploaded_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])
    evaluate_clicked = st.button("🚀 Evaluate Resume", use_container_width=True)

if evaluate_clicked:
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
        with c1:
            st.markdown(
                f"""<div class="score-card">
                    <div class="label">Skill Score</div>
                    <div class="value">{result['skill_score']}<span>/100</span></div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="score-card">
                    <div class="label">Experience Score</div>
                    <div class="value">{result['experience_score']}<span>/100</span></div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.write("")

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
        is_pass = result["verdict"] == "Interview Ready (Fresher)"
        verdict_icon = "🟢" if is_pass else "🟡"
        banner_class = "verdict-pass" if is_pass else "verdict-fail"
        reason_html = (
            f'<div class="verdict-reason">{result["verdict_reason"]}</div>'
            if result.get("verdict_reason")
            else ""
        )
        st.markdown(
            f"""<div class="verdict-banner {banner_class}">
                {verdict_icon} {result['verdict']}
                {reason_html}
            </div>""",
            unsafe_allow_html=True,
        )

        st.write("")

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
            use_container_width=True,
        )

    except Exception as e:  # noqa: BLE001
        st.error(f"❌ Error: {e}")
        with st.expander("Show full traceback"):
            st.code(traceback.format_exc())
