# ui_streamlit.py

import streamlit as st
import tempfile

from career_pipeline import evaluate_resume

st.set_page_config(page_title="AI Career Copilot", layout="centered")

st.title("🧑‍💼 AI Career Copilot")
st.write("Upload your resume and select the target role.")

# ---------------- ROLE SELECTION ---------------- #

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

    "Auto Detect (from JD)": ""
}

selected_role = st.selectbox(
    "Select Target Role",
    list(ROLE_JD_MAP.keys())
)

# If auto-detect, allow custom JD input
if selected_role == "Auto Detect (from JD)":
    jd_text = st.text_area(
        "Paste Job Description",
        height=180,
        placeholder="Paste the job description here..."
    )
else:
    jd_text = ROLE_JD_MAP[selected_role]

# ---------------- RESUME UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# ---------------- RUN PIPELINE ---------------- #

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

    result = evaluate_resume(resume_path, jd_text)

    # ---------------- OUTPUT ---------------- #

    st.subheader("🧠 Evaluation Result")
    st.write("**Detected Role Profile:**", result["role"])

    c1, c2 = st.columns(2)
    c1.metric("Skill Score", result["skill_score"])
    c2.metric("Experience Score", result["experience_score"])

    st.subheader("📄 ATS Check")
    if result["ats_issues"]:
        for issue in result["ats_issues"]:
            st.warning(issue)
    else:
        st.success("No ATS issues found")

    st.subheader("✍️ Resume Improvement Suggestions")
    for s in result["improvement_suggestions"]:
        st.info(s)

    st.subheader("🎯 Interview Questions & Answers")
    for i, (q, a) in enumerate(
        zip(result["interview_questions"], result["interview_answers"]), 1
    ):
        st.markdown(f"**Q{i}. {q}**")
        st.markdown(f"🗣️ {a}")

    st.subheader("✅ Final Verdict")
    st.success(result["verdict"])
