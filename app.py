from career_pipeline import evaluate_resume

# ---------------- INPUTS ----------------

# Resume path
resume_path = "data/resumes/sample_resume.pdf"

# Job Description (file se)
with open("data/jds/ai.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()

# ---------------- RUN PIPELINE ----------------

result = evaluate_resume(resume_path, jd_text)

# ---------------- OUTPUT ----------------

print("\n===== AI CAREER COPILOT RESULT =====\n")

print("Detected Role:", result["role"])
print("Skill Score:", result["skill_score"])
print("Experience Score:", result["experience_score"])
print("ATS Issues:", result["ats_issues"])
print("FINAL VERDICT:", result["verdict"])

print("\n===================================\n")
