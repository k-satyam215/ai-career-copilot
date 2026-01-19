# roles_config.py

# Skill domains (general & extensible)
SKILL_DOMAINS = {
    "genai": [
        "llm", "rag", "langchain", "prompt",
        "agent", "vector", "embedding",
        "chroma", "faiss", "react", "genai"
    ],

    "backend": [
        "backend", "api", "rest", "graphql",
        "flask", "django", "fastapi",
        "database", "sql", "microservice",
        "scalability", "performance"
    ],

    "data": [
        "data analysis", "pandas", "numpy",
        "etl", "warehouse", "dashboard",
        "visualization", "analytics"
    ],

    "ml": [
        "machine learning", "model training",
        "classification", "regression",
        "feature engineering", "evaluation",
        "metrics", "scikit-learn"
    ],

    "frontend": [
        "frontend", "react", "javascript",
        "html", "css", "ui", "ux",
        "streamlit", "dashboard"
    ],

    "devops": [
        "docker", "kubernetes", "ci/cd",
        "aws", "gcp", "azure",
        "deployment", "monitoring"
    ]
}

# Generic thresholds (used by pipeline)
EVALUATION_THRESHOLDS = {
    "skill_score": 60,
    "experience_score": 55
}

# 🔥 BACKWARD COMPATIBILITY (IMPORTANT)
# Old pipeline still expects ROLE_CONFIG
ROLE_CONFIG = {
    domain: {
        "keywords": keywords,
        "skill_threshold": EVALUATION_THRESHOLDS["skill_score"],
        "experience_threshold": EVALUATION_THRESHOLDS["experience_score"]
    }
    for domain, keywords in SKILL_DOMAINS.items()
}
