FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# ---- API service ----
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Streamlit UI service ----
FROM base AS ui
EXPOSE 8501
CMD ["streamlit", "run", "ui_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]

# ---- MCP server ----
FROM base AS mcp
CMD ["python", "-m", "mcp_server.server"]
