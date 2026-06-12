# AI Career Copilot 🧠

[![CI](https://github.com/k-satyam215/ai-career-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/k-satyam215/ai-career-copilot/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-157%20passing-brightgreen)](tests/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](api/main.py)
[![MCP](https://img.shields.io/badge/tool--layer-MCP-7c3aed)](mcp_server/server.py)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED)](docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Not a ChatGPT wrapper.** An end-to-end agentic system that evaluates resumes against job descriptions using hybrid RAG, deterministic rule-based agents, LLM agents, a FastAPI service layer, an MCP tool layer, and a Streamlit UI — all wired together in a production-oriented pipeline.

---

## ✨ What Makes This Different

| Concern | Approach |
|---|---|
| Resume scoring | Deterministic keyword-density agents (no hallucination) |
| Evidence retrieval | HYDE + MMR (diversity-aware semantic search) |
| LLM usage | Scoped to improvement bullets, interview Q&A only |
| Role detection | JD intent + resume strength → adaptive label |
| Freshness | Fresher-safe: no fake metrics, no fabricated experience |
| Explainability | Each agent has a single responsibility, traceable output |

---

## 🏗️ Architecture

```
PDF Resume ──► loader ──► chunker
                              │
                         retriever ◄── HYDE + MMR + keyword filter
                              │
                         AgentState
                         ┌────┴────────────────────────────┐
                         │  skill_agent    (deterministic)  │
                         │  experience_agent (deterministic)│
                         │  ats_agent       (deterministic) │
                         │  improvement_agent   (LLM)       │
                         │  llm_interview_agent (LLM)       │
                         │  llm_answer_agent    (LLM)       │
                         │  judge_agent     (deterministic) │
                         └────────────────────────────────-─┘
                                        │
                    ┌───────────────────┼───────────────────┐
                 FastAPI             Streamlit           MCP Server
               /evaluate             UI (port             (tool layer
              /health                 8501)               for agents)
```

---

## 🚀 Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/k-satyam215/ai-career-copilot
cd ai-career-copilot
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit UI

```bash
streamlit run ui_streamlit.py
```

### 4. Run the FastAPI server

```bash
uvicorn api.main:app --reload
# Swagger docs → http://localhost:8000/docs
```

### 5. Run the MCP server

```bash
python -m mcp_server.server
```

---

## 🐳 Docker (Recommended for Production)

```bash
cp .env.example .env   # fill in GROQ_API_KEY
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI API | http://localhost:8000 |
| Streamlit UI | http://localhost:8501 |
| MCP Server | internal |

---

## 🧪 Tests

```bash
pytest --cov=. --cov-report=term-missing
```

157 tests covering:
- All deterministic agents (skill, experience, ATS, judge)
- All LLM agents with mocked LLM (improvement, interview, answer)
- Retrieval modules (chunking, HYDE, MMR, hybrid, retriever)
- Role detection and config
- LLM provider caching + error handling
- FastAPI endpoints (evaluate, health, validation)
- MCP server tools

---

## 📡 API Reference

### `GET /health`
```json
{"status": "ok"}
```

### `POST /evaluate`
Multipart form:
- `resume` — PDF file
- `jd_text` — job description string

Response:
```json
{
  "role": "GenAI Engineer",
  "skill_score": 70.0,
  "experience_score": 64.0,
  "ats_issues": [],
  "improvement_suggestions": ["..."],
  "interview_questions": ["..."],
  "interview_answers": ["..."],
  "verdict": "Interview Ready (Fresher)"
}
```

---

## 🛠️ MCP Tool Layer

Three tools exposed via `mcp_server/server.py`:

| Tool | Description |
|---|---|
| `parse_resume` | Load PDF → chunk list |
| `retrieve_resume_evidence` | HYDE + MMR evidence extraction |
| `evaluate_resume_against_jd` | Full pipeline → structured result |

---

## 📁 Project Structure

```
ai-career-copilot/
├── agents/                  # All evaluation agents
│   ├── skill_agent.py       # Deterministic skill scoring
│   ├── experience_agent.py  # Deterministic experience scoring
│   ├── ats_agent.py         # ATS structure validation
│   ├── improvement_agent.py # LLM resume bullet rewriting
│   ├── llm_interview_agent.py # LLM interview question gen
│   ├── llm_answer_agent.py  # LLM interview answer gen
│   ├── judge_agent.py       # Final verdict
│   └── state.py             # Shared AgentState TypedDict
├── retrieval/
│   ├── loader.py            # PDF text extraction (PyMuPDF)
│   ├── chunking.py          # Fixed-size resume chunker
│   ├── hyde.py              # Hypothetical profile generation
│   ├── mmr.py               # Maximal Marginal Relevance
│   ├── hybrid.py            # Dense + sparse hybrid search
│   ├── embeddings.py        # Lazy sentence-transformer wrapper
│   └── retriever.py         # Orchestrates retrieval pipeline
├── api/
│   └── main.py              # FastAPI app (/evaluate, /health)
├── mcp_server/
│   └── server.py            # MCP tool layer (3 tools)
├── tests/                   # 157 pytest tests
├── career_pipeline.py       # End-to-end pipeline orchestrator
├── role_detector.py         # JD + resume role inference
├── roles_config.py          # Skill domains and thresholds
├── llm_provider.py          # Cached Groq LLM client
├── ui_streamlit.py          # Streamlit frontend
├── Dockerfile               # Multi-stage: api / ui / mcp
├── docker-compose.yml       # All services
├── .github/workflows/ci.yml # Lint → test → Docker build
└── requirements.txt
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Model ID |
| `GROQ_TEMPERATURE` | `0.3` | Sampling temperature |

---

## 🎯 Design Decisions

**Why Groq + LLaMA?** Fast inference, generous free tier — ideal for fresher portfolio demos.

**Why deterministic agents for scoring?** Prevents hallucinated scores. Scores are explainable and reproducible.

**Why HYDE?** Direct query embedding misses intent. A hypothetical ideal-candidate profile aligns the embedding space with JD expectations.

**Why MMR?** Prevents redundant chunks — diverse evidence = richer context for LLM agents.

**Why MCP?** Enables any LLM host (Claude Desktop, Cursor, etc.) to drive the evaluation pipeline as tool calls.

---

## 👤 Author

**Satyam Kumar** — [LinkedIn](https://linkedin.com/in/satyam-kumar) · [GitHub](https://github.com/k-satyam215)
