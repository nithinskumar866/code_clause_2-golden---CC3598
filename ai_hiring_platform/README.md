# AI Hiring Intelligence Platform

> Explainable Multi-Agent Retrieval-Augmented Generation (RAG) Recruitment Dashboard Orchestrated via LangGraph.

This platform operates as an explainable recruitment copilot. It structures candidate resumes against job descriptions, indexes them in a local vector store, and runs intent-based evaluation reasoning to output multi-perspective compatibility scores.

---

## 🏛 System Design Architecture

```text
                         React Client (Vite SPA)
                                    │
                                    ▼
                         FastAPI Routing Engine
                                    │
                                    ▼
                     LangGraph State Graph Workflow
                                    │
             ┌──────────────────────┴──────────────────────┐
             │                                             │
             ▼                                             ▼
  Candidate Intelligence Agent                   Hiring Decision Agent
  [Node 1: Local RAG Pipeline]                  [Node 2: LLM Reasoning]
             │                                             │
      Parser (PyMuPDF)                                     │
             │                                             │
    TextNodes Structuring                                  │
             │                                             │
  Embedding (BGE-Small v1.5)                               │
             │                                             │
   Vector Index (FAISS)                                    │
             │                                             │
   JD Semantic Retrieval                                   │
             │                                             │
             └──────────────► Evidence State ──────────────┘
                                                           │
                                                           ▼
                                                Explainable Report
                                                (Weights Normalized)
```

---

## ✨ Features (recruiter-facing)

- **Resume & JD upload** — PDF/DOCX, parsed and locally embedded (BGE) into a per-resume FAISS index.
- **Explainable evaluation** — two-agent Agentic RAG (evidence gathering → reasoning) producing a full hiring report where every score traces to retrieved evidence.
- **Authenticity & keyword-stuffing detection** — deterministic credibility score, over-claimed-skill flags, corroboration ratio.
- **Candidate profile & seniority fit** — name/title/years/level vs the JD's stated requirement.
- **Requirement matching** — per-requirement evidence, confidence, must-have vs nice-to-have badges, importance-weighted coverage.
- **Skill gap + learning roadmap + interview questions + recruiter recommendation**.
- **Multi-candidate ranking** — score many resumes against one JD, leaderboard with top candidate, sort/filter, CSV export.
- **Analytics** — totals, average score, score distribution, recommendation split, daily/weekly/monthly trends, top resumes/jobs, most-common missing skills.
- **Recruiter workflow** — editable, persistent pipeline stage (Applied → Screening → Reviewed → Interview Scheduled → Interview Completed → Selected → Rejected → Offer Sent).
- **Recruiter notes** — full add/edit/delete CRUD per candidate.
- **History** — search, filter, sort, paginate; open any candidate; JSON + server-rendered **PDF** export.
- **Offline-first** — the deterministic engine produces the entire report with no LLM key; add a provider key to enable LLM reasoning.

**Retrieval quality**: section-aware, sentence-aware chunking; hybrid retrieval fusing dense (FAISS cosine) and sparse (BM25) rankings via Reciprocal Rank Fusion.

---

## 🔌 API surface (`/api/v1`)

`health` · `resume` (upload/list) · `job` (upload/list) · `analysis/evaluate` · `analysis/rank` ·
`analysis/history` (list/detail/delete/clear) · `dashboard/*` · `analytics/*` (overview, score-distribution,
recommendation-distribution, trends, top-resumes, top-jobs, skill-frequency, recent) ·
`analysis/{id}/notes` + `notes/{id}` (CRUD) · `analysis/{id}/status` (get/patch) ·
`analysis/{id}/export/json|pdf`. Every response uses the `{success, message, data, meta?}` envelope
(file exports stream bytes). Interactive docs at `http://localhost:8000/docs`.

---

## ⚙ Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.12+)

---

### Backend Setup
1. Navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install the python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   The backend API docs are available at `http://localhost:8000/docs`.

5. Run the automated test suites:
   ```bash
   python -m pytest
   ```

---

### Frontend Setup
1. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install npm modules:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   Vite will serve the page at `http://localhost:5173`.

4. Build production static bundle:
   ```bash
   npm run build
   ```

---

## 🔧 Environment variables

Copy the provided examples and edit as needed:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env      # optional; only to change the API host
```

**Backend** (`backend/.env`) — all optional:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `google` (aliases: `gemini`, `claude`) |
| `LLM_MODEL` | *(per-provider default)* | Explicit model id |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | *(empty)* | Set only the one for your provider. **All empty → deterministic engine.** |
| `DATABASE_URL` | `sqlite:///./hiring_platform.db` | SQLAlchemy URL |
| `RETRIEVAL_MIN_SIMILARITY` | `0.30` | Cosine floor below which evidence is dropped |
| `HYBRID_RETRIEVAL_ENABLED` | `true` | Fuse BM25 + vector via RRF |
| `HYBRID_WEIGHT_DENSE` / `HYBRID_WEIGHT_SPARSE` / `HYBRID_RRF_K` | `0.6` / `0.4` / `60` | Fusion tuning |
| `CHUNK_TARGET_CHARS` / `CHUNK_MAX_CHARS` / `CHUNK_SENTENCE_OVERLAP` | `350` / `600` / `1` | Sentence-aware chunking |
| `LOG_LEVEL` | `INFO` | Log verbosity |

**Frontend** (`frontend/.env`): `VITE_API_BASE` (default `http://localhost:8000`) — scheme + host of the backend; the client appends `/api/v1`.

---

## 🚀 Deployment

**Backend** (any ASGI host):
```bash
cd backend
pip install -r requirements.txt
# Production server (no --reload); scale workers to taste.
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```
- SQLite is created/migrated automatically on startup (`init_db` + additive `ensure_schema`). For multi-worker or containerized deployments, point `DATABASE_URL` at a shared DB and mount `backend/storage/` (FAISS indices + report JSONs) on a persistent volume.
- CORS currently allows all origins (dev-friendly). Restrict `allow_origins` in `app/main.py` for production.
- First run downloads/loads the BGE model (slow once, then cached).

**Frontend** (static SPA):
```bash
cd frontend
VITE_API_BASE=https://your-backend.example.com npm run build
# Serve the ./dist folder from any static host / CDN (Nginx, Netlify, S3+CloudFront, …).
```

**Verification gate** (run before shipping):
```bash
# backend
cd backend && python -m pytest -q                 # 133 passed
# frontend
cd frontend && npm run build && npm run lint       # tsc+vite exit 0, oxlint clean
```
