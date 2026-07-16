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

## 📋 Milestone Sprint Delivery

1. **Sprint 1 (Foundation)**: Setup FastAPI routers, SQLAlchemy models for SQLite, custom logging, React components shell, and unit route validations.
2. **Sprint 2 (Knowledge Retrieval Engine)**: Integrated PyMuPDF layout parser, BGE embeddings running in offline cache mode, and local FAISS vector indexing.
3. **Sprint 3 (Hiring Decision Engine)**: Algorithmic assessment metrics mapping experience alignment, project relevance, detail confidence, and resume quality.
4. **Sprint 4 (LangGraph & UI Dashboard)**: Structured state graph compilation linking sequential agent nodes. Created client-side JSON downloads and sub-score metrics.
5. **Sprint 5 (Intelligence Refinement)**: Unified config weight calculation metrics, requirement classification taxonomies, skill relationships transferability, and validation questions.
6. **Final Sprint (Production Polish)**: Added verification assertions, documentation logs, and normalized error handlers.

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
