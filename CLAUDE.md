# CLAUDE.md — AI Hiring Intelligence Platform

The permanent brain of this project. Every Claude Code session reads this first. It is the source of truth for what the system is, how it runs, the rules that never change, and how to work in it.

---

## 1. System Overview

**Purpose.** The platform evaluates how well a candidate matches a Job Description using an explainable **Agentic Retrieval-Augmented Generation (Agentic RAG)** pipeline.

Unlike ATS systems that rely on keyword matching, it performs semantic retrieval, evidence gathering, reasoning, transferability estimation, and recruiter-oriented decision making. **Every hiring decision must be explainable and traceable back to retrieved evidence.**

It is **NOT** a chatbot, **NOT** a keyword ATS, **NOT** a static rule engine.

Active project: `ai_hiring_platform/`. A legacy personality-predictor project (`src/`, `data/`, `model_architecture.md`) also sits at the repo root — **do not touch it** during hiring-platform work.

---

## 2. Runtime Flow (end to end)

```
Resume PDF/DOCX + Job Description
        │
        ▼
Document Parsing            (services/ai/document_loader.py)
        │
        ▼
Semantic Chunking           (services/ai/resume_structuring_service.py — section-aware TextNodes)
        │
        ▼
Embedding Generation        (services/ai/embedding_service.py — BAAI/bge-small-en-v1.5)
        │
        ▼
FAISS Index (per resume)    (services/ai/vector_store_service.py)
        │
        ▼
Requirement Extraction      (services/ai/jd_requirement_extractor.py)
        │
        ▼
Semantic Retrieval + Ranking(services/ai/retrieval_service.py)
        │
        ▼
╔═══════════════════════════════════╗
║ Candidate Intelligence Agent      ║  ── gathers evidence (Node 1)
╚═══════════════════════════════════╝
        │
        ▼
Structured Evidence Package (LangGraph state: evidence_report)
        │
        ▼
╔═══════════════════════════════════╗
║ Hiring Decision Agent             ║  ── reasons over evidence (Node 2)
╚═══════════════════════════════════╝
        │
        ▼
LLM Reasoning + algorithmic scoring (services/ai/evaluation_service.py, skill_semantics_service.py)
        │
        ▼
Structured Hiring Report    (schemas/analysis.py :: HiringReport)
        │
        ▼
Frontend Visualization      (React — pages/Analysis)
```

LangGraph wiring (`workflows/hiring_workflow.py`): `START → candidate_intelligence → hiring_decision → END`. Sequential, no branching, no loops.

---

## 3. The Two Runtime Agents

The application contains **exactly two runtime AI agents**. Do not add a third (no planner, summarizer, reflection, or memory agent). Development sessions (Backend/Frontend/QA Claudes) are **not** runtime agents — never confuse them.

### Agent 1 — Candidate Intelligence Agent (the investigator)
> Class name is `CandidateIntelligenceAgent`; its true role is **evidence intelligence**. Do not rename the class — just understand it as the evidence gatherer.

Responsibilities:
- Parse documents · build embeddings · build the vector store · perform semantic retrieval · rank evidence · assemble a structured evidence package.

Must **never**:
- make hiring decisions · reject candidates · hallucinate.

**Output: evidence only.**

### Agent 2 — Hiring Decision Agent (the hiring manager)
Consumes **only** the evidence package. Must **never** read PDFs, search FAISS, or compute embeddings. It only reasons over evidence.

Produces: sub-scores + overall score · recruiter recommendation · learning roadmap · interview questions · rejection email · strengths/weaknesses/gaps · explanations.

**Why exactly two agents:** it mirrors real hiring — one party gathers evidence, another decides. Separating "collect" from "decide" is what makes every conclusion auditable and prevents a black box.

---

## 4. Algorithms vs. LLM — the division of labor

**The LLM should not do everything.** Use deterministic algorithms whenever the answer must be reproducible; use the LLM only where genuine natural-language reasoning is required. This split is permanent.

**Deterministic algorithms (reproducible):**
- ✓ embeddings ✓ semantic search ✓ cosine similarity ✓ evidence ranking
- ✓ confidence calculation ✓ score aggregation ✓ transferability estimation
- ✓ category classification (nearest-centroid)

**LLM (reasoning / generation):**
- ✓ reasoning over evidence ✓ explanations ✓ recruiter recommendation
- ✓ learning roadmap narrative ✓ interview questions ✓ natural-language report text

When no LLM key is configured, a deterministic mock engine produces the full report — the algorithmic backbone must always work without the LLM.

---

## 5. Golden Rules (never violated)

1. Never bypass semantic retrieval.
2. Never let the LLM directly inspect raw PDFs/DOCX.
3. Evidence must exist **before** reasoning.
4. No hardcoded skill mappings or relationships. **Algorithms over examples.**
5. Everything must generalize to any resume and any JD.
6. Keep backend and frontend independent (backend never edits UI; frontend never changes algorithms).
7. No duplicated business logic — one responsibility per module; reuse services.
8. Every score/decision must be explainable: Requirement → Evidence → Reasoning → Confidence → Decision.
9. Backend owns the API contract; frontend mirrors it.
10. Tests must remain green; the deterministic path stays tested.

(Git rule, also non-negotiable: **no Claude/AI authorship traces** in commits/PRs — no `Co-Authored-By: Claude`, no "Generated with Claude Code". Author is the user only.)

---

## 6. Project Principles

- **Explainability** — every output traces to evidence.
- **Reproducibility** — deterministic where the answer must be exact.
- **Evidence-first reasoning** — retrieve, then reason.
- **Modularity** — clean service/agent/router/model boundaries.
- **Maintainability** — small, single-responsibility units; config-driven thresholds.
- **Scalability** — local-first now, architecture that can grow without rewrites.
- **Human-readable reports** — recruiter-grade, not raw model output.
- **Interview-grade engineering** — every decision defensible (see `.claude/project/interview_notes.md`).

---

## 7. Repository Structure (folder responsibilities)

```
ai_hiring_platform/
├── backend/app/
│   ├── agents/          # the two runtime agents (candidate_intelligence, hiring_decision) — orchestrate services
│   ├── services/ai/     # all logic: document_loader, resume_structuring, embedding, vector_store,
│   │                    #   jd_parser, jd_requirement_extractor, retrieval, evaluation,
│   │                    #   skill_semantics (generic classifier + transfer), report
│   ├── workflows/       # hiring_workflow.py — the LangGraph state machine
│   ├── api/v1/routers/  # HTTP endpoints: health, resume, job, analysis (thin; delegate to services/agents)
│   ├── schemas/         # Pydantic request/response shapes — the API CONTRACT
│   ├── models/          # SQLAlchemy tables (Resume, JobDescription, Analysis)
│   ├── core/            # config, constants, database, logging, exceptions, dependencies
│   ├── rag/ prompts/ repositories/   # interface/scaffolding packages (extend as needed)
│   └── tests/           # pytest suite (11 tests): rag, hiring_decision, workflows, routes
├── frontend/src/
│   ├── pages/           # Dashboard, Analysis, Job, Resume, SystemStatus
│   ├── types/           # index.ts — API types (mirror of backend schemas)
│   ├── assets/          # static assets
│   ├── App.tsx main.tsx # app shell + entry
│   └── components/ hooks/   # (convention — introduce as UI grows; do not fabricate prematurely)
└── docs/                # architecture.md, Algorithms.md, Workflow.md
```
One-line rule per layer: **services** implement logic · **agents** orchestrate · **routers** expose · **schemas** define the contract · **models** hold data · **core** holds config. Never put algorithms in routers or reasoning in Agent 1.

**Stack:** BGE embeddings · FAISS · SQLite · PDF/DOCX. Only LLM reasoning is optional/remote (OpenAI or Anthropic via llama-index; deterministic mock fallback). Everything else is local.

---

## 8. How to add a feature (the recipe)

```
Plan → Design → Backend → Schema (contract) → Frontend → Testing → Verification → Merge → Push
```
1. **Plan** the vertical slice; record it in `.claude/project/roadmap.md`.
2. **Design**; identify contract impact and dependencies.
3. **Backend** implements logic in a service, orchestrated by an agent (never inline in a router).
4. **Schema** — if the contract changes, backend lands `schemas/*.py` first and notes it in `api_contract.md` + `decisions.md`.
5. **Frontend** mirrors the contract into `types/index.ts` and builds the UI (mock until backend merges).
6. **Testing** — add/extend pytest + keep the build green.
7. **Verification** — `/verify-all` (backend + build + integration).
8. **Merge** into `main` in the main worktree (`git merge --no-ff dev/backend` then `dev/frontend`).
9. **Push** only when the full gate is green (`/release`), then sync dev branches forward.

---

## 9. How to debug (playbook)

**Frontend fails →** check the API contract → check `types/index.ts` vs backend `schemas` → check the actual response shape → check the build/type error at its source.

**Backend fails →** run pytest and read the failing assertion → check the LangGraph node/state → check the responsible service → check embeddings/FAISS (model loaded? index present?) → check config thresholds.

**Empty/odd report →** confirm evidence was retrieved *before* reasoning (Agent 1 output non-empty); Agent 2 can only reason over what Agent 1 gathered.

**LLM issues →** with no API key the deterministic mock engine runs; confirm which path executed before debugging reasoning output.

---

## 10. Working discipline (how every session behaves)

Do not jump straight to code. For any non-trivial task: **Inspect → Design → Explain → Implement → Test → Verify → Summarize.** Never assume — read the code first. Report honestly, including anything skipped or still red. Stay in your lane (backend/frontend/QA boundaries).

---

## 11. Parallel development workflow

| Role | Worktree | Branch | Scope |
|---|---|---|---|
| Backend (Session 1) | `../cc2-backend` | `dev/backend` | RAG, FAISS, embeddings, retrieval, agents, algorithms, FastAPI, DB, schemas, evaluation, LLM. Never edits `frontend/**`. |
| Frontend (Session 2) | `../cc2-frontend` | `dev/frontend` | React/TS/Tailwind, components, dashboard, charts, UX, API integration. Treats backend as an API. |
| Integration/QA (You) | this repo (main) | `main` | merge, full pytest, build, run app, regression, push. Fixes only verified bugs. |

**Contract-first:** the only coordination surface is the API contract (backend `schemas` ↔ frontend `types`). Backend lands the schema first; frontend mocks until merged. Develop locally on dev branches; merge into `main`; **push `origin/main` only when green.** After a checkpoint, sync dev branches forward (`git merge main`).

---

## 12. Verification commands

Backend (reuse the main worktree venv — no per-worktree reinstall):
```
cd ai_hiring_platform/backend
"E:/projects from desktops/codeclause2/ai_hiring_platform/backend/.venv/Scripts/python.exe" -m pytest -q
```
Frontend:
```
cd ai_hiring_platform/frontend
npm run build      # tsc -b && vite build
npm run lint       # oxlint
```
Baseline: **120 tests pass**, **build exit 0**. First backend run is slow (BGE load).

---

## 13. Knowledge base map

- **Skills** (`.claude/skills/`): `architecture`, `rag`, `langgraph`, `backend-dev`, `frontend-dev`, `testing`, `qa-verify`, `git-workflow`.
- **Commands** (`.claude/commands/`): `/verify-backend`, `/verify-frontend`, `/verify-all`, `/build`, `/merge`, `/release`.
- **Project docs** (`.claude/project/`): `roadmap.md`, `api_contract.md` (coordination surface), `decisions.md` (ADR log), `session_log.md` (cross-session memory), `interview_notes.md` (design-rationale Q&A bank), `future_vision.md` (direction, non-binding).

After meaningful work, append to `session_log.md`; on significant choices, append to `decisions.md`.

---

## 14. Future Vision (direction, not commitments)

Possible future directions — see `.claude/project/future_vision.md`. Examples: candidate/recruiter dashboards, interview simulator, conversation/memory, multi-company support, multi-modal resume analysis. **None may violate the core architecture** (two agents, evidence-first, algorithms-over-examples, explainability).

---

## 15. Known non-blocking debt

Pydantic v2 `class Config` → `ConfigDict` (schemas); outdated Anthropic model id in `evaluation_service.get_llm()`; `datetime.utcnow()` deprecation. Warnings only — fix opportunistically.
