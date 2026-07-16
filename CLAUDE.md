# CLAUDE.md — AI Hiring Intelligence Platform

Authoritative context for every Claude Code session in this repo. Read this first; it is the source of truth for architecture, rules, and workflow.

---

## 1. What this project is

An **Explainable AI Hiring Intelligence Platform** built on **Agentic RAG**.

It answers one question for a recruiter: *"Is this candidate genuinely suitable for this job, and why?"* — with every conclusion traceable to resume evidence.

It is **NOT** a chatbot. **NOT** a keyword ATS. **NOT** a static rule engine.

Active project lives in `ai_hiring_platform/`. A legacy personality-predictor project (`src/`, `data/`, `model_architecture.md`) also exists at the repo root — **do not touch it** during hiring-platform work.

---

## 2. Runtime architecture — exactly TWO agents

The application contains **exactly two runtime AI agents**, orchestrated by a sequential LangGraph workflow:

```
START → Candidate Intelligence Agent → Hiring Decision Agent → END
```

- **Agent 1 · Candidate Intelligence** (the investigator): parse → structure → embed (BGE) → FAISS index → extract JD requirements → retrieve per-requirement evidence → return **structured evidence**. It gathers; it never decides.
- **Agent 2 · Hiring Decision** (the hiring manager): reasons **only** over Agent 1's evidence. It never re-reads raw files. Produces the explainable report (compatibility, strengths, gaps, transferable skills, roadmap, questions, recommendation, confidence).

**Do NOT introduce additional runtime agents** — no planner, summarizer, reflection, or memory agent. The LangGraph is intentionally simple: sequential, no branching, no loops.

> Development sessions (Backend/Frontend/QA Claudes) are **not** runtime agents. Never confuse the two.

---

## 3. The reasoning contract (how features must work)

```
Input → RAG retrieves evidence → LLM reasons over evidence → Algorithm validates/explains → Structured output
```

- The Candidate Intelligence Agent retrieves **structured evidence first**. The Hiring Decision Agent reasons **only from that evidence** — never point the LLM at raw documents.
- Prefer LLM reasoning over deterministic business logic where reasoning is the right tool. Use algorithms to **support and validate** reasoning, not replace it.

---

## 4. Non-negotiable rules

- **Algorithms, not examples.** Never `if skill == "python"`. Iterate over extracted requirements/entities. Everything must generalize to *any* resume and *any* JD.
- **Never hardcode** skills, technologies, relationships, or examples as implementation logic. Examples inside prompts are illustrative only.
- **Explainability:** every score/decision must trace Requirement → Evidence → Reasoning → Confidence → Decision.
- **Modular monolith:** services implement logic; agents orchestrate; routers expose APIs; models hold data; constants/config hold configuration.
- **Git:** never add Claude/AI authorship traces — **no `Co-Authored-By: Claude`, no "Generated with Claude Code"** in commits or PRs. Commits are authored solely by the user.

---

## 5. Repository structure

```
ai_hiring_platform/
├── backend/                       # FastAPI + LangGraph + local RAG
│   ├── app/
│   │   ├── agents/                # candidate_intelligence/, hiring_decision/  (ONLY these two)
│   │   ├── workflows/             # hiring_workflow.py  (the LangGraph)
│   │   ├── services/ai/           # document_loader, embedding, vector_store, retrieval,
│   │   │                          #   jd_parser, jd_requirement_extractor, evaluation,
│   │   │                          #   skill_semantics (generic classifier + transfer), report
│   │   ├── api/v1/routers/        # health, resume, job, analysis
│   │   ├── schemas/               # analysis, job, resume, response  ← API CONTRACT
│   │   ├── core/                  # config, constants, database, logging, exceptions
│   │   └── models/                # SQLAlchemy models
│   ├── tests/                     # pytest suite (11 tests)
│   └── .venv/                     # local, gitignored (multi-GB ML deps)
├── frontend/                      # React + Vite + TypeScript + Tailwind
│   └── src/
│       ├── pages/                 # Dashboard, Analysis, Job, Resume, SystemStatus
│       ├── types/index.ts         # ← API CONTRACT (mirror of backend schemas)
│       └── ...
└── docs/                          # architecture.md, Algorithms.md, Workflow.md
```

**Stack:** BAAI/bge-small-en-v1.5 embeddings · FAISS · SQLite · PDF/DOCX. Only LLM reasoning is optional/remote (OpenAI or Anthropic via llama-index; deterministic mock engine fallback when no API key). Everything else is local.

---

## 6. Parallel development workflow

Three roles, two new worktrees, main worktree as the hub:

| Role | Worktree | Branch | Scope |
|---|---|---|---|
| Backend (Session 1) | `../cc2-backend` | `dev/backend` | LangGraph, RAG, FAISS, embeddings, retrieval, agents, algorithms, FastAPI, DB, schemas, evaluation, LLM. **Never edits `frontend/**`.** |
| Frontend (Session 2) | `../cc2-frontend` | `dev/frontend` | React/TS/Tailwind, components, dashboard, charts, UX, API integration. **Treats backend as an API; never changes backend algorithms.** |
| Integration/QA (You) | this repo (main) | `main` | merge, full pytest, npm build, run app, regression, push. **Only fixes verified bugs; builds no features.** |

**Contract-first:** the only coordination surface is the API contract — backend `{schemas/*.py, response.py, routers/*}` ↔ frontend `{types/index.ts, API client}`. Backend lands the schema first; frontend builds against the agreed shape (mock until merged). Both sessions work independently and only merge when a vertical slice is complete.

**Merge & push:** slice done → in main worktree `git merge --no-ff dev/backend` then `dev/frontend` (disjoint trees ⇒ conflict-free) → run full verification → **push `origin/main` only when green.** GitHub holds verified checkpoints only. After merging, sync dev branches forward (`git merge main`).

---

## 7. Verification commands

Backend (reuse the main worktree's venv — no per-worktree reinstall):
```
cd ai_hiring_platform/backend
"E:/projects from desktops/codeclause2/ai_hiring_platform/backend/.venv/Scripts/python.exe" -m pytest -q
```
Frontend:
```
cd ai_hiring_platform/frontend
npm run build        # tsc -b && vite build
npm run lint         # oxlint
```
First backend run is slow (BGE model load); subsequent runs ~6–12s. Expected baseline: **11 tests pass**, **build exit 0**.

---

## 8. Development workflow (per slice)

```
Plan → Implement → Test → Review → Merge → Verify → Push
```
One vertical slice at a time. Define the API contract first, build backend + frontend in parallel on their branches, merge into `main` in the main worktree, run the full gate, push only when green.

## 9. Working discipline (how every session should behave)

**Do not jump straight to writing code.** For any non-trivial task:
1. **Inspect** the relevant code first — never assume.
2. **Design** the approach and identify dependencies/contract impact.
3. **Explain** the plan briefly before large changes.
4. **Implement** in the correct layer (services do logic, agents orchestrate, routers expose).
5. **Test** — add/extend tests; run them.
6. **Verify** — tests + build (+ integration for QA) green.
7. **Summarize** what changed, honestly (including anything skipped or still red).

Stay in your lane: backend never edits frontend; frontend never changes backend algorithms; QA fixes only verified bugs.

## 10. Knowledge base map

- **Skills** (`.claude/skills/`): `architecture`, `rag`, `langgraph`, `backend-dev`, `frontend-dev`, `testing`, `qa-verify`, `git-workflow`.
- **Commands** (`.claude/commands/`): `/verify-backend`, `/verify-frontend`, `/verify-all`, `/build`, `/merge`, `/release`.
- **Project docs** (`.claude/project/`): `roadmap.md`, `api_contract.md` (the coordination surface), `decisions.md` (ADR log), `session_log.md` (cross-session memory), `interview_notes.md` (design rationale Q&A).

Update `session_log.md` after meaningful work and `decisions.md` when you make a significant choice, so the next session inherits full context.

## 11. Known non-blocking debt

Pydantic v2 `class Config` → `ConfigDict` (schemas); outdated Anthropic model id in `evaluation_service.get_llm()`; `datetime.utcnow()` deprecation. Warnings only — do not block work; fix opportunistically.
