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
Shared chunk store          (services/ai/embedding_store.py — parsed ONCE, model-independent)
        │
        ▼
Embedding per model         (bge 384d · mxbai 1024d · gpu 768d — explicit, incremental)
        │
        ▼
One FAISS index per model   (vectors-<model>.faiss over the SAME chunk set)
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

### The Recruiter Assistant is NOT a third agent

The chatbot (`services/ai/chat_service.py`) is a **service**, not a runtime agent. It
neither joins the LangGraph workflow nor makes hiring decisions — it retrieves evidence
across the pool and explains matches. Rule "exactly two runtime agents" is intact: do
not promote it to an agent, and do not wire it into `hiring_workflow.py`.

It reuses the existing stack (embeddings, BM25, hybrid RRF, profile extraction, the LLM
factory) and adds a pool-wide index rather than duplicating retrieval logic:

| Concern | Module |
|---|---|
| **Shared vector store — parsing, chunking and vectors for the whole platform** | `services/ai/embedding_store.py` |
| One model's query-ready view of that store (BM25, profiles, `Corpus`) | `services/ai/corpus_index_service.py` |
| **Corpus-derived lexicons — what the pool says its words mean** | `services/ai/chat_lexicon.py` |
| Typed query understanding (intent + facets) | `services/ai/chat_query_understanding.py` |
| LLM span-labelling assist, re-validated against the corpus | `services/ai/chat_llm_parser.py` |
| Staged pool search, per-person fact lookup, scoring | `services/ai/chat_retrieval_service.py` |
| Input/output guardrails (scope, injection, secrets, fairness, grounding) | `services/ai/chat_guardrails.py` |
| Orchestration + conversation memory | `services/ai/chat_service.py` |

Invariant: **the LLM never supplies a fact.** Every name, score, year, contact and quote
originates in deterministic retrieval; `ground_candidates` deletes anything not tied to
a real indexed resume with real evidence.

**Second invariant: the LLM never supplies a *requirement* either.** A hallucinated
constraint silently changes the question before retrieval runs, which is just as
damaging as a hallucinated answer. `chat_llm_parser` therefore only asks the model to
mark up spans of the recruiter's own sentence; every span must appear verbatim in the
message *and* resolve through `chat_lexicon` before it can affect the search. With no
LLM configured, or on any parse failure, the deterministic reading stands.

**The pool decides what its own words mean.** Whether `now` is a technology or ordinary
English is not answerable in the abstract — only against this corpus. `chat_lexicon`
measures, over the indexed text, how often a term is written in plain lowercase and how
widely it is spread; a term that reads as English in this pool can never become a search
requirement. Places and people are harvested the same way, so the corpus is its own
gazetteer and its own name index. No maintained keyword list, Golden Rule 4 intact.

**Uploading and embedding are separate decisions.** At 300+ CVs the slow model costs
~40 minutes for a full pass, so the platform keeps a **working set** — the resumes
actually parsed into `documents.db`, and therefore the only ones that can be embedded or
searched. The Documents screen (`pages/Documents`) is where a recruiter chooses who is in
it, with photo-gallery selection gestures (`hooks/useGallerySelection`). Uploads under
`AUTO_INDEX_MAX_BATCH` still index themselves; a bigger batch is stored and waits to be
selected. Startup catches models up on the existing working set but never grows it.
Removal comes in two flavours that must not be merged: *remove from index* (reversible,
keeps the file) and *delete permanently*.

**Four answer shapes, not one.** A recruiter question is a *search*, a *fact* about one
person, an *explanation* of the previous answer, or a *question back*. Treating every
question as a pool search is what made the assistant refuse its own follow-ups, answer a
two-constraint question by dropping one constraint, and re-score a candidate while
defending that candidate's score. An explanation is answered from the stored previous
result and never re-runs retrieval.

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

**Embedding models (three, interchangeable).** `bge` (BAAI/bge-small-en-v1.5, 384d, local) · `mxbai` (mixedbread-ai/mxbai-embed-large-v1, 1024d, local) · `gpu` (nomic-embed-text, 768d, remote endpoint). All three index the SAME chunk set from `embedding_store`, and every section — ranking, analysis, chatbot — reads that one store. Each model carries its **own** cosine floor, calibrated to equal selectivity on real resumes: sharing a threshold would reward whichever model scores higher in absolute terms rather than the one that ranks better. Indexing is **explicit and per model** (`POST /embeddings/index`), never implicit on upload — the implicit path is what let 300 resumes enter one model's index and no other, silently. `/embeddings/coverage` reports the gap; the Model Lab page compares models over the resumes they have all indexed.

**Stack:** FAISS · SQLite · PDF/DOCX. Only LLM reasoning is optional/remote (OpenAI or Anthropic via llama-index; deterministic mock fallback). Everything else is local.

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
Baseline: **348 tests pass**, **build exit 0**. First backend run is slow (BGE load).

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
