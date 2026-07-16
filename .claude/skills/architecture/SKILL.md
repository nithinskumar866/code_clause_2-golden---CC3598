---
name: architecture
description: System architecture of the AI Hiring Platform — the two-agent Agentic RAG design, LangGraph flow, folder responsibilities, and where each concern lives. Use when reasoning about where code belongs or how data flows.
---

# Architecture

## The system in one line
Explainable **Agentic RAG** hiring intelligence: retrieve evidence, then reason over it, and explain every conclusion.

## Runtime flow (exactly two agents)
```
Resume + JD
    │
    ▼
Candidate Intelligence Agent   ── parse → structure → embed (BGE) → FAISS → extract JD requirements → retrieve per-requirement evidence
    │
    ▼
Structured Evidence (state)
    │
    ▼
Hiring Decision Agent          ── reasons ONLY over evidence → scores, gaps, roadmap, questions, recommendation
    │
    ▼
Explainable Report
```
**Why two agents:** it mirrors real hiring — one party gathers evidence (investigator), another decides (hiring manager). This separation is what keeps every decision auditable and prevents the "black box." Do not add a third runtime agent.

## Layered responsibilities (modular monolith)
| Folder | Responsibility | Never does |
|---|---|---|
| `agents/` | Orchestrate services into an agent's job | Contain core algorithms |
| `services/ai/` | Implement logic (embed, retrieve, evaluate, classify) | Expose HTTP / hold DB rows |
| `workflows/` | The LangGraph state machine wiring the two agents | Business logic |
| `api/v1/routers/` | Expose HTTP endpoints, validate, persist | Reasoning/algorithms |
| `schemas/` | Pydantic request/response shapes — the **API contract** | Side effects |
| `models/` | SQLAlchemy tables | Business logic |
| `core/` | config, constants, db, logging, exceptions | Feature logic |
| `frontend/` | React UI consuming the API | Change backend algorithms |

## Data & storage
BGE embeddings · FAISS (per-resume index) · SQLite (resume/JD/analysis rows) · report JSON in `storage/reports/`. Everything local except optional LLM reasoning.

## Where to add things
- New retrieval/scoring behavior → a **service** in `services/ai/`, called by an agent. Never inline into a router.
- New reasoning output field → extend `schemas/analysis.py` (HiringReport) + the evaluation service + tests, and note the contract change for Frontend.
- New endpoint → a **router**, delegating to a service/agent. Keep routers thin.
