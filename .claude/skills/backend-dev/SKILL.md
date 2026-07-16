---
name: backend-dev
description: Backend development conventions for the AI Hiring Platform (LangGraph, RAG, FAISS, agents, FastAPI, evaluation). Use when editing anything under ai_hiring_platform/backend.
---

# Backend Development

You are the **Backend** session (worktree `../cc2-backend`, branch `dev/backend`). Scope: `ai_hiring_platform/backend/**` only. **Never edit `frontend/**`.**

## Architecture you must preserve
- Exactly **two runtime agents**: `agents/candidate_intelligence` (retrieval/evidence) and `agents/hiring_decision` (reasoning). Do not add runtime agents.
- LangGraph in `workflows/hiring_workflow.py` stays sequential: `START → candidate_intelligence → hiring_decision → END`. No branching/loops.
- Layering: **services** implement logic, **agents** orchestrate services, **routers** expose APIs, **models** hold data, **core/config + constants** hold configuration.

## Reasoning contract
`Input → RAG retrieves evidence → LLM reasons over evidence → algorithm validates/explains → structured output.`
- Agent 1 returns structured evidence; Agent 2 reasons only from it and never re-reads raw files.
- Do not point the LLM at raw documents. Prefer LLM reasoning over deterministic business logic; use algorithms to support/validate.

## Hard rules
- **Algorithms, not examples.** Never `if skill == "python"`. Iterate over extracted requirements/entities; generalize to any resume + any JD.
- Never hardcode skills/technologies/relationships. Prompt examples are illustrative only.
- Reuse the generic primitives in `services/ai/skill_semantics_service.py` (nearest-centroid category classification + embedding transfer) instead of writing lookup tables.

## API contract (coordination with Frontend)
The contract lives in `schemas/*.py` + `response.py` + `routers/*`. When you change a request/response shape, **land the schema first** and note it so Frontend can mirror it into `frontend/src/types/index.ts`. Additive changes preferred; avoid breaking existing fields.

## Verify before signaling a slice ready
```
cd ai_hiring_platform/backend
"E:/projects from desktops/codeclause2/ai_hiring_platform/backend/.venv/Scripts/python.exe" -m pytest -q
```
Must be green (baseline: 11 pass). Add/extend tests for new behavior. First run is slow (BGE load).

## Git
Commit to `dev/backend` locally. No Claude attribution in commit messages. Do not push — the main/QA session pushes verified checkpoints.
