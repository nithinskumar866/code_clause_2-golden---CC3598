---
name: testing
description: Testing conventions for the AI Hiring Platform — pytest patterns, fixtures, regression, and build verification. Use when writing or running tests, or before signaling a slice ready.
---

# Testing

## Backend (pytest)
Run (reuse the main worktree venv):
```
cd ai_hiring_platform/backend
"E:/projects from desktops/codeclause2/ai_hiring_platform/backend/.venv/Scripts/python.exe" -m pytest -q
```
Baseline: **11 pass**. First run loads BGE (~slow); later runs ~6–16s.

Suite map:
- `test_rag.py` — ingestion, embedding, retrieval evidence shape.
- `test_hiring_decision.py` — evaluation engine over mock evidence (category, status, scores, roadmap).
- `test_workflows.py` — LangGraph compilation + node outputs.
- `test_routes.py` — API endpoints (upload, etc.).
- `conftest.py` — fixtures (app/client, temp db).

## Writing tests
- Add/extend tests with every behavior change; a slice isn't ready until green.
- Test **algorithms generically** — feed synthetic evidence, assert structural properties (ranges 0–100, status ∈ {Matched,Partial,Missing}, category non-empty), not hardcoded skill outcomes beyond what the contract guarantees.
- Deterministic engine path (no API key) must stay tested; don't require live LLM in unit tests.

## Frontend (build = the gate)
```
cd ai_hiring_platform/frontend
npm run build     # tsc -b && vite build; type errors fail the build — fix them
npm run lint      # oxlint
```

## Regression / integration (QA, main worktree)
After a merge: full pytest + build + end-to-end (launch API, drive resume→job→evaluate, confirm a valid explainable report). Only an all-green state becomes a checkpoint.
