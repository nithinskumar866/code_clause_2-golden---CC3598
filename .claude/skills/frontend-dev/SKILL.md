---
name: frontend-dev
description: Frontend development conventions for the AI Hiring Platform (React, Vite, TypeScript, Tailwind, dashboard, charts, API integration). Use when editing anything under ai_hiring_platform/frontend.
---

# Frontend Development

You are the **Frontend** session (worktree `../cc2-frontend`, branch `dev/frontend`). Scope: `ai_hiring_platform/frontend/**` only. **Treat the backend as an API — never change backend algorithms or schemas.**

## Stack & structure
- React + Vite + TypeScript + Tailwind CSS, Lucide icons.
- Pages: `src/pages/{Dashboard, Analysis, Job, Resume, SystemStatus}`. Shared API types: `src/types/index.ts`.
- Keep components focused and reusable; match existing naming/styling idiom.

## API contract (coordination with Backend)
- `src/types/index.ts` **mirrors** the backend `schemas/*.py` + `response.py`. Keep it in sync with what the backend actually returns.
- If a feature needs a backend field/endpoint that doesn't exist yet, build against the **agreed contract with a local mock**; wire to the real endpoint once Backend merges it. Do not block waiting.
- The report shape is explainable by design: surface Requirement → Evidence → Reasoning → Confidence → Decision. Visualizations must reflect real evidence, never fabricate.

## Verify before signaling a slice ready
```
cd ai_hiring_platform/frontend
npm run build      # tsc -b && vite build  — must exit 0
npm run lint       # oxlint
```
Type errors fail the build; fix them, don't suppress. Baseline: build succeeds.

## Git
Commit to `dev/frontend` locally. No Claude attribution in commit messages. Do not push — the main/QA session pushes verified checkpoints.
