---
name: qa-verify
description: QA / integration verification for the AI Hiring Platform — run full backend tests, frontend build, launch the app, integration + regression checks before merging or pushing. Use in the main worktree when validating a vertical slice.
---

# QA / Integration Verification

You are the **Integration/QA** role in the **main worktree** (branch `main`). You verify a vertical slice before it becomes a checkpoint. **You fix only verified bugs — you build no features.**

## When to run
On-demand: when Backend and/or Frontend signal a slice is ready and merged into `main` locally.

## Merge order (in main worktree)
```
git merge --no-ff dev/backend      # if backend slice ready
git merge --no-ff dev/frontend     # if frontend slice ready
```
Backend/frontend edit disjoint trees, so merges are normally conflict-free.

## Full verification gate
```
# Backend
cd ai_hiring_platform/backend
".venv/Scripts/python.exe" -m pytest -q          # baseline 11 pass

# Frontend
cd ../frontend
npm run build                                     # tsc + vite, exit 0

# Integration (end-to-end)
# start the API, exercise health/resume/job/analysis endpoints, confirm the
# LangGraph pipeline returns a structured, explainable report; run the app UI
# against it and confirm the dashboard renders real evidence.
```

## Regression
Re-run the full backend suite and frontend build after every merge. A slice is "green" only when: pytest passes, build passes, and the end-to-end pipeline produces a valid explainable report.

## Push (only when green)
```
git push origin main
```
Push **only** verified checkpoints. Fast-forward only. No Claude attribution in any commit. After a checkpoint, sync dev branches forward (`git merge main` in each worktree).

## If a bug is found
Report it to the responsible session (backend/frontend) with a concrete repro. Do not push a red state.
