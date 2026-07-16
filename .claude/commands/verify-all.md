Run the full verification gate for the AI Hiring Platform (use in the main/QA worktree before pushing a checkpoint).

1. Backend tests:
```
cd ai_hiring_platform/backend
".venv/Scripts/python.exe" -m pytest -q
```
2. Frontend build:
```
cd ../frontend
npm run build
```
3. Integration: start the FastAPI app, exercise the health/resume/job/analysis endpoints, and confirm the LangGraph pipeline returns a structured, explainable report end-to-end.

Report a single verdict: GREEN (all pass) or RED (with the specific failure). Only a GREEN result may be pushed to `origin/main`. Never push a red state, and never add Claude attribution to commits.
