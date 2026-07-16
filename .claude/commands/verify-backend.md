Run the backend test suite for the AI Hiring Platform and report the result.

```
cd ai_hiring_platform/backend
"E:/projects from desktops/codeclause2/ai_hiring_platform/backend/.venv/Scripts/python.exe" -m pytest -q
```

Report pass/fail counts. If anything fails, show the failing test and the assertion, and diagnose the root cause. Baseline is 11 passing. Do not treat Pydantic/datetime deprecation warnings as failures.
