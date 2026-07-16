# API Contract

The single coordination surface between Backend (`dev/backend`) and Frontend (`dev/frontend`). Backend owns it (schemas + routers); Frontend mirrors it into `src/types/index.ts`. Backend lands a change here **first**; Frontend builds against the agreed shape (mock until merged). Prefer additive changes.

Base URL: `/api/v1`. Every response is wrapped in the envelope below.

## Envelope — `ApiResponse<T>` (`schemas/response.py`)
```
{ "success": boolean, "message": string, "data": T | null }
```

## Endpoints
| Method | Path | Query/Body | `data` payload |
|---|---|---|---|
| GET  | `/health` | — | `dict` (service status) |
| POST | `/resume/upload` | multipart file (PDF/DOCX) | `ResumeResponse` |
| GET  | `/resume` | — | `ResumeResponse[]` |
| POST | `/job/upload` | multipart file (PDF/DOCX) | `JobDescriptionResponse` |
| GET  | `/job` | — | `JobDescriptionResponse[]` |
| POST | `/analysis/evaluate` | `?resume_id=&jd_id=` | `{ analysis_id, status, report: HiringReport }` |

`/analysis/evaluate` runs the full LangGraph pipeline (ingest → embed → FAISS → retrieve evidence → reason) and returns the explainable report.

## Core shapes (backend `schemas/`)
`ResumeResponse` / `JobDescriptionResponse`: `{ id, filename, status, upload_time/created_at }`

`HiringReport` (`schemas/analysis.py`):
```
overall_score, coverage_score, experience_score, project_score,
confidence_score, quality_score : int (0-100)
summary : string
requirements : RequirementFit[]
strengths, weaknesses, skill_relationships, missing_skills, interview_questions : string[]
learning_roadmap : LearningRoadmapItem[]
recruiter_recommendation : string
rejection_email : string | null
```
`RequirementFit`: `{ requirement, category, status ("Matched"|"Partial"|"Missing"), matched_evidence, explanation, limitations, confidence(0-100) }`
`LearningRoadmapItem`: `{ skill, estimated_time, reason }`

## Known contract gap (frontend TODO)
`frontend/src/types/index.ts` currently defines `ResumeMetadata`, `JobDescriptionMetadata`, `SystemStatusData` — but **no `HiringReport` / `RequirementFit` / `LearningRoadmapItem` types**. The Analysis page needs these added to consume `/analysis/evaluate`. This is the first contract item to align on.

## Change protocol
1. Backend edits schema + endpoint, keeps tests green, records it in `decisions.md` and appends a note in `session_log.md`.
2. Frontend mirrors into `types/index.ts` and wires the UI.
3. QA verifies the round trip end-to-end.
