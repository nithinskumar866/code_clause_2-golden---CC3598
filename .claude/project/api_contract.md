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
authenticity : AuthenticityAssessment | null
```
`AuthenticityAssessment` (keyword-stuffing / over-claim detection, deterministic):
```
credibility_score : int (0-100)          # corroboration expressed 0-100
keyword_stuffing_risk : "Low"|"Medium"|"High"
over_claimed_skills : string[]           # listed but never demonstrated
corroboration_ratio : float (0.0-1.0)    # demonstrated / claimed
explanation : string
```
Note: `quality_score` is **no longer a constant** — it is now derived deterministically from
evidence corroboration + depth (was hardcoded `85`). `authenticity` is nullable so reports
persisted before this field still validate (e.g. on export).
`RequirementFit`: `{ requirement, category, status ("Matched"|"Partial"|"Missing"), matched_evidence, explanation, limitations, confidence(0-100), importance ("must"|"nice"|null), weight (float|null) }`
`importance`/`weight` are derived from the JD wording (must-have vs nice-to-have); `coverage_score` is importance-weighted so missing must-haves cost more. Nullable for backward-compat.
`LearningRoadmapItem`: `{ skill, estimated_time, reason }`

## Known contract gap (frontend TODO)
`frontend/src/types/index.ts` currently defines `ResumeMetadata`, `JobDescriptionMetadata`, `SystemStatusData` — but **no `HiringReport` / `RequirementFit` / `LearningRoadmapItem` types**. The Analysis page needs these added to consume `/analysis/evaluate`. This is the first contract item to align on.

## Change protocol
1. Backend edits schema + endpoint, keeps tests green, records it in `decisions.md` and appends a note in `session_log.md`.
2. Frontend mirrors into `types/index.ts` and wires the UI.
3. QA verifies the round trip end-to-end.
