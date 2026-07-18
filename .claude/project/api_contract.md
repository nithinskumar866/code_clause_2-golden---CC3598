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
candidate_profile : CandidateProfile | null
```
`CandidateProfile` (deterministic identity + seniority fit):
```
name, title : string | null
total_years, required_years : float | null
seniority_level : "Junior"|"Mid"|"Senior"|"Lead" | null
seniority_fit : "Below"|"Meets"|"Exceeds"|"Unknown" | null
explanation : string
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

## Contract gap — CLOSED (2026-07-18)
`frontend/src/types/index.ts` now mirrors the full report contract: `RequirementFit` (incl. `importance`/`weight`), `LearningRoadmapItem`, `AuthenticityAssessment`, `CandidateProfile`, and `AnalysisReport.authenticity`/`candidate_profile`. The Analysis + CandidateProfile pages render authenticity, seniority/profile, and must/nice badges. The backend base URL is centralized in `frontend/src/api/client.ts` (`VITE_API_BASE`-aware).

## Still backend-only (no UI yet — see roadmap)
These endpoints exist, are tested, and return the shapes above but have no frontend surface: `/dashboard/*` aggregates, all `/analytics/*`, recruiter `/notes`, workflow `/status`, and server-side `/export/pdf`. Multi-candidate ranking is not implemented at all.

## Change protocol
1. Backend edits schema + endpoint, keeps tests green, records it in `decisions.md` and appends a note in `session_log.md`.
2. Frontend mirrors into `types/index.ts` and wires the UI.
3. QA verifies the round trip end-to-end.
