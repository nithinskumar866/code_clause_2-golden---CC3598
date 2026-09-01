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
overall_score : float (0-100, ONE DECIMAL) -- the Match Score
match_score : MatchScore | null           -- its nine-parameter decomposition
coverage_score, experience_score, project_score,
confidence_score, quality_score : int (0-100)   -- evidence sub-scores, not the headline
summary : string
requirements : RequirementFit[]
strengths, weaknesses, skill_relationships, missing_skills, interview_questions : string[]
learning_roadmap : LearningRoadmapItem[]
recruiter_recommendation : string
rejection_email : string | null
authenticity : AuthenticityAssessment | null
candidate_profile : CandidateProfile | null
```
`MatchScore` / `MatchParameter` (deterministic; see `docs/Match_Score.md`):
```
MatchScore  : { score : float, band : string, parameters : MatchParameter[] }
MatchParameter : {
  key          : "skill"|"experience"|"technology"|"designation"|"industry"
               |"education"|"location"|"availability"|"freshness"
  label        : string
  weight       : float   -- 0.20, 0.15, 0.14, 0.14, 0.09, 0.08, 0.08, 0.07, 0.05
  score        : float   -- 0-100, one decimal
  contribution : float   -- score x weight
  basis        : string  -- recruiter-readable reason
  neutral      : bool    -- the JD never stated this requirement
}
```
Bands: >=85 Excellent fit · >=70 Strong fit · >=50 Moderate fit · else Weak fit.
`match_score` is null on analyses produced before this contract existed.

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


## Prompt Lab (2026-08-21) — `/api/v1/prompt-lab/*`
Mirrored in `frontend/src/types/index.ts` and consumed by `api/promptLab.ts`.

| Method | Path | Body → Data |
|---|---|---|
| GET | `/rules` | → `RuleInfo[]` `{ id, title, description }` |
| GET | `/starter` | → `{ name, prompt, cases }` — the prompt in the field + the turns exercising each clause |
| POST | `/run` | `{ variants[{label,prompt}], cases[], temperature, rules?, suite_id?, persist }` → `RunResult` |
| POST | `/score` | `{ prompt, case, answer, rules? }` → `ScoreResult` (deterministic; needs no LLM) |
| GET/POST | `/suites` · POST/PUT/DELETE `/suites/{id}` | `Suite` `{ id, name, description, prompt, cases, created_at, updated_at }` |
| GET | `/runs?suite_id=&limit=` · `/runs/{id}` | `RunSummary[]` · `RunDetail` (adds `results`) |

`PromptCase`: `{ id?, name?, question, context, history[{role,content}], expectations }`.
`CaseExpectations`: `{ max_lines, max_chars, link_allowed, link_required, years_known, expected_years, jobs_requested, list_allowed, skills_requested, must_contain[], must_not_contain[] }` — properties of the TURN, never of the prompt, so one suite grades any rewrite.
`RuleVerdict.status` is `pass | fail | na`; `score` is over applicable rules only and is `null` when none applied — render it as a dash, never as 0% or 100%.
`RunResult`: `{ variants[VariantResult], deltas[RuleDelta], llm_available, model }`. With no LLM configured the call still succeeds with `llm_available:false` and empty answers — the lab reports a missing model rather than faking one.

## Change protocol
1. Backend edits schema + endpoint, keeps tests green, records it in `decisions.md` and appends a note in `session_log.md`.
2. Frontend mirrors into `types/index.ts` and wires the UI.
3. QA verifies the round trip end-to-end.

## `/api/v1/company-chat` — company knowledge base (optional module)

Backend owns `backend/app/schemas/company_chat.py`; frontend mirrors it in
`src/types/index.ts` (`CompanyResult`, `CompanyChatResponse`, `CompanyStoreStatus`).
Kept separate from the chat contract on purpose — a company and a candidate share no
fields, and one union type would force every UI branch to check which half it holds.

| Method | Path | Purpose |
|---|---|---|
| POST | `/company-chat/query` | Answer one question → `CompanyChatResponse` |
| POST | `/company-chat/stream` | Same, streaming real pipeline stages as SSE |
| POST | `/company-chat/reset` | Forget a company conversation's follow-up memory |
| GET | `/company-chat/status` | `CompanyStoreStatus` — configured / reachable / populated |
| GET | `/company-chat/companies` | `[{company_id, company_name}]` |
| POST | `/company-chat/refresh` | Re-read the name index after a load, no restart |

`CompanyChatResponse.mode` is `company` (the question named one), `open` (pool-wide
search) or `none` (refused before retrieval). `engine` is `llm` or `deterministic` —
the FACTS are identical either way; only the prose differs, because the model is never
allowed to supply a fact.

Ingest is a CLI script, not an endpoint: `python -m scripts.load_companies <file.xlsx>`
drops and rebuilds the collection from the sheet.
