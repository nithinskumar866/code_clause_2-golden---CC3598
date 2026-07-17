# Frontend Handoff — Authenticity + Requirement Priority (must/nice)

Backend landed new, recruiter-facing intelligence on the existing hiring report. The
frontend already renders `AnalysisReport` (HiringReportView / ScoreOverview); this
handoff adds **two additive, nullable fields** to surface. No breaking changes.

Example real response: [`frontend_handoff_example_response.json`](./frontend_handoff_example_response.json)
(the exact `POST /api/v1/analysis/evaluate` → `data` shape, produced by the engine).

## 1. Contract additions (mirror into `src/types/index.ts`)

### `RequirementFit` — gains priority
```ts
export interface RequirementFit {
  // ...existing fields...
  importance?: 'must' | 'nice' | null;   // JD marks it required vs nice-to-have
  weight?: number | null;                // scoring weight (must=1.0, nice=0.4)
}
```

### New `AuthenticityAssessment` + `AnalysisReport.authenticity`
```ts
export interface AuthenticityAssessment {
  credibility_score: number;                        // 0-100 (corroboration as %)
  keyword_stuffing_risk: 'Low' | 'Medium' | 'High';
  over_claimed_skills: string[];                    // listed but never demonstrated
  corroboration_ratio: number;                      // 0.0-1.0
  explanation: string;                              // recruiter-readable rationale
}

export interface AnalysisReport {
  // ...existing fields...
  authenticity?: AuthenticityAssessment | null;
  candidate_profile?: CandidateProfile | null;
}
```

### New `CandidateProfile` (F3 — identity + seniority)
```ts
export interface CandidateProfile {
  name: string | null;
  title: string | null;
  total_years: number | null;
  seniority_level: 'Junior' | 'Mid' | 'Senior' | 'Lead' | null;
  required_years: number | null;                    // from the JD, if stated
  seniority_fit: 'Below' | 'Meets' | 'Exceeds' | 'Unknown' | null;
  explanation: string;
}
```

> **Nullable is deliberate.** Analyses created before this change (and any legacy stored
> report) have `authenticity: null` and `importance/weight: null`. Render gracefully when absent.

## 2. Suggested UI (surface the new signals)

**Report header / ScoreOverview**
- **Credibility badge** from `authenticity.credibility_score` + a **keyword-stuffing risk chip**
  (`Low`→green, `Medium`→amber, `High`→red). Put `authenticity.explanation` in a tooltip.
- Note: `quality_score` is now a *real* computed value (was previously always 85) — no UI change
  needed, but it will now vary per candidate.

**Requirement list (RequirementFit rows)**
- Show a **must / nice badge** per requirement (`importance`). Consider grouping/sorting
  must-haves first — they drive the (now importance-weighted) `coverage_score`.
- Visually flag any requirement whose name is in `authenticity.over_claimed_skills`
  (e.g. a small "listed, not demonstrated" marker) — these are the keyword-stuffing hits.

**Weaknesses**
- Missing must-haves already arrive first in `weaknesses` and are prefixed
  `"Missing must-have requirement: ..."`; a High stuffing risk adds a
  `"Possible keyword stuffing: ..."` line. No special handling required — just render the list.

**Candidate profile (the existing `CandidateProfile` page / report header)**
- Show `name`, `title`, and `total_years`. Render a **seniority-fit chip** from
  `seniority_fit` (`Below`→red, `Meets`→green, `Exceeds`→blue, `Unknown`→grey) with
  `explanation` as subtext (e.g. "6 yrs (Senior) vs 5+ required — meets the bar").
- All fields are nullable — hide any that are `null`.

## 3. Not in this handoff
- LLM provider/model is backend config only (`LLM_PROVIDER`/`LLM_MODEL`) — nothing to mirror.
- Held for later (backend): better chunking, hybrid retrieval, multi-candidate ranking.
