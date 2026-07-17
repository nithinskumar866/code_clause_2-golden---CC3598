# Session Log

Running record of what each session accomplished, so context survives across cleared sessions. Newest first. Keep entries short: date · who (backend/frontend/qa/orchestrator) · what changed · verify result.

## 2026-07-18 · Backend (main) — F1: provider-agnostic LLM + grounded reasoning
New `llm_service.py` (config-driven provider/model factory: openai/anthropic/google-gemini; installed google-genai integration; reads GOOGLE_API_KEY or GEMINI_API_KEY). Rebuilt evaluation LLM path into grounded, testable helpers (`_build_reasoning_prompt`/`_extract_json`/`_finalize_llm_report`) — LLM sees only retrieved evidence + must/nice importance; deterministic algorithms still own coverage/quality/authenticity/overall. Removed stale hardcoded model id. 8 new tests (llm_service + reasoning via fake LLM). Contract change: **no** (config + internal). Verify: pytest **115 pass**. Not committed. Enable live: set `LLM_PROVIDER=google` + `GEMINI_API_KEY` in backend/.env. **F3 (candidate profile/seniority) still pending.** Held for later: chunking, hybrid search, multi-candidate ranking.

## 2026-07-18 · Backend (main) — F2: requirement prioritization + weighted coverage
Added deterministic `classify_priorities` (must-have vs nice-to-have from JD wording); Agent 1 attaches importance/weight to evidence; `coverage_score` now importance-weighted in both paths; `RequirementFit` gained optional importance/weight; missing must-haves prioritized in weaknesses. 6 new tests. Contract change: **yes** (additive, nullable). Verify: pytest **107 pass**. Not committed. Next: F1 (grounded LLM reasoning core, folds in F3 profile/seniority) — needs `ANTHROPIC_API_KEY` for live runs; deterministic fallback keeps working meanwhile.

## 2026-07-18 · Backend (main) — RAG generalization (JD extraction) + audit
Audited the RAG end-to-end. Verified (probe) that retrieval is genuine cosine (BGE `normalize=True` → unit vectors → FAISS `IndexFlatIP`) — no normalization bug. Rebuilt `jd_requirement_extractor` from taxonomy-only keyword lookup into a two-layer deterministic extractor (taxonomy precision + grammar-cue generalization) so non-taxonomy skills / soft skills / domain / seniority are captured for any JD. Replaced the hardcoded `tech_patterns` regex in `retrieval_service` with a generic morphology+taxonomy tech-specificity signal. Added 2 RAG tests. Contract change: **no** (internal). Verify: pytest **101 pass**. Not committed yet.

## 2026-07-18 · Backend (main) — authenticity & real quality score
New deterministic `services/ai/authenticity_service.py` (keyword-stuffing / over-claim detection from evidence sections). Added `AuthenticityAssessment` schema + optional `authenticity` field on `HiringReport`; wired into `evaluation_service` on both LLM and mock paths; replaced hardcoded `quality_score=85` with a corroboration+depth blend; High risk surfaces in recruiter weaknesses. Added config thresholds (`STUFFING_*_FRACTION`, `QUALITY_WEIGHT_*`) and `tests/test_authenticity.py` (5 tests). Contract change: **yes** (additive, nullable). Verify: pytest **99 pass**, build not run (backend-only). Not committed yet.

## 2026-07-17 · Orchestrator (main) — knowledge base v2
Expanded `CLAUDE.md` into the full "permanent brain": System Overview, end-to-end runtime flow, explicit Agent 1 / Agent 2 contracts (+ "evidence intelligence" note, no class rename), Algorithms-vs-LLM split, 10 Golden Rules, Project Principles, truthful repo tree, feature recipe, debug playbook, Future Vision pointer. Added `.claude/project/future_vision.md`; extended `interview_notes.md` (cosine similarity, deterministic vs LLM, single-LLM). No runtime code touched.

## 2026-07-17 · Orchestrator (main)
Set up parallel development. Built the permanent knowledge base (`CLAUDE.md`, `.claude/skills`, `.claude/commands`, `.claude/project`), committed to `main`. Created worktrees `cc2-backend` (`dev/backend`) and `cc2-frontend` (`dev/frontend`); configured envs (backend reuses main venv; frontend `npm ci`). Verified: backend **11 pass** from the worktree via reused venv, frontend **build ✓**. `main` is ahead of `origin/main` by the knowledge-base commits (not yet pushed).

## 2026-07-16 · Orchestrator (main)
Pushed the stable baseline. Resolved an in-progress rebase, split hiring-platform vs legacy into two commits, stripped Claude attribution, force-pushed clean history. Generalized skill semantics (removed hardcoded category dict + transfer if-chains) and removed placeholder agents. Baseline: **11 tests pass**, **build ✓**.

---
### Template for new entries
```
## <date> · <backend|frontend|qa|orchestrator>
<slice/feature>. <files touched at a high level>. Contract change: <yes/no + note>.
Verify: pytest <n pass>, build <ok/fail>, integration <ok/fail>.
```
