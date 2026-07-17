    # Development Roadmap

    Goal: finish an interview-grade, explainable AI Hiring Platform in ~2 days using parallel Backend + Frontend sessions with on-demand QA.

    ## Dependency framework (how to parallelize any feature)
    - **Parallel-safe** (assign to different sessions): backend algorithm/service/endpoint work ∥ frontend UI/visualization work — disjoint file trees.
    - **Serialized only at the contract**: a frontend feature that needs a *new* backend field/endpoint depends on the schema. Mitigate with **contract-first** — backend lands the schema, frontend mocks it — so both still start immediately.
    - **QA depends on both** → runs at merge time in the main worktree.
    - Blocking rule: if task B needs task A's contract, A publishes the schema first; B never waits on a push.

    ## Current state (baseline)
    Sprints 1–5 complete: foundation, RAG engine, hiring-decision engine, LangGraph workflow, retrieval refinement + generic skill semantics. Baseline is green (11 tests, build ✓). See `session_log.md`.

    ## Completed slices
    - **F3 — Candidate profile & seniority** (2026-07-18, backend) — deterministic `profile_service` (name/title/years + seniority-fit vs JD required years); `HiringReport.candidate_profile`. 120 tests pass. Intelligent core (F1+F2+F3) complete. **Next held items: chunking → hybrid retrieval → multi-candidate ranking.**
    - **F1 — Provider-agnostic LLM + grounded reasoning core** (2026-07-18, backend) — config-driven `llm_service` (openai/anthropic/google-gemini via `LLM_PROVIDER`+`LLM_MODEL`); grounded evidence-only reasoning prompt with must/nice awareness; deterministic algorithms still own the numbers; robust JSON parsing + fallback. 115 tests pass. **Pending: F3** (candidate profile/seniority — fold into synthesis). Enable live via `.env` (Gemini). Held: chunking, hybrid search, multi-candidate ranking.
    - **F2 — Requirement prioritization + weighted coverage** (2026-07-18, backend) — deterministic must-have vs nice-to-have from JD wording; importance-weighted `coverage_score`; `RequirementFit.importance/weight`; missing must-haves flagged first. 107 tests pass. **Next: F1** (evidence-grounded LLM reasoning core, folds in F3 profile/seniority). Held for later: chunking, hybrid search, multi-candidate ranking.
    - **Generalized JD requirement extraction** (2026-07-18, backend) — extractor no longer keyword-only; two deterministic layers (taxonomy + grammar cues) capture unknown tools, soft skills, domain, and seniority for any JD. Also removed the hardcoded tech-specificity regex in retrieval. RAG audited: retrieval confirmed genuine cosine. 101 tests pass. *(Was candidate slice "Generalized JD requirements".)*
    - **Authenticity & real quality score** (2026-07-18, backend) — deterministic keyword-stuffing / over-claim detection (`authenticity_service`); `AuthenticityAssessment` on `HiringReport`; `quality_score` now derived (was hardcoded 85). 99 tests pass. *Frontend follow-up:* mirror `AuthenticityAssessment` into `types/index.ts` and render the credibility block + stuffing badge on the Analysis page.

    ## Candidate slices (to be prioritized with the user — not yet started)
    Each is one vertical slice: design → backend → contract → frontend → tests → verify → merge.
    1. **Report typing + Analysis UI** — add `HiringReport`/`RequirementFit`/`LearningRoadmapItem` to `types/index.ts`; render the explainable report (requirement table, evidence, confidence). *(Contract gap already identified — natural first slice. Backend contract already exists, so this is mostly frontend.)*
    2. **Evidence drill-down** — per-requirement evidence provenance (chunk, section, score) surfaced in the UI. Backend evidence already carries it; mostly frontend + minor schema exposure.
    3. **Scoring transparency** — visualize the weighted sub-scores (coverage/experience/project/confidence/quality) and how they compose `overall_score`. Frontend-led.
    4. **Roadmap & interview questions panels** — render `learning_roadmap` + `interview_questions`. Frontend-led.
    5. **Robustness** — real-LLM path hardening, error/empty-state handling, larger-JD requirement extraction. Backend-led.

    ## Working rule
    Pick ONE slice at a time, define its contract, run Backend + Frontend in parallel, merge + QA, push only when green. Keep this file updated as slices are chosen and completed.
