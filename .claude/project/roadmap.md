    # Development Roadmap

    Goal: finish an interview-grade, explainable AI Hiring Platform in ~2 days using parallel Backend + Frontend sessions with on-demand QA.

    ## Dependency framework (how to parallelize any feature)
    - **Parallel-safe** (assign to different sessions): backend algorithm/service/endpoint work ∥ frontend UI/visualization work — disjoint file trees.
    - **Serialized only at the contract**: a frontend feature that needs a *new* backend field/endpoint depends on the schema. Mitigate with **contract-first** — backend lands the schema, frontend mocks it — so both still start immediately.
    - **QA depends on both** → runs at merge time in the main worktree.
    - Blocking rule: if task B needs task A's contract, A publishes the schema first; B never waits on a push.

    ## Current state (baseline)
    Sprints 1–5 complete: foundation, RAG engine, hiring-decision engine, LangGraph workflow, retrieval refinement + generic skill semantics. Baseline is green (11 tests, build ✓). See `session_log.md`.

    ## Candidate slices (to be prioritized with the user — not yet started)
    Each is one vertical slice: design → backend → contract → frontend → tests → verify → merge.
    1. **Report typing + Analysis UI** — add `HiringReport`/`RequirementFit`/`LearningRoadmapItem` to `types/index.ts`; render the explainable report (requirement table, evidence, confidence). *(Contract gap already identified — natural first slice. Backend contract already exists, so this is mostly frontend.)*
    2. **Evidence drill-down** — per-requirement evidence provenance (chunk, section, score) surfaced in the UI. Backend evidence already carries it; mostly frontend + minor schema exposure.
    3. **Scoring transparency** — visualize the weighted sub-scores (coverage/experience/project/confidence/quality) and how they compose `overall_score`. Frontend-led.
    4. **Roadmap & interview questions panels** — render `learning_roadmap` + `interview_questions`. Frontend-led.
    5. **Robustness** — real-LLM path hardening, error/empty-state handling, larger-JD requirement extraction. Backend-led.

    ## Working rule
    Pick ONE slice at a time, define its contract, run Backend + Frontend in parallel, merge + QA, push only when green. Keep this file updated as slices are chosen and completed.
