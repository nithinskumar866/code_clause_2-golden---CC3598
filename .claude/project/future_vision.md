# Future Vision

Direction, not commitments. This is where the platform *could* go. None of it is scheduled, and **nothing here may violate the core architecture** (exactly two runtime agents, evidence-first reasoning, algorithms-over-examples, full explainability, modular monolith).

## Possible future directions
- **Recruiter dashboard** — pipeline view across many candidates for a role, ranked with explainable scores.
- **Candidate dashboard** — a candidate-facing view of fit, gaps, and a learning roadmap.
- **Interview simulator** — generate and run practice interviews from the evidence + gaps already computed.
- **Conversation / memory** — let a recruiter ask follow-up questions grounded in the same evidence package (still evidence-first; no raw-doc LLM access).
- **Multi-company / multi-tenant** — org-scoped data, roles, and JD libraries.
- **Multi-modal resume analysis** — portfolios, links, and richer document types feeding the same retrieval pipeline.
- **Analytics** — aggregate insights (common gaps per role, funnel metrics) over stored analyses.

## Guardrails for any future feature
1. Retrieval stays the source of truth; the LLM never inspects raw documents directly.
2. New capabilities are added as **services** and, where needed, orchestrated by the existing two agents — not as new runtime agents.
3. Deterministic where reproducibility matters; LLM only for reasoning/generation.
4. Every new output must be explainable and traceable to evidence.
5. Backend/frontend independence and the contract-first workflow are preserved.
