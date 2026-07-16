# Interview Notes — Design Rationale (Q&A)

Talking points for defending the design. Each answer is the *reason*, not just the *what*. Refine with the user; future sessions can help sharpen these without re-explaining the project.

### Why Agentic RAG instead of plain RAG?
Plain RAG retrieves chunks and lets the LLM answer in one shot. Here, retrieval and reasoning are **separate agent responsibilities**: one agent gathers structured, provenance-carrying evidence; a second reasons only over that evidence. This makes the decision auditable (you can point to the exact resume snippet behind every score) and prevents the model from hallucinating from raw text.

### Why exactly two agents?
It mirrors real hiring: an investigator gathers evidence, a hiring manager decides. Separating "collect" from "decide" keeps reasoning explainable and each stage independently testable. More agents (planner/summarizer/reflection) would add complexity and opacity without improving the decision.

### Why LangGraph?
Not because we need many agents — because it makes the pipeline an explicit, inspectable state machine. Every node is testable, every state transition is visible, and the flow (`START → Candidate Intelligence → Hiring Decision → END`) is trivial to reason about and extend safely.

### Why FAISS?
Fast local approximate-nearest-neighbor search over embeddings, no external service. Keeps the system self-contained (privacy: resumes never leave the machine) and cheap, while giving semantic retrieval at interactive speed.

### Why semantic retrieval instead of keyword matching?
Keyword/ATS matching misses paraphrase, synonyms, and transferable experience ("built REST APIs in Python" should match a "backend engineering" requirement). Embedding similarity retrieves evidence by meaning, so the system evaluates *demonstrated capability*, not literal token overlap.

### Why embeddings (BAAI/bge-small-en-v1.5)?
A strong, small, local sentence-embedding model — good retrieval quality at low latency and no API cost, and it doubles as the basis for generic skill categorization (nearest-centroid) and transferability, so we avoid hardcoded skill tables.

### Why cosine similarity?
Embeddings encode meaning as direction in vector space; cosine similarity compares direction independent of magnitude, so it measures semantic relatedness robustly. It's cheap, bounded, and interpretable — ideal for ranking evidence and estimating skill transfer.

### Why deterministic algorithms instead of letting the LLM do everything?
Anything that must be reproducible and auditable — score aggregation, confidence, ranking, category classification, transfer estimation — is done with algorithms so the same input always yields the same output and every number is explainable. LLMs are non-deterministic and can drift; using them for exact computation would make results unstable and hard to defend.

### Why not a single LLM doing the whole task?
A single LLM reading a resume and "deciding" is a black box: unreproducible, prone to hallucinating qualifications, and unexplainable. Splitting into retrieval (evidence) + reasoning (over evidence), with algorithms for the exact parts, gives grounding, reproducibility, and a decision you can trace back to specific resume evidence.

### Why evidence before reasoning?
So conclusions are grounded and explainable. The decision agent can only cite what was retrieved, which makes every strength/gap/score traceable to a resume location and keeps the LLM from inventing qualifications.

### Why not keyword matching / a rule engine?
Rules and keywords don't generalize to unseen resumes/JDs and can't explain *why* beyond "term present." The goal is recruiter-grade judgment with reasoning, not a checklist.

### Why a modular monolith (not microservices)?
The domain is one cohesive pipeline; microservices would add deployment/network complexity with no scaling need at this stage. A modular monolith keeps clean boundaries (services/agents/routers/models) while staying simple to run, test, and demo.

### Why combine LLM reasoning with deterministic algorithms?
Use each where it's strongest: algorithms for what should be exact and reproducible (confidence weighting, score composition, category classification, transfer estimation — all config-driven and generalizing), and the LLM for judgment that needs natural-language reasoning. Algorithms support and validate the reasoning rather than replace it, which keeps outputs both explainable and consistent.

### How is it explainable end-to-end?
Every requirement flows Requirement → Evidence (with provenance) → Reasoning → Confidence → Decision, and scores decompose into named sub-scores with configurable weights. Nothing appears "magically."
