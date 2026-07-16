---
name: rag
description: Retrieval-Augmented Generation conventions for the AI Hiring Platform — embeddings, FAISS, semantic retrieval, evidence ranking, and the evidence-before-reasoning principle. Use when working on ingestion, retrieval, or evidence.
---

# RAG / Retrieval

## Principle: evidence before reasoning
The Candidate Intelligence Agent must produce **structured evidence** before any reasoning happens. The Hiring Decision Agent reasons only from that evidence — never point an LLM at raw resume/JD text. This is what makes the system Agentic RAG rather than "LLM reads a document."

## Pipeline
```
document_loader   → extract text (PDF/DOCX)
resume_structuring_service → semantic TextNodes with section metadata (Experience/Projects/Skills/Summary)
embedding_service → BAAI/bge-small-en-v1.5 vectors (lazy singleton model)
vector_store_service → FAISS index, persisted per resume_id
jd_requirement_extractor → requirements from JD (taxonomy-boundary matching)
retrieval_service → per-requirement top-k retrieval + ranking
```

## Ranking is algorithmic and config-driven
Each match's confidence combines normalized signals with weights in `core/config.py`:
`similarity (RETRIEVAL_WEIGHT_SIMILARITY) + section importance + detail density + technical specificity`.
Tune via config, never by hardcoding per-skill logic.

## Generic semantics (no hardcoded skills)
Category classification and transferability live in `services/ai/skill_semantics_service.py`:
- `classify_category()` — nearest-centroid over prototypes built from `TECH_TAXONOMY` embeddings (generalizes to unseen skills).
- `estimate_transfer()` — max cosine similarity of a missing skill to demonstrated skills + same-category boost.
Reuse these. Do NOT reintroduce lookup dicts or `if skill == "x"` chains.

## Rules
- Retrieve structured evidence with provenance (chunk, section, score, confidence).
- Keep embeddings local; the model is a lazy singleton — don't reload per call.
- Evidence must be traceable back to the resume; never fabricate matches.
