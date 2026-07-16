# Architecture Specifications - Master Design

This document details the system architecture of our **Explainable Hiring Intelligence Platform**, built as an intent-driven agentic copilot.

---

## High-Level System Architecture

Our platform follows a **Modular Monolith** pattern, combining a compiled Single Page Application (SPA) React dashboard with a FastAPI backend orchestrated via **LangGraph state graphs**.

```
                         React Client (Vite SPA)
                                    │
                                    ▼
                         FastAPI Routing Engine
                                    │
                                    ▼
                     LangGraph State Graph Workflow
                                    │
             ┌──────────────────────┴──────────────────────┐
             │                                             │
             ▼                                             ▼
  Candidate Intelligence Agent                   Hiring Decision Agent
  [Node 1: Local RAG Pipeline]                  [Node 2: LLM Reasoning]
             │                                             │
      Parser (PyMuPDF)                                     │
             │                                             │
    TextNodes Structuring                                  │
             │                                             │
  Embedding (BGE-Small v1.5)                               │
             │                                             │
   Vector Index (FAISS)                                    │
             │                                             │
   JD Semantic Retrieval                                   │
             │                                             │
             └──────────────► Evidence State ──────────────┘
                                                           │
                                                           ▼
                                                Explainable Report
                                                (Weights Normalized)
```

---

## Architectural Components

### 1. Unified Recruiter Dashboard (SPA)
- **Tech Stack:** React, Vite, TypeScript, Tailwind CSS, Lucide icons.
- **Role:** Handles document ingest (PDF/DOCX) for candidates and roles, coordinates API pipeline states, and displays explainable matching tables, roadmaps, and copyable communication templates.

### 2. FastAPI Services Layer
- **Role:** Serves API request routing, file ingest validation, database persistence, and exposes analysis reports.
- **Database:** SQLAlchemy ORM mapping candidate, JD, and analysis states to a local SQLite instance with Cascades.

### 3. Candidate Intelligence Agent (State Node 1)
- **Role:** Performs structural parsing, semantic chunk segmentation, local BGE vector embeddings representation, FAISS indexing, and runs requirement-wise evidence search. Writes results as structured evidence nodes into the graph state.

### 4. Hiring Decision Agent (State Node 2)
- **Role:** Consumes ONLY the evidence nodes from state. Performs multi-perspective LLM reasoning to evaluate coverage, experience fit, project relevance, evidence confidence, and general quality, compiling the final recruiter-ready report.

### 5. LangGraph Orchestrator
- **Role:** Replaces ad-hoc sequences with a compiled state machine (`StateGraph`), enforcing clean state boundaries and predictable execution flows.
