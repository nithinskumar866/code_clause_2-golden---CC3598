# Workflow Specifications - LangGraph Orchestration

This document details the execution lifecycle of the **Explainable Hiring Intelligence Platform** orchestrated by LangGraph.

---

## LangGraph Node Workflow

The workflow is compiled as a StateGraph containing two principal nodes. The orchestration enforces state immutability, passing data cleanly between nodes through the graph context.

```
       [START]
          │
          ▼
┌──────────────────┐
│ Ingest Documents │ ──► Validates file headers, types, and persists paths.
└──────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Candidate Intelligence Node     │
│ [Agent 1]                       │
└─────────────────────────────────┘
          │  ├── 1. Ingests Resume and Job Description
          │  ├── 2. Segment Resume text into TextNodes with metadata coordinates
          │  ├── 3. Embed text chunks using BAAI/bge-large-en-v1.5
          │  ├── 4. Search local FAISS index for each JD requirement
          │  └── 5. Write retrieved evidence chunks to Graph State
          │
          ▼
┌─────────────────────────────────┐
│ Hiring Decision Node            │
│ [Agent 2]                       │
└─────────────────────────────────┘
          │  ├── 1. Consumes structured evidence from State
          │  ├── 2. Runs algorithmic evaluations (Coverage, Gaps, Confidence)
          │  ├── 3. Computes weighted compatibility average
          │  ├── 4. Generates learning roadmaps, interview guides, and emails
          │  └── 5. Updates final unified report on disk
          │
          ▼
       [END]
```

---

## State Transition Dictionary

The `AgentState` object holds variables passed during execution:

| State Variable | Data Type | Description | Producer | Consumer |
| :--- | :--- | :--- | :--- | :--- |
| `resume_id` | `int` | Unique database identifier for the candidate's resume | FastAPI Route | Candidate Agent |
| `resume_path` | `str` | Absolute file path to the resume document on disk | FastAPI Route | Candidate Agent |
| `jd_id` | `int` | Unique database identifier for the job description | FastAPI Route | Candidate Agent |
| `jd_path` | `str` | Absolute file path to the job description document on disk | FastAPI Route | Candidate Agent |
| `analysis_id` | `int` | Unique database record tracker for execution logs | FastAPI Route | Candidate Agent, Decision Agent |
| `evidence_report` | `dict` | Structured retrieved evidence nodes mapped per requirement | Candidate Agent | Decision Agent |
| `final_report` | `dict` | Unified explainable candidate assessment report | Decision Agent | FastAPI Response, Frontend Client |
