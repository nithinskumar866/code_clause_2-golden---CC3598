# AI Hiring Intelligence Platform

> Explainable Multi-Agent Retrieval-Augmented Generation (RAG) Recruitment Dashboard Orchestrated via LangGraph.

This platform operates as an explainable recruitment copilot. It structures candidate resumes against job descriptions, indexes them in a local vector store, and runs intent-based evaluation reasoning to output multi-perspective compatibility scores.

---

## 🏛 System Design Architecture

```text
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
  Embedding (BGE-Large v1.5)                               │
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

## ✨ Features (recruiter-facing)

- **Resume & JD upload** — PDF/DOCX, parsed and locally embedded (BGE) into a per-resume FAISS index.
- **Explainable evaluation** — two-agent Agentic RAG (evidence gathering → reasoning) producing a full hiring report where every score traces to retrieved evidence.
- **Authenticity & keyword-stuffing detection** — deterministic credibility score, over-claimed-skill flags, corroboration ratio.
- **Candidate profile & seniority fit** — name/title/years/level vs the JD's stated requirement.
- **Requirement matching** — per-requirement evidence, confidence, must-have vs nice-to-have badges, importance-weighted coverage.
- **Skill gap + learning roadmap + interview questions + recruiter recommendation**.
- **Multi-candidate ranking** — score many resumes against one JD, leaderboard with top candidate, sort/filter, CSV export.
- **Analytics** — totals, average score, score distribution, recommendation split, daily/weekly/monthly trends, top resumes/jobs, most-common missing skills.
- **Recruiter workflow** — editable, persistent pipeline stage (Applied → Screening → Reviewed → Interview Scheduled → Interview Completed → Selected → Rejected → Offer Sent).
- **Recruiter notes** — full add/edit/delete CRUD per candidate.
- **History** — search, filter, sort, paginate; open any candidate; JSON + server-rendered **PDF** export.
- **Offline-first** — the deterministic engine produces the entire report with no LLM key; add a provider key to enable LLM reasoning.

**Retrieval quality**: section-aware, sentence-aware chunking; hybrid retrieval fusing dense (FAISS cosine) and sparse (BM25) rankings via Reciprocal Rank Fusion.

---

## 🧭 Job Portal (candidate-facing sections + the assistant popup)

This SPA is the whole website. Alongside the recruiter console it hosts the
candidate-facing job portal, served by the **separate .NET 9 API** in
`job_portal/` — a different backend and a different audience, but not a different
application. The portal code is quarantined in `frontend/src/portal/`, mirroring
that API's contract rather than the Python one.

- **Career Assistant popup** — the button in the bottom-right corner of *every*
  page. Drop in a resume, watch it narrate parsing → profiling → search →
  ranking, and get scored matches with a per-role recruiter note. It streams over
  SignalR (`portal/useChatHub.ts`) and is mounted once by the app shell, so
  navigating between sections never drops the conversation or the socket; replies
  that land while it is closed come back as an unread badge.
- **Job Board** (sidebar → *Job Portal*) — browse open roles with exact filters,
  import the recruiter platform's JDs, remove postings.
- **Post a Job** — upload a JD document (parsed, structured and indexed on the
  way in) or fill in the form.
- **Apply, with a review step** — a candidate can apply to one role or to a whole
  set of matches. Nothing is sent until they confirm: the review lists every
  posting with the letter that would go out, the fit it would be judged on, and
  any it has already been sent to (shown and locked, never silently dropped).
  Each letter is editable, each role can be deselected, and the count on the
  button is the count that gets sent. Capped at 25 postings per run.
- **Applicants** — the recruiter's inbox: who applied to what, the skill verdicts
  and score **frozen at the moment they applied**, the letter with its provenance
  (composed, model-written, or edited by the candidate), and the CV to download.
- **Copilot mode** — the assistant drives the site. "Remote roles over $120k"
  re-filters the board; a shortlist navigates you to the board showing exactly
  those roles, in its ranking, with a banner saying why.
- **Ask it where things are** — "take me to the resume update section" opens
  Resumes; "what is the Model Lab?" explains it and offers to open it. Requests
  it cannot place are answered honestly: it asks ("Did you mean System Status?",
  or offers the three pages under Documents) rather than guessing, and says
  plainly that there is no such page rather than inventing one.

**Why a cover letter is the riskiest text here.** Everything else the model writes
describes a result the candidate can check for themselves; a cover letter goes to
an employer under their name. So the model is given only the candidate's evidenced
skills, title and years, never the resume text to embroider — and the result is
then verified: a letter mentioning a skill scored *Missing*, or containing an
email address or phone number, is discarded for the deterministic one. Contact
details are appended from the parsed document, because an email is either
literally in the resume or it is wrong. With no model configured the composed
letter ships, and it is a real letter rather than a placeholder.

**Why navigation is not left to the model.** Intent is resolved deterministically
against `NAV_GROUPS` — the same registry the sidebar renders — in
`components/layout/navIntent.ts`. A model asked "which page do they want?" will
answer "the Settings page" for an app that has none, and the user then hunts for
something that was never built; scoring against the real registry means the
assistant can only ever offer a page that exists. Each route carries its own
`description` and `keywords`, so a route and the words that reach it stay one
fact — `navConfig.test.ts` fails if a route is added without them. The resolver
also declines: anything that does not clearly read as navigation goes to the
job-portal conversation untouched, so "show me remote roles" stays a job search.
Navigation is answered locally and never touches the .NET API, so it keeps
working when the portal backend is down.

Nothing contacts the .NET API until one of these is actually used, so running only
the recruiter console needs only the Python backend. When it is not running, the
sections and the popup say so rather than failing silently. Point the SPA at it
with `VITE_PORTAL_API_BASE` (default `http://localhost:5290`).

---

## 💬 Recruiter Assistant (chatbot over the whole talent pool)

An additive layer on top of the JD pipeline. Where the evaluation flow answers *"how
does THIS candidate match THIS JD"*, the assistant answers *"which of my candidates
matches this need"* — over every uploaded resume, with no JD required.

```
HR question ─► guardrails ─► intent parse ─► profile pre-filter ─► ONE hybrid search
                 (refuse)     (skills,        (years/name, over      over the unified
                              min years)      100 tiny records)      corpus index
                                                      │
                                                      ▼
                          ranked candidates ◄── deterministic scoring
                                   │              (coverage · evidence · experience)
                                   ▼
                          reasoning (LLM or deterministic) ─► grounding guard ─► answer
```

**Efficiency.** The evaluation pipeline keeps one FAISS index *per resume*, so a
pool-wide question would cost 100 index loads + 100 BM25 tokenizations. The assistant
adds a **unified corpus index** (`services/ai/corpus_index_service.py`): one flat FAISS
index over every chunk with `resume_id` metadata, precomputed BM25 statistics, an
in-process cache, and **incremental sync** (adding resume #101 embeds only #101).
Measured on a 108-resume pool: **17–24 ms** per question on the deterministic path,
1 vector search instead of 108.

**Embedding engines (dual, selectable per question).** Retrieval quality is bounded by
the embedding model, so two are available and switchable at runtime — both self-hosted,
no third-party vector service:

| Engine | Model | Dims | Where | Latency | Evidence floor |
|---|---|---|---|---|---|
| `bge` *(default)* | BAAI/bge-large-en-v1.5 | 1024 | in this process, CPU | **~120 ms** | 0.62 |
| `gpu` | nomic-embed-text | 768 | your Ollama endpoint | ~2 s | 0.51 |

Measured separation between genuine and unrelated matches on the same probe set:
**BGE +0.069** (relevant ≥0.646, unrelated ≤0.577) vs **nomic +0.165** (relevant
≥0.592, unrelated ≤0.426). nomic discriminates ~2.4× more cleanly, at ~65× the latency.

Because thresholds are a property of the *model*, each engine carries its own floor —
reusing BGE's 0.62 on nomic silently discarded real matches (a query returned 1 result
instead of 3 until this was fixed). Vectors from the two models are not comparable, so
**each engine owns a separate index** (`storage/vectors/corpus-<engine>/`); switching
engine switches index and never mixes them. Requesting `gpu` while its endpoint is down
serves the local index and *says so* (`diagnostics.embedding_engine`) rather than
failing or silently substituting. Revert to local-only with `EMBEDDING_ENGINE=bge`.

**Vector storage.** FAISS holds the vectors (in-process, which is what makes search
~30 ms); chunk text, metadata and the resume manifest live in a **SQLite database**
beside it (`corpus-<engine>/corpus.db`) — inspectable, backup-able, and written in one
transaction so a crash cannot leave a half-built corpus.

**Anti-hallucination.** The reasoning layer is never the source of facts. Names,
percentages, years, contacts and quotes come from deterministic retrieval; the LLM only
phrases the summary. Any candidate it invents has no `resume_id` and is deleted by
`ground_candidates` before the recruiter sees it. When nothing matches, the assistant
says so rather than offering a weak guess.

**Guardrails** (`services/ai/chat_guardrails.py`), applied before retrieval and after
reasoning: out-of-scope questions, prompt injection, probes for credentials/config/
source, screening on protected attributes (age, gender, religion, caste, marital
status, nationality) and abusive input are all refused with an explanation. Candidate
contact details **are** shown — recruiters need to reach people. 14 guardrail cases and
6 must-not-block cases are covered by `tests/test_chat_guardrails.py`.

**Query understanding** (`services/ai/chat_query_understanding.py`). A question is
classified before anything is searched — `profile` · `skill_search` · `existence` ·
`compare` · `count` · `ambiguous`. Two rules keep it honest:

- **A word is a skill only if the data says so.** It must match the tech taxonomy or a
  term that genuinely appears in the indexed resumes (the corpus vocabulary), exactly
  or via a close-typo match (`javaa` → `java`, stated in the answer, never silent).
  Ordinary English can no longer become a requirement — the earlier version searched
  for the "skill" `let` in *"let me know more about David Pillai"*.
- **When nothing is recognised, ask.** An unknown person or an unparseable question
  returns a clarifying question with the closest real names, never a guess.

**Scoring by evidence depth.** Each requested skill earns a depth score from *where* it
is proven — Experience 1.0 · Projects 0.9 · Certifications 0.85 · Summary 0.45 ·
Education 0.40 · Skills-list 0.35 — blended 75/25 with how broadly it recurs across
sections. Coverage is the mean depth, so it varies instead of being a flat 100%: Java
across certifications + experience + skills scores 100% and outranks Java in
experience + skills (88%), which outranks a bare Skills-list mention (~33%).

**Follow-ups.** Conversation memory (in-process, TTL-bounded) supports *"of those, who
also knows AWS?"* and *"tell me more about the 2nd one"*. Every answer also returns
suggested next steps; *"Evaluate X against a job description"* hands the candidate to
the existing AI Analysis page via `?view=analysis&resume=<id>` rather than duplicating
that flow in the chat.

Endpoints: `chat/query` · `chat/reset` · `chat/corpus/status` · `chat/corpus/sync` ·
`chat/health`. UI at **Evaluation → Recruiter Assistant**.

Bulk-load a folder of resumes (registers + indexes them):
```bash
cd backend && python -m scripts.bulk_load_resumes /path/to/resumes
```

---

## 🔌 API surface (`/api/v1`)

`health` · `resume` (upload/list) · `job` (upload/list) · `analysis/evaluate` · `analysis/rank` ·
`analysis/history` (list/detail/delete/clear) · `dashboard/*` · `analytics/*` (overview, score-distribution,
recommendation-distribution, trends, top-resumes, top-jobs, skill-frequency, recent) ·
`analysis/{id}/notes` + `notes/{id}` (CRUD) · `analysis/{id}/status` (get/patch) ·
`analysis/{id}/export/json|pdf`. Every response uses the `{success, message, data, meta?}` envelope
(file exports stream bytes). Interactive docs at `http://localhost:8000/docs`.

---

## ⚙ Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.12+)

---

### Backend Setup
1. Navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install the python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   The backend API docs are available at `http://localhost:8000/docs`.

5. Run the automated test suites:
   ```bash
   python -m pytest
   ```

---

### Frontend Setup
1. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install npm modules:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   Vite will serve the page at `http://localhost:5173`.

4. Build production static bundle:
   ```bash
   npm run build
   ```

---

## 🔧 Environment variables

Copy the provided examples and edit as needed:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env      # optional; only to change the API host
```

**Backend** (`backend/.env`) — all optional:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `google` (aliases: `gemini`, `claude`) |
| `LLM_MODEL` | *(per-provider default)* | Explicit model id |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | *(empty)* | Set only the one for your provider. **All empty → deterministic engine.** |
| `DATABASE_URL` | `sqlite:///./hiring_platform.db` | SQLAlchemy URL |
| `RETRIEVAL_MIN_SIMILARITY` | `0.30` | Cosine floor below which evidence is dropped |
| `HYBRID_RETRIEVAL_ENABLED` | `true` | Fuse BM25 + vector via RRF |
| `HYBRID_WEIGHT_DENSE` / `HYBRID_WEIGHT_SPARSE` / `HYBRID_RRF_K` | `0.6` / `0.4` / `60` | Fusion tuning |
| `CHUNK_TARGET_CHARS` / `CHUNK_MAX_CHARS` / `CHUNK_SENTENCE_OVERLAP` | `350` / `600` / `1` | Sentence-aware chunking |
| `LOG_LEVEL` | `INFO` | Log verbosity |

**Frontend** (`frontend/.env`):

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE` | `http://localhost:8000` | Scheme + host of the Python backend; the client appends `/api/v1`. |
| `VITE_PORTAL_API_BASE` | `http://localhost:5290` | Scheme + host of the .NET job-portal API behind the Job Board, Post a Job and the assistant popup. Separate because it is a separate process that can be deployed independently. Leave the portal API stopped and those sections simply report it as unreachable. |

---

## 🚀 Deployment

**Backend** (any ASGI host):
```bash
cd backend
pip install -r requirements.txt
# Production server (no --reload); scale workers to taste.
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```
- SQLite is created/migrated automatically on startup (`init_db` + additive `ensure_schema`). For multi-worker or containerized deployments, point `DATABASE_URL` at a shared DB and mount `backend/storage/` (FAISS indices + report JSONs) on a persistent volume.
- CORS currently allows all origins (dev-friendly). Restrict `allow_origins` in `app/main.py` for production.
- First run downloads/loads the BGE model (slow once, then cached).

**Frontend** (static SPA):
```bash
cd frontend
VITE_API_BASE=https://your-backend.example.com npm run build
# Serve the ./dist folder from any static host / CDN (Nginx, Netlify, S3+CloudFront, …).
```

**Verification gate** (run before shipping):
```bash
# backend
cd backend && python -m pytest -q                 # 133 passed
# frontend
cd frontend && npm run build && npm run lint       # tsc+vite exit 0, oxlint clean
```
