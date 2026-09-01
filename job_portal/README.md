# Job Portal — .NET 9 API

A candidate-facing job portal that sits beside the Python recruiter platform and
shares its database. A candidate drops a resume into a chat, the bot narrates what
it is doing while it works, and it comes back with scored matches and a
per-role recruiter note.

Separate stack, separate process, separate port. The Python backend does not know
this exists, and nothing here imports from it.

> **The UI now lives in the recruiter platform.**
> `job_portal/frontend/` is **retired** and no longer run. Its three pages were
> folded into the AI Hiring Platform SPA, which is now the single website:
> the chat became a popup available from every section
> (`ai_hiring_platform/frontend/src/portal/ChatWidget.tsx`), and the board and
> post-a-job pages became sections in its sidebar under **Job Portal**. Only the
> UI moved — this API is unchanged in role, still a separate process on its own
> port, and still owns the whole contract. The folder is kept for reference; it
> is not built, not served, and not part of any verification gate.

---

## What it does

| Feature | Where |
|---|---|
| Post a job by uploading a PDF/DOCX, or through a form | `Services/Jobs/JobExtractionService.cs`, `pages/PostJobPage.tsx` |
| Semantic "vibe" matching — vectors, not keywords | `Services/Matching/MatchingService.cs` |
| Thought-process streaming while it works | `Hubs/ChatHub.cs` → `hooks/useChatHub.ts` |
| Skill gap + recruiter note per role | `MatchingService.RecruiterNote`, `MatchCard.tsx` |
| Conversational filtering ("remote only?") | `ChatOrchestrator.ComposeNarrowingQuestion` |
| Copilot mode — the bot drives the UI | `UiActionDto` → `portal/useChatHub.ts` |
| Markdown tables, bold skills, links in replies | `react-markdown` + `remark-gfm` |
| Import the recruiter platform's existing JDs | `POST /api/jobs/import-from-platform` |

---

## Running it

Two terminals — this API, and the recruiter platform's SPA that now hosts its UI.

```bash
# 1 — API on :5290
cd job_portal/backend/JobPortal.Api
cp .env.example .env          # then fill it in; see "Configuration"
dotnet run

# 2 — the website, on :5173
cd ai_hiring_platform/frontend
npm install
npm run dev
```

Then open <http://localhost:5173>: the assistant is the button in the bottom-right
corner of every page, and the board is under **Job Portal** in the sidebar. The
SPA reaches this API at `VITE_PORTAL_API_BASE` (default `http://localhost:5290`),
and `Portal:AllowedOrigins` in `appsettings.json` already allows `:5173`.

`GET /api/health` reports what is actually working; Swagger is at `/swagger` in
development.

Tests:

```bash
cd job_portal/backend/JobPortal.Api.Tests && dotnet test   # 37 tests
cd ai_hiring_platform/frontend && npm run build            # tsc -b && vite build
```

---

## Configuration

Everything is optional, and **the portal runs with none of it set** — that is a
design rule, not an accident. What changes is capability, and the health endpoint
and the chat both say which mode is active rather than degrading silently.

| Setting | Effect when unset |
|---|---|
| `Ollama__ChatModel` + `Ollama__BaseUrl` | Replies are composed deterministically instead of written by a model. Every fact is identical either way. |
| `Ollama__EmbedModel` | Matching falls back to a deterministic hashing embedder that compares **wording**, not **meaning**. |
| `Portal__DatabasePath` | Defaults to the Python platform's `hiring_platform.db`. |

For Ollama Cloud:

```
Ollama__BaseUrl=https://ollama.com
Ollama__ApiKey=<key>
Ollama__ChatModel=glm-5.2
```

`GET /api/health` lists the model tags the endpoint actually serves — check yours
against that list, because a model id the endpoint does not have fails every call
and the portal quietly runs deterministic.

### About the fallback embedder

It is a real, reproducible, offline similarity measure over word and
character-trigram counts, and trigrams let it see through spelling variation
("Postgres" ≈ "PostgreSQL"). It does **not** give semantic adjacency: it cannot
know that "React" answers "frontend framework", because those two phrases share no
text. That is exactly the capability vibe-matching exists to provide, and it needs
a real embedding model. The chat says so out loud when it is running this way.

---

## Sharing the Python database

One SQLite file, two owners, disjoint table sets.

- **Portal-owned**: everything prefixed `portal_` — `portal_jobs`,
  `portal_job_vectors`, `portal_resumes`, `portal_chat_sessions`,
  `portal_chat_messages`. Created with idempotent `CREATE TABLE IF NOT EXISTS`
  (`Data/PortalSchemaInitializer.cs`).
- **Python-owned**: `job_descriptions`, `resumes`, `analyses`, `recruiter_notes`.
  Mapped read-only. Never written, never migrated from .NET.

Why raw DDL rather than EF migrations: a migrations history table would assert
ownership over a schema SQLAlchemy actually owns, and a stray
`dotnet ef database update` could then try to "fix" the recruiter platform's
tables. `EnsureCreated` is worse still — it does nothing once the file contains any
table, so on a database SQLAlchemy created first the portal's tables would simply
never appear.

Candidate resumes uploaded here are deliberately **not** written into the Python
`resumes` table. Those are the recruiter's screening pool; these are walk-in job
seekers, and merging them would silently put every applicant into the recruiter's
candidate pool.

---

## Architecture

```
Resume (PDF/DOCX)                    Job description (PDF/DOCX or form)
      │                                            │
      ▼                                            ▼
PdfPig / OpenXml text extraction ────────── DocumentTextExtractor
      │                                            │
      ▼                                            ▼
ResumeProfileService                       JobExtractionService
 (deterministic structure, then             (deterministic structure, then
  optional LLM gap-fill, grounded)           optional LLM gap-fill, grounded)
      │                                            │
      ▼                                            ▼
  resume vector                          job vector ──► portal_job_vectors (BLOB)
      │                                            │
      └────────────► MatchingService ◄─────────────┘
                          │
             ┌────────────┴────────────┐
             │ 1. vector retrieval     │  sees past vocabulary
             │ 2. deterministic filter │  the candidate's constraints win
             │ 3. four-dimension score │  semantic·skills·title·experience
             └────────────┬────────────┘
                          ▼
                  ChatOrchestrator ──► IChatSink ──► SignalR ──► React
                                        (thoughts, tokens, matches, UI actions)
```

### The rule that shapes everything

**The model never supplies a fact, and never supplies a requirement.**

Every job title, fit percentage, skill verdict and salary the candidate sees comes
out of deterministic retrieval and scoring. The model's whole job is to phrase
those findings. Concretely:

- Extraction runs deterministically **first**; the LLM only fills gaps, and every
  skill it proposes must appear verbatim in the source document or it is dropped
  (`JobExtractionService.Ground`). A hallucinated requirement becomes a real gap in
  a candidate's fit analysis and sends them to learn something no employer asked
  for.
- Contact details are never taken from the model. An email is either literally in
  the document or it is wrong.
- The comparison table and recruiter notes are composed in C#, from the scored
  result — not requested from the model.
- The structured result goes to the UI on its own channel, so the client renders
  cards from data rather than parsing prose.
- An "explain" follow-up is answered from the **stored** previous result and never
  re-runs retrieval. Re-running would let one question produce two rankings and
  leave the explanation defending a number no longer on screen.

### Why retrieval comes before filtering

Filters are cheap and exact; similarity is the expensive part. More importantly, a
filter must never reach into the ranking and reorder what it did not remove — a
candidate who says "remote only" means it, and no similarity score should overrule
that.

Filters are also generous about silence: a posting that does not state its work
mode is not excluded by a "remote" filter. Excluding it would treat an unstated
fact as a stated contradiction, and most postings leave most fields unstated.

### No hardcoded vocabulary

There is no list of skills, technologies, companies or cities anywhere in this
codebase, and there must not be — a portal that only recognises the skills someone
remembered to type into a constant is the keyword ATS this system replaces.

- Extraction keys on document **structure**: headings, bullets, `Label:` lines.
- Chat filters key on `JobBoardLexicon` — the locations, companies and skill names
  that actually occur in the indexed postings. The board decides what its own words
  mean.
- Skill equivalence is cosine over skill-name vectors, with the same calibrated
  thresholds the Python platform uses (0.80 equivalent, 0.68 adjacent).

---

## Three defects worth knowing about

The first two were found by running the pipeline on the repository's real
documents; the third is why the chat never worked at all. All are covered by
regression tests.

0. **A pinned SignalR assembly, on the wrong framework.** The project referenced
   `Microsoft.AspNetCore.SignalR.Common` **10.0.10** while targeting `net9.0`,
   overriding one assembly of the ASP.NET Core 9 shared framework with a build of
   a different one. The framework's own `JsonHubProtocol` and that 10.x
   `IInvocationBinder` disagree, so parsing the **first** hub message threw
   `MissingMethodException`. The handshake succeeded and every send then killed
   the connection, which surfaced in the browser only as *"Connection closed with
   an error"* — no server log above Warning, nothing pointing at a version
   conflict. `Microsoft.NET.Sdk.Web` already supplies every SignalR server type;
   the reference was removed and must not come back. SignalR assemblies come from
   the shared framework, never from a pin.

1. **Bundled requirements.** A JD bullet reading `Kubernetes, Docker, Helm` was
   compared as a single phrase, so a candidate listing all three was told they were
   *Missing* all three and advised to go and learn them. Requirements are now
   atomised (`JobExtractionService.AtomiseRequirements`). On the repository's DevOps
   JD this moved skill coverage from 33% to 73%.

2. **Sub-labels ending a section.** `Languages: Python, Bash, Go` inside a resume's
   Skills block was read as a new section heading, silently dropping every skill
   after it — which were then scored as gaps. "Languages" is no longer a heading.

---

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | What is working, which models, index counts |
| GET | `/api/jobs` | Browse the board (exact filters) |
| GET | `/api/jobs/{id}` | One posting in full |
| POST | `/api/jobs/upload` | Upload a JD document; extracts and indexes |
| POST | `/api/jobs` | Create from the form |
| DELETE | `/api/jobs/{id}` | Remove a posting and its vectors |
| POST | `/api/jobs/import-from-platform` | Pull the recruiter platform's JDs in |
| GET | `/api/jobs/index-status` | Vector counts per model |
| POST | `/api/jobs/reindex?force=` | Re-embed after a model change |
| POST | `/api/resumes/upload` | Upload a candidate resume |
| POST | `/api/resumes/{id}/matches` | Match with no conversation around it |
| POST | `/api/applications/preview` | What applying would send. Stores nothing |
| POST | `/api/applications` | Submit the reviewed applications |
| GET | `/api/applications` | Who applied (filter by `jobId`, `status`) |
| GET | `/api/applications/{id}` | One application in full |
| PATCH | `/api/applications/{id}/status` | Accept / decline; carries the Python `analysisId` |
| GET | `/api/applications/{id}/resume` | The CV as uploaded |
| GET | `/api/chat/{sessionId}` | Restore a conversation |
| POST | `/api/chat/{sessionId}/message` | Non-streaming turn (same pipeline) |
| POST | `/api/chat/{sessionId}/analyze/{resumeId}` | Non-streaming resume analysis |

**Hub** `/hubs/chat` — invoke `SendMessage(sessionId, message)` or
`AnalyzeResume(sessionId, resumeId)`. Events: `Thought`, `Token`, `Matches`,
`UiAction`, `Complete`, `Error`, `End`.

`End` always fires, however the turn ended. Without it a client that lost the
connection mid-reply keeps its typing indicator up forever, which reads as a hang.

Files go over REST rather than the hub: a multi-megabyte PDF pushed through a
WebSocket would have to be hand-chunked for no benefit. The client posts the file,
then calls `AnalyzeResume` with the id and watches the narration.

---

## Not built

The following were discussed and are **not** in this slice:

- **External auto-apply.** Applying to postings *on this board* is built (see
  `Services/Applications/`, `POST /api/applications/preview` → `POST /api/applications`).
  Submitting into third-party ATS forms is not: it needs per-vendor form handling,
  real credentials and a durable job runner, it breaks whenever a vendor changes
  their markup, and mass automated submission violates most job boards' terms.
  The contracts are shaped so an *assisted* external path — prepare the letter and
  answers, open the real form, let the human submit — can be added without redoing
  the on-board flow.
- **Standing auto-apply agent.** Criteria saved once, applied automatically to new
  postings. Deliberately not built: every application currently passes a review
  step, and an unattended agent submits things nobody looked at.
- ~~**`show_jobs` navigation.**~~ Now acted on. Once the board became a section of
  the same SPA as the chat, the action could finish what it started: it sets the
  shortlist and navigates to the board (`portal/PortalState.tsx` → `onShowBoard`).
- **Qdrant / SQL Server vectors.** Brute-force cosine over SQLite blobs is right at
  this size. `IVectorStore` exists so that stays a one-class change.
