# How Retrieval Works — Upload to Model Lab

A walkthrough of what actually happens to a resume, which model touches it where, and
what makes the search fast and accurate. Written to be read aloud to a team.

---

## 1. The one sentence version

> A resume is **parsed once**, **embedded once per model**, and stored in **one shared
> place**. Every screen — Resume, AI Analysis, Ranking, Recruiter Assistant, Model Lab —
> *reads* those same vectors. Nothing re-embeds when you change screens. A question is
> answered by **one** query embedding plus **one** scan of a flat index, so the cost of
> asking does not grow with the size of the talent pool in any way a recruiter can feel.

---

## 2. The journey of one resume

```
  UPLOAD  (POST /resume/upload)
      │   validate extension + MIME, SHA-256 the bytes, reject exact duplicates
      │   save to storage/uploads/resumes/<id>_<filename>
      │   ── returns immediately ──────────────────────────────────┐
      ▼                                                            │
  DOCUMENT LAYER  (services/ai/embedding_store.sync_documents)     │ background thread
      │   PDF/DOCX  ->  text        (document_loader)              │ services/
      │   text      ->  sections    (resume_structuring_service)   │ indexing_service.py
      │   sections  ->  chunks      (~350 chars, 600 max, 1 sentence overlap)
      │   text      ->  profile     (name, title, years, contact, skills — deterministic)
      │                                                            │
      │   WRITES: storage/vectors/shared/documents.db              │
      │           tables: resumes(fingerprint, profile) · chunks(section, page, text)
      │                                                            │
      │   MODEL-INDEPENDENT. This is the step that makes a fair    │
      │   comparison possible: every model is shown the SAME chunks.│
      ▼                                                            │
  EMBEDDING  (embedding_store.index_model, once per model)         │
      │                                                            │
      ├── bge    BAAI/bge-large-en-v1.5   1024d  in-process ONNX  ─┤
      ├── mxbai  mxbai-embed-large-v1     1024d  in-process ONNX  ─┤
      └── gpu    nomic-embed-text          768d  your Ollama box  ─┘
              │
              │   WRITES: vectors-<model>.faiss   (FAISS IndexFlatIP)
              │           rows-<model>.json       (row -> chunk_id map)
              ▼
  SEARCHABLE — by every screen, for every model that finished.
```

**Nothing above happens twice.** The document layer is content-fingerprinted (SHA-256 of
the file), so re-running costs nothing for a resume that has not changed. Each model's
indexer reuses vectors it already holds and embeds only the new chunks.

---

## 3. Which model is used where, and why

| Stage | Model | Where it runs | Why that one |
|---|---|---|---|
| Parse / chunk / profile | **none** | in-process, pure Python | Regex + layout rules. Deterministic and free — no model needed to know where the Skills section is. |
| Embedding (indexing) | **bge · mxbai · nomic** | bge/mxbai in-process (CPU ONNX), nomic on your GPU box | All three, so any of them can answer and Model Lab can compare them like-for-like. |
| Embedding (query time) | **the one you selected** | same place | A query must be embedded by the model that built the index it searches. |
| Sparse matching | **BM25** | in-process, precomputed | Catches exact terms an embedding can blur (`SAP`, `K8s`, a product name). |
| Ranking / scoring | **none** | pure arithmetic | Section-depth scoring. Reproducible, defensible, auditable. |
| Query understanding | **rules first, llama3.1:8b only if stuck** | RunPod | The LLM labels spans; the corpus decides what they mean. |
| Answer wording | **llama3.1:8b** | RunPod | Phrasing only. It never supplies a fact, a name, a number or a score. |

### Are all three really embedding, or just BGE?

**All three, as of now.** Before this change, only BGE was being filled reliably, because
indexing was a manual button on one screen and nobody pressed it for the other two.
`services/indexing_service.py` now fans an upload out to **every model that is reachable
at that moment**, in sequence. A model whose endpoint is down is skipped — not failed —
and `/embeddings/coverage` shows it behind until it catches up.

> Why sequentially and not in parallel? They compete for the same CPU, and a 0.64 GB ONNX
> model running beside a 1.3 GB one is how the allocator runs out of memory.

### Why each model has its own similarity floor

| model | dims | floor | measured on |
|---|---|---|---|
| bge | 1024 | **0.62** | BGE's own distribution |
| mxbai | 1024 | **0.537** | the value at which mxbai admits the same *fraction* of pairs BGE admits at 0.62 |
| nomic | 768 | **0.51** | relevant pairs ≥ 0.592, unrelated ≤ 0.426 — the floor sits in its gap |

Cosine values are **not comparable across models**. A single shared threshold would hand
the win to whichever model happens to score higher in absolute terms — that is threshold
luck, not retrieval quality. Equal selectivity is what makes the three-way comparison
about ranking.

---

## 4. Why the search is fast

### The shape that was replaced

The old chatbot index stored **one FAISS index per resume**. Answering "who in my pool
knows Java" therefore cost, *per question*:

```
300 index loads from disk  +  300 retriever constructions  +  300 BM25 corpus tokenizations
```

Seconds per question, growing linearly with every CV you add.

### The shape now

```
ONE flat index over every chunk of every resume, with resume_id in per-chunk metadata.
```

One question costs:

| step | cost |
|---|---|
| 1. Understand the question | pure regex + dict lookups — **microseconds** |
| 2. Hard pre-filter (years, location, named person) | dict scan over ~300 small profile records — **microseconds** |
| 3. Embed the query | **one** forward pass — ~10–30 ms local |
| 4. Dense search | **one** FAISS `IndexFlatIP` scan — a matrix multiply |
| 5. Sparse search | postings lookup over **precomputed** BM25 stats |
| 6. Fuse + score | Reciprocal Rank Fusion + arithmetic — **microseconds** |

Five things make it quick, and none of them is "we bought a bigger GPU":

1. **One index, not N.** The single biggest win. Nothing is loaded per resume.
2. **Cheap filters run before expensive ones.** "10+ years in T. Nagar" is answered from
   small profile dicts, so the vector search only ever sees candidates that already
   cleared the hard constraints.
3. **BM25 is precomputed at load.** Term frequencies, document lengths and IDF are built
   once and cached. A query touches only the postings of its own terms.
4. **Everything is cached in-process.** Index, chunk table, BM25 stats and the corpus
   lexicon are built on first use and reused for the process's lifetime.
5. **Indexing is incremental.** Adding resume #301 embeds resume #301. Nothing else moves.

`IndexFlatIP` is an exact, brute-force index — no approximation, no recall loss. At this
scale (a few thousand chunks × 1024 dims) that is a single BLAS matrix multiply, which is
*faster* than the overhead of a fancier index would be. If the pool reached millions of
chunks you would swap in an IVF/HNSW index; the interface would not change.

**Measured, end to end:** deterministic answers land in ~200 ms–1.8 s. With llama3.1:8b
writing the prose, 1.4–3.2 s — i.e. **the LLM is ~80% of the wait**, and the retrieval it
is describing is the fast part.

---

## 5. The question everyone asks: is it the prompt or the embedding model?

They control **different things**, and confusing them is how teams tune the wrong knob.

| What you notice | What actually controls it |
|---|---|
| "It missed a candidate who obviously fits" | **Embedding model + chunking.** This is recall. A prompt cannot retrieve something the search never returned. |
| "The right people came back in the wrong order" | **Scoring.** Section-depth + hybrid fusion — deterministic code, not the model. |
| "It answered a different question than I asked" | **Query understanding.** Not the embedder, not the prompt. |
| "The wording is clunky / too long" | **The prompt.** This is the *only* thing the prompt controls. |
| "It's slow" | **The LLM call**, almost always. Retrieval is milliseconds. |

### The invariant that makes this safe to say out loud

> **The LLM never supplies a fact, and never supplies a requirement.**

Every name, percentage, year, email and quote comes from deterministic retrieval. The
model receives a context block of already-computed facts and writes 2–4 sentences about
them. `ground_candidates()` then deletes any candidate not tied to a real indexed resume
with real evidence. And in query understanding the model only marks up *spans of the
recruiter's own sentence* — every span must appear verbatim in the message **and** resolve
against the corpus before it can affect the search.

That is why the platform still works with the LLM switched off entirely: turn it off and
you get the same candidates, the same scores and the same evidence, in slightly plainer
English.

**The honest headline:** the biggest accuracy gains in this project did **not** come from
a better embedding model. They came from *understanding the question properly* — not
reading `now` as a skill, not reading the word `candidate` as somebody's name, not
dropping the second half of a two-constraint query.

---

## 6. Should we use a GPU to make embedding faster?

Separate the two costs — they have opposite answers.

**Indexing (bulk, one-off): yes, GPU helps a lot.**
mxbai runs at roughly **1.9 chunks/second** on a laptop CPU. A 300-resume pool is ~4,500
chunks, so ≈ 40 minutes for that one model. That is exactly the workload a GPU eliminates.

**Querying (per question): no, and it can be slower.**
A query is *one* embedding. Locally that is ~10–30 ms. Sending it to a remote GPU adds a
network round trip that usually costs more than the compute it saves. That is why the
local model is the default for answering, and why the GPU engine transparently falls back
to local when its endpoint is unreachable.

The architecture already supports the sensible split — index with the GPU, answer locally —
because **each engine owns its own index** and the engine is part of that index's identity.

Other levers worth knowing, in order of value for money:

1. **Don't re-embed.** Already handled: fingerprinting + incremental indexing.
2. **Chunk well.** Fewer, better chunks beat more, noisier ones — the cost is linear in
   chunk count.
3. **Batch size.** Not a free knob: it is part of the model's identity. mxbai runs at
   batch 8 because at FastEmbed's default of 256 it asked ONNX for a 1.15 GB activation
   buffer and died.
4. **Then** buy the GPU.

---

## 7. Things a reviewer is likely to probe

**"How do you know a model isn't answering about a different set of resumes?"**
`/embeddings/coverage` reports each model's indexed resume count against the shared
document set, plus the intersection. Model Lab's *fair mode* restricts a comparison to
that intersection and says so when it does.

**"Vectors from different models aren't comparable — how is that enforced?"**
Structurally. The engine name is part of the storage path (`vectors-<model>.faiss`), a
query is always embedded by the engine that owns the index, and dimension mismatch
triggers a rebuild rather than a silent bad search.

**"What if the LLM hallucinates a candidate?"**
It cannot reach the recruiter. `ground_candidates()` drops anything without a real
`resume_id` and at least one retrieved evidence chunk.

**"What happens when the GPU box goes down?"**
Indexing skips that model and reports it behind. Querying falls back to the local model
and the answer says which engine actually served it. Never a silent substitution — a
silent one would search the wrong index.

**"Is FAISS brute force going to scale?"**
Exact search is correct and fast to ~10⁵–10⁶ chunks. Beyond that, swap `IndexFlatIP` for
IVF/HNSW inside `ModelIndex`; nothing above it changes.

---

## 8. Cheat sheet

```
storage/vectors/shared/
├── documents.db            ← parsed ONCE, model-independent (resumes + chunks + profiles)
├── vectors-bge.faiss       ← BGE-large     1024d, floor 0.62
├── vectors-mxbai.faiss     ← mxbai-large   1024d, floor 0.537
├── vectors-gpu.faiss       ← nomic          768d, floor 0.51
└── rows-<model>.json       ← FAISS row -> chunk_id
```

| | |
|---|---|
| Chunk size | ~350 chars target, 600 max, 1-sentence overlap |
| Index type | FAISS `IndexFlatIP` on L2-normalised vectors (inner product = cosine) |
| Hybrid | dense 0.6 / sparse 0.4, fused by RRF (k = 60) |
| Reasoning LLM | llama3.1:8b on RunPod — wording only, never facts |
| Auto-index | on upload + one incremental pass at startup (`AUTO_INDEX_ON_UPLOAD`, `AUTO_INDEX_ON_STARTUP`) |
| Works with no LLM? | Yes — same candidates, same scores, plainer prose |
