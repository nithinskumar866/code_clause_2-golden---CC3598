# company_intel

Answers questions about companies from their own websites. Pages are crawled when a
company is onboarded, stored in Qdrant, and answered from the index.

**The chat path never crawls.** Answer latency depends on Qdrant and the embedding
model, never on whether a company's site is up, slow, or rate-limiting us today.
Freshness is a scheduled concern handled separately, so a question costs one vector
search rather than a live fetch of somebody else's website.

Standalone: it shares no code, no database and no configuration with anything else in
this repository.

## Setup

```powershell
cd company_intel
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env      # then fill in QDRANT_URL and QDRANT_API_KEY
.\run.ps1
```

Open <http://localhost:8123/docs>. After the first setup, `.un.ps1` is the whole
daily loop — see [docs/running.md](docs/running.md) for flags, ports and a
troubleshooting table.

`GET /api/v1/health` reports what is and is not configured, and answers even when
Qdrant is unreachable — a health check that dies with its database tells you nothing.

**Embeddings** come from one of two providers, chosen in `.env`:

| `EMBED_PROVIDER` | what it is | measured |
|---|---|---|
| `fastembed` | local ONNX (BGE-small, 384d), no key, no GPU | ~1.3 chunks/s on 4 cores |
| `ollama` | a remote GPU endpoint (nomic-embed-text, 768d) | ~24 chunks/s — **18x faster** |

They are **not** interchangeable at runtime. Different dimensions mean a fallback would
write vectors the collection cannot compare, so an unreachable provider raises rather
than quietly degrading — and switching provider means dropping `ci_content` and
re-crawling.

**Answering** is optional and *does* fall back safely: `ANSWER_PROVIDER=ollama` (or
`openai`) phrases the retrieved passages, `none` quotes them verbatim. A model that
cannot be reached costs readability, never accuracy, because retrieval fixes the facts
before the model is called.

## Use it

```bash
# 1. register a company (LinkedIn URLs are stored as links, never crawled)
curl -X POST localhost:8123/api/v1/companies -H 'content-type: application/json' -d '{
  "name": "Acme", "domain": "acme.com",
  "linkedin_urls": ["https://www.linkedin.com/in/jane-doe"],
  "deny_patterns": ["/blog/20", "/tag/"]
}'

# 2. crawl it
curl -X POST localhost:8123/api/v1/companies/acme-com/crawl \
  -H 'content-type: application/json' -d '{"max_pages": 40}'

# 3. watch progress
curl localhost:8123/api/v1/companies/acme-com

# 4. ask
curl -X POST localhost:8123/api/v1/chat/query -H 'content-type: application/json' \
  -d '{"message": "what products does Acme sell?"}'
```

### Look before you index

```bash
python scripts/preview_crawl.py acme.com --max-pages 10 --show-chunks
```

Crawls and prints exactly what would be stored, without embedding or writing anything.
Bad extraction is the quietest way a system like this fails — once text is inside a
vector, a corpus full of cookie banners looks the same as a corpus full of company
information, and retrieval just gets mysteriously worse. Run this on a new site first.

## How it fits together

```
register → frontier → fetch → extract → hash → chunk → embed → Qdrant
                        │        │        │
                     304? stop   │    unchanged? stop
                                 │
                          (nav/cookie/footer dropped)

ask → embed query → Qdrant search (+ company filter) → fence → answer + citations
```

| Concern | Module |
|---|---|
| The only datastore — three collections, all filters and writes | [`store/qdrant.py`](app/store/qdrant.py) |
| Who we track, and full deletion across collections | [`sources/registry.py`](app/sources/registry.py) |
| Per-URL crawl state: etag, hash, next due, failures | [`sources/state.py`](app/sources/state.py) |
| URL canonicalisation and crawl scope | [`crawl/urls.py`](app/crawl/urls.py) |
| robots.txt, and the per-host rate gate | [`crawl/robots.py`](app/crawl/robots.py), [`crawl/fetcher.py`](app/crawl/fetcher.py) |
| What to visit and in what order | [`crawl/frontier.py`](app/crawl/frontier.py) |
| HTML to clean, sectioned text | [`extract/html_text.py`](app/extract/html_text.py) |
| Page type → refresh cadence and crawl priority | [`extract/classify.py`](app/extract/classify.py) |
| Page hashing, heading-aware chunking | [`process/`](app/process/) |
| The only writer of content points | [`ingest/pipeline.py`](app/ingest/pipeline.py) |
| Re-read what has come due | [`ingest/refresh.py`](app/ingest/refresh.py) |
| Search, scoping, per-company diversity | [`chat/retrieval.py`](app/chat/retrieval.py) |
| Injection fencing, secret redaction, the system prompt | [`chat/guardrails.py`](app/chat/guardrails.py) |
| Grounded answering, LLM optional | [`chat/answer.py`](app/chat/answer.py) |

### Qdrant, and only Qdrant

Three collections. `ci_content` holds one point per chunk and is the only one ever
searched. `ci_companies` and `ci_sources` hold no meaningful vectors — they carry a
1-dimensional placeholder and are read by filtered `scroll`. That is a deliberate
trade: crawl state is key-value data, and modelling it as points keeps the service on
one database instead of adding a second one for two small tables.

### Why refresh is cheap

Three tiers, cheapest first: a conditional `GET` that returns `304` costs one round
trip and nothing else; an unchanged content hash stops before chunking and embedding;
only a real edit pays for vectors. The hash is taken over *normalised extracted text*
per **page**, never over raw HTML (which changes on every request) and never per chunk
(whose boundaries shift when a page is edited, so one added sentence would invalidate
every chunk below it).

Cadence follows page type — news every 2 days, products every 14, leadership every 30.
One global interval is wrong in both directions at once.

### Why updates delete first

Every page write is `delete_by_filter(url_hash) → upsert`. Deterministic ids overwrite
chunks that still exist but cannot remove ones that no longer do: a page edited from
nine chunks down to four would otherwise leave five orphans in the index permanently,
still matching questions, still cited with a real URL, quoting text the company deleted
months ago.

### Crawled text is untrusted

Every chunk was written by somebody else. A page can contain "ignore your previous
instructions", and by the time it reaches the model it looks like the evidence beside
it. Retrieved content is fenced and labelled as quoted material, instruction-shaped
spans are replaced (and the substitution reported in the answer), and credentials are
redacted. Every claim carries its source URL.

## Measured performance

Two real sites (basecamp.com, zapier.com), 12 pages each, 309 chunks, against Qdrant
Cloud in `sa-east-1` from a 4-core laptop.

| | local CPU (fastembed) | remote GPU (ollama) |
|---|---|---|
| Crawl, per page | ~8–15 s | **~3.3–4.2 s** |
| Crawl, 12 pages | 94–182 s | **39–50 s** |
| Embedding throughput | 1.3 chunks/s | **24 chunks/s** |
| Re-crawl, nothing changed | 2.5 s | 2.5 s |
| Answer latency | ~0.9 s (quoted) | ~2.5–4.7 s (LLM-phrased), 1.1 s when refused |
| Qdrant round trip | 372 ms (cluster is in São Paulo) | same |

**On CPU the first crawl is dominated by embedding**, not the network — which is why
moving it to the GPU endpoint cut per-page time by about 3x. What remains is Qdrant
round trips (5 per page × 372 ms) and the politeness delay, so a Qdrant region near you
is the next largest win.

**Refresh is where the architecture pays for itself.** A second crawl of unchanged sites
costs 2.5 s instead of 182 s, because the ETag and content-hash short-circuits stop
before extraction and embedding. That is the difference between periodic refresh being
routine and being unaffordable.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest      # 101 tests, ~80s
```

No network, no Qdrant and no model download: the store, the site and the embedder are
faked (the ~80 s is mostly `trafilatura` parsing sample HTML). The fake store is a real implementation of the slice of Qdrant used here rather
than a call-recording mock, because the invariant most worth testing — that deleting a
page's chunks actually removes the stale ones — is exactly what a mock would fail to
prove.

## Known limits

- **JavaScript-rendered sites yield nothing, and some sites refuse crawlers outright.**
  The fetcher reads HTML; it does not run scripts. Separately, tcs.com, infosys.com and
  freshworks.com return 403 to *any* datacenter request — even for `robots.txt`, even
  with a browser User-Agent — because their edge WAF excludes non-residential traffic.
  That is the site's decision and this crawler respects it: such pages are retired with
  a clear reason rather than retried or worked around. Run `preview_crawl.py` on a
  domain before registering it.
- **PDFs are skipped**, deliberately. They need a different extractor, and silently
  indexing an empty string from one is worse than skipping it visibly.
- **Coverage is capped** at `CRAWL_MAX_PAGES` per company. When the cap is hit, the
  crawl report says `budget_exhausted: true` — truncation is reported, never silent.
  The frontier spends the budget on About/Products/Leadership before the blog.
- **LinkedIn is not crawled.** See [`docs/linkedin.md`](docs/linkedin.md) for why, and
  for the design of how officials do get in.
- **Embedding is CPU-bound** at roughly 1.3 chunks/s on 4 cores. Onboarding hundreds of
  companies is an overnight job, not an interactive one. Refresh is unaffected.

### The retrieval floor is per model

`EMBED_MIN_SIMILARITY=0` means "use the value calibrated for the configured model".
Measured on the same corpus, off-topic questions top out at **0.43 with bge** but
**0.553 with nomic** — the two models simply use different parts of the cosine range.
Carrying bge's correct 0.55 floor over to nomic let *"who won the 2018 world cup"*
(0.553) through as a company answer. Floors: bge 0.55, nomic 0.62.

## Found by running it

Bugs the live crawls exposed that the test suite could not have, each now pinned by a
regression test:

- **`www.` and the apex crawled as separate sites**, indexing the homepage twice under
  different keys.
- **14% of chunks exceeded the embedding model's 512-token window**, so their tails were
  silently dropped from the vector while the full text stayed stored and citable.
- **A cookie banner was answering questions about the company.** Basecamp marks it
  `class="tracking tracking--hidden" aria-hidden="true"` — no "cookie" or "consent" in
  the markup at all. The fix reads the page's own `aria-hidden` / `display:none`
  declaration rather than adding another keyword; "tracking" is real vocabulary for a
  logistics company, and blacklisting it would have deleted genuine content.
- **A bot wall was indexed as company content.** techmahindra.com returns HTTP 200 with
  "JavaScript is disabled. In order to continue, we need to verify that you're not a
  robot." That text extracts, chunks and embeds perfectly — 21 chunks of it became
  citable company information. [`extract/blocked.py`](app/extract/blocked.py) now
  rejects short interstitials while leaving long pages *about* bot detection alone.
- **One page consumed a whole crawl budget.** wipro.com spent 11 of its 12 pages on one
  careers page in 11 languages, via `?locale=`. Locale parameters are now stripped like
  tracking parameters — which also keeps the corpus monolingual, since a German copy of
  a page dilutes an English question's matches rather than adding to them.
- **A company that redirects to another domain crawled one page.** ltimindtree.com
  redirects to ltm.com, so every link on the landing page was rejected as off-site. The
  crawler now adopts a domain the site's *own redirect* led it to.
- **A 403 was retried on a backoff schedule forever.** It is a permission answer, not a
  transient one, so it now retires the URL immediately with a plain-English reason.
- **A redirect's tracking parameters ended up in a citation.** The landed URL is now
  normalised before it is stored and shown.
