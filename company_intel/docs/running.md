# Running company_intel

## Every time — the short version

```powershell
cd "d:\nithin.s works\jd comparre\code_clause_2-golden---CC3598\company_intel"
.\run.ps1
```

Then open <http://localhost:8123/docs>. Stop it with `Ctrl+C`.

That is the whole daily loop. Everything below is what `run.ps1` does and what to do
when something is off.

## First time only

```powershell
cd company_intel
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env      # then fill in QDRANT_URL and QDRANT_API_KEY
```

The `.env` is already written and pointed at your Qdrant cluster, using its own
collections (`ci_companies`, `ci_sources`, `ci_content`). Nothing it does touches the
`ragcompanydetails` collection the hiring platform uses.

Embeddings and answers both come from your RunPod endpoint (`EMBED_PROVIDER=ollama`,
`ANSWER_PROVIDER=ollama`). **If the pod is stopped, Web mode stops** — not just
crawling but search too, because the 768d vectors it produces are the only ones the
collection can be compared against. The Web tab says so explicitly rather than
failing quietly.

To work without the pod, set `EMBED_PROVIDER=fastembed` and `EMBED_DIM=384` — but that
is a different vector space, so the `ci_content` collection must be dropped and every
company re-crawled. Answers alone can be made local by setting `ANSWER_PROVIDER=none`,
which costs prose and nothing else.

## Why port 8123 and not 8080

Port 8080 is inside a Windows reserved exclusion range on this machine — binding it
fails with `WinError 10013`. 8123 is outside it. To check what is reserved:

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

## The daily loop, by hand

```powershell
# start (leave this window open)
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8123

# add --reload while you are editing code; leave it off otherwise, it doubles startup
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8123 --reload
```

Confirm it is alive:

```powershell
curl http://localhost:8123/api/v1/health
```

`status: ok` means Qdrant answered. `status: degraded` means it did not, and the
`detail` field says why. The health endpoint never depends on the store, so it answers
either way.

## Adding a company and asking it things

```powershell
$B = "http://localhost:8123/api/v1"

# 1. register — LinkedIn URLs are stored as links, never crawled
irm "$B/companies" -Method Post -ContentType application/json -Body '{
  "name": "Acme", "domain": "acme.com",
  "linkedin_urls": ["https://www.linkedin.com/in/jane-doe"],
  "deny_patterns": ["/blog/20", "/tag/"]
}'

# 2. crawl (background:true returns immediately)
irm "$B/companies/acme-com/crawl" -Method Post -ContentType application/json `
  -Body '{"max_pages": 20, "background": true}'

# 3. watch progress
irm "$B/companies/acme-com" | ConvertTo-Json -Depth 5

# 4. ask
irm "$B/chat/query" -Method Post -ContentType application/json `
  -Body '{"message": "what does Acme sell?"}'
```

Or just use <http://localhost:8123/docs> — every endpoint is there with a Try-it button,
which is easier than quoting JSON in PowerShell.

## Before crawling a site for the first time

```powershell
.\.venv\Scripts\python.exe scripts\preview_crawl.py acme.com --max-pages 10 --show-chunks
```

Prints exactly what would be indexed, without embedding or writing anything. Worth 30
seconds on every new site: if the chunks come back as navigation menus or a
"please enable JavaScript" notice, indexing it will only bury that problem inside
vectors where you cannot see it.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest        # 111 tests, ~90s
```

No network, no Qdrant, no model download — the store, the sites and the embedder are
all faked. Safe to run any time.

## When something looks wrong

| Symptom | Where to look |
|---|---|
| `WinError 10013` on start | Port is reserved. Use another one — see above. |
| `status: degraded` | `QDRANT_URL` / `QDRANT_API_KEY` in `.env`, or the cluster is down. |
| `503` from every data route | Same thing — the store guard is refusing cleanly rather than throwing a 500. |
| Crawl indexes 0 pages | Run `preview_crawl.py` on the domain. Usually robots.txt, or a JS-rendered site. |
| Answers say "nothing in the indexed pages" | Either genuinely not crawled, or the question scored below `EMBED_MIN_SIMILARITY`. Check `GET /companies/{id}` for what is actually indexed. |
| Answers quote nav/cookie text | Extraction problem. `preview_crawl.py --show-chunks` shows it directly. |
| Everything is slow | Expected on first crawl. On GPU ~4 s/page; on local CPU 8-15 s/page. |
| Web tab shows "embedding endpoint is unreachable" | The RunPod pod is stopped or its URL changed. Update `EMBED_OLLAMA_URL` / `ANSWER_OLLAMA_URL` in `company_intel/.env` and restart. |
| A site indexes 0 pages with a 403 | That site's WAF blocks datacenter traffic (tcs.com, infosys.com and freshworks.com all do). It cannot be crawled from here. |

## Leaving it running

The hourly refresh scheduler starts with the app, so anything that comes due is
re-checked while the process lives. Stop the process and refresh stops with it — due-ness
is stored in Qdrant, so nothing is lost, and the next start picks up where it left off.

For a real deployment, run it as a service and drop `--reload`.
