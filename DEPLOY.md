# Deploying the AI Hiring Platform

The whole app (React frontend + FastAPI backend + BGE embeddings + FAISS) runs in **one Docker container** — the frontend is built and served by the backend at the same origin, so there's a single URL and no CORS setup.

- `Dockerfile` (repo root) — multi-stage build: builds the frontend, installs the CPU-only backend, pre-downloads the embedding model, serves everything on port **7860**.
- The root `README.md` YAML header configures this repo as a **Hugging Face Spaces – Docker** app.

---

## Option A — Hugging Face Spaces (recommended: free, 16 GB RAM)

You need a free Hugging Face account. These are the only steps that require *your* login.

1. **Create the Space**
   - Go to https://huggingface.co/new-space
   - Name it (e.g. `ai-hiring-platform`), **SDK = Docker**, **blank template**, visibility your choice. Create it.

2. **Get a write token**
   - https://huggingface.co/settings/tokens → *New token* → role **Write**. Copy it.

3. **Push this repo to the Space** (from the repo root):
   ```bash
   git remote add space https://<HF_USERNAME>:<HF_TOKEN>@huggingface.co/spaces/<HF_USERNAME>/ai-hiring-platform.git
   git push space main
   ```
   The Space builds the Dockerfile (~10–15 min the first time — it installs PyTorch and bakes in the model).

4. **Open the app** at `https://<HF_USERNAME>-ai-hiring-platform.hf.space`

5. **(Optional) Enable Gemini reasoning** — Space → *Settings* → *Variables and secrets* → add secrets:
   - `LLM_PROVIDER = google`
   - `GOOGLE_API_KEY = <your key with quota>`
   - `LLM_MODEL = gemini-2.0-flash`
   Restart the Space. With no key it runs the deterministic engine.

> **Storage note:** free Spaces have an *ephemeral* filesystem — uploaded resumes, the SQLite DB, and FAISS indexes reset when the Space restarts/rebuilds. Fine for a demo. For persistence, attach Spaces **persistent storage** and set `STORAGE_DIR=/data/storage` and `DATABASE_URL=sqlite:////data/hiring_platform.db` in the Space variables.

---

## Option B — Run the same image anywhere with Docker

```bash
docker build -t ai-hiring-platform .
docker run -p 7860:7860 ai-hiring-platform
# open http://localhost:7860
```
Add Gemini with `-e LLM_PROVIDER=google -e GOOGLE_API_KEY=<key> -e LLM_MODEL=gemini-2.0-flash`.

---

## Configuration (env vars the image honors)
| Var | Default | Purpose |
|---|---|---|
| `PORT` | `7860` | Listen port |
| `STORAGE_DIR` | `/app/runtime/storage` | Uploads, FAISS, reports (point at a volume to persist) |
| `DATABASE_URL` | `sqlite:////app/runtime/hiring_platform.db` | DB location |
| `LLM_PROVIDER` / `GOOGLE_API_KEY` / `LLM_MODEL` | unset → deterministic engine | Enable LLM reasoning (Gemini recommended) |

The API key is **never** baked into the image or committed — it's supplied only as a runtime secret.
