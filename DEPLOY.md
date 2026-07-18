# Deploying the AI Hiring Platform

The whole app (React frontend + FastAPI backend + BGE **ONNX** embeddings + FAISS) runs in **one Docker container** — the frontend is built and served by the backend at the same origin, so there's a single URL and no CORS setup.

Embeddings run on **FastEmbed (ONNX)**, not PyTorch, so the image is small (~1 GB) and fits **512 MB free tiers** — no GPU, no torch, no credit card needed for the hosts below.

- `Dockerfile` (repo root) — builds the frontend, installs the backend, pre-downloads the BGE ONNX model, serves everything on port **7860**.
- The root `README.md` YAML header also makes this repo a valid Hugging Face Docker Space.

---

## Option A — Render (free, no card for free web services)

1. Push is already done — the repo is public on GitHub.
2. Go to https://dashboard.render.com → **New → Web Service** → connect the GitHub repo `code_clause_2-golden---CC3598`.
3. Render auto-detects the **Dockerfile**. Set:
   - **Instance type:** Free
   - **Environment:** Docker
   - (no build/start command needed — the Dockerfile handles it)
4. Create. First build takes ~5–10 min. App goes live at `https://<name>.onrender.com`.
5. **(Optional) Gemini:** Environment → add `LLM_PROVIDER=google`, `GOOGLE_API_KEY=<key>`, `LLM_MODEL=gemini-2.0-flash`.

> Free Render instances sleep after 15 min idle and have an ephemeral disk (uploads/DB reset on restart) — fine for a demo.

## Option B — Koyeb (free "nano", no card)

1. https://app.koyeb.com → **Create Web Service** → **GitHub** → pick the repo.
2. Builder: **Dockerfile**. Instance: **Free (nano, 512 MB)**. Port: **7860**.
3. Deploy → live at `https://<name>-<org>.koyeb.app`.
4. Gemini: add the same env vars as above under the service's *Variables*.

## Option C — Hugging Face Spaces (free, but currently asks for a card on file)

Only if you're OK adding a payment method (you won't be charged on CPU-basic):
```bash
git remote add space https://<HF_USER>:<HF_TOKEN>@huggingface.co/spaces/<HF_USER>/ai-hiring-platform.git
git push space main
```

## Option D — Plain Docker (local or any VM)
```bash
docker build -t ai-hiring-platform .
docker run -p 7860:7860 ai-hiring-platform    # http://localhost:7860
```

---

## Configuration (env vars the image honors)
| Var | Default | Purpose |
|---|---|---|
| `PORT` | `7860` | Listen port |
| `STORAGE_DIR` | `/app/runtime/storage` | Uploads, FAISS, reports (point at a volume to persist) |
| `DATABASE_URL` | `sqlite:////app/runtime/hiring_platform.db` | DB location |
| `LLM_PROVIDER` / `GOOGLE_API_KEY` / `LLM_MODEL` | unset → deterministic engine | Enable LLM reasoning (Gemini recommended) |

The API key is **never** baked into the image or committed — it's supplied only as a runtime secret/env var.
