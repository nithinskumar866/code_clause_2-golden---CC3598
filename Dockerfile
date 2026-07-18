# AI Hiring Platform — single-container deploy (frontend + backend).
# Built for Hugging Face Spaces (Docker SDK, port 7860) but runs anywhere Docker does.

# ---- Stage 1: build the React frontend (same-origin API calls) ----
FROM node:20-slim AS frontend
WORKDIR /fe
COPY ai_hiring_platform/frontend/package*.json ./
RUN npm ci
COPY ai_hiring_platform/frontend/ ./
# Empty base => the SPA calls the backend at same-origin "/api/v1".
ENV VITE_API_BASE=""
RUN npm run build

# ---- Stage 2: Python backend that also serves the built frontend ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    STORAGE_DIR=/app/runtime/storage \
    DATABASE_URL=sqlite:////app/runtime/hiring_platform.db \
    FRONTEND_DIST=/app/frontend_dist \
    PORT=7860

WORKDIR /app

# No PyTorch: embeddings run on FastEmbed/ONNX, so the image stays small and fits
# low-memory free tiers. Only lightweight build tools are needed.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ai_hiring_platform/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Backend source and the built frontend.
COPY ai_hiring_platform/backend/ ./
COPY --from=frontend /fe/dist ./frontend_dist

# Pre-download the BGE ONNX model into the image for fast cold starts.
RUN python -c "from llama_index.embeddings.fastembed import FastEmbedEmbedding; FastEmbedEmbedding(model_name='BAAI/bge-small-en-v1.5').get_text_embedding('warmup')"

# Writable runtime dirs (Spaces may run as a non-root user).
RUN mkdir -p /app/runtime/storage && chmod -R 777 /app/runtime /app/.cache

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
