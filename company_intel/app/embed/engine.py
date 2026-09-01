"""
Embeddings, from one of two providers.

    fastembed   local ONNX, no key, no GPU, always available — and slow on a laptop
    ollama      a remote GPU endpoint (RunPod), measured 18x faster

Which one is in use is a config choice, but it is NOT a runtime fallback. The two
produce different dimensions (384 vs 768), and a vector of the wrong width is not a
worse answer — it is a meaningless one, either rejected by Qdrant or silently compared
against numbers that mean something else entirely. So when the configured provider is
unreachable this module raises. Failing loudly beats a corpus quietly poisoned with
vectors from a second model.

**Documents and queries are embedded differently.** Both model families were trained
with task prefixes — BGE puts an instruction on the query only, nomic labels both sides
— and skipping them costs retrieval quality invisibly: nothing errors, results are just
quietly worse. `embed_query` and `embed_documents` are separate functions so the
distinction cannot be lost at a call site.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional, Sequence

from app.core.config import settings
from app.core.logging import logger

FASTEMBED = "fastembed"
OLLAMA = "ollama"

# Task prefixes, per model family. nomic labels documents and queries differently;
# BGE instructs the query alone. Applied automatically so the choice of provider does
# not change how the rest of the system calls this module.
_PREFIXES = {
    "bge": ("", "Represent this sentence for searching relevant passages: "),
    "nomic": ("search_document: ", "search_query: "),
}

_model: Optional[object] = None
_lock = threading.Lock()
_client: Optional[object] = None


# --- which provider, which prefixes -----------------------------------------
def provider() -> str:
    return (settings.EMBED_PROVIDER or FASTEMBED).strip().lower()


def model_name() -> str:
    return settings.EMBED_OLLAMA_MODEL if provider() == OLLAMA else settings.EMBED_MODEL


def _prefixes() -> tuple:
    name = model_name().lower()
    for family, pair in _PREFIXES.items():
        if family in name:
            return pair
    return ("", "")  # an unknown model gets no prefix rather than a guessed one


def dimension() -> int:
    return int(settings.EMBED_DIM)


def describe() -> str:
    """One line for logs and the health endpoint."""
    if provider() == OLLAMA:
        return f"{model_name()} @ {settings.EMBED_OLLAMA_URL} ({dimension()}d)"
    return f"{model_name()} local ({dimension()}d)"


# --- fastembed (local) ------------------------------------------------------
def _fastembed_model():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding

            logger.info(f"Loading local embedding model '{settings.EMBED_MODEL}'...")
            _model = TextEmbedding(model_name=settings.EMBED_MODEL)
            logger.info("Local embedding model ready.")
        return _model


def _embed_fastembed(texts: Sequence[str]) -> List[List[float]]:
    model = _fastembed_model()
    return [
        list(map(float, v))
        for v in model.embed(list(texts), batch_size=settings.EMBED_BATCH)
    ]


# --- ollama (remote GPU) ----------------------------------------------------
def _ollama_client():
    global _client
    with _lock:
        if _client is None:
            import httpx

            _client = httpx.Client(
                timeout=httpx.Timeout(settings.EMBED_TIMEOUT_SECONDS),
                headers={"Content-Type": "application/json"},
            )
        return _client


def _embed_ollama(texts: Sequence[str]) -> List[List[float]]:
    """
    Batched POSTs to `/api/embed`.

    Batching is the whole performance story here, not a tidiness choice: measured on a
    RunPod pod, one text per request managed 4/s while batches of 16 reached 24/s. The
    round trip dominates, so the fewer of them the better.

    Retried on transport failure because a tunnelled endpoint drops connections for
    reasons that have nothing to do with the request — but never falling back to the
    local model, whose vectors would be the wrong width for this collection.
    """
    url = settings.EMBED_OLLAMA_URL.rstrip("/") + "/api/embed"
    client = _ollama_client()
    batch_size = max(1, settings.EMBED_BATCH)
    out: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        window = list(texts[start:start + batch_size])
        last_error = ""
        for attempt in range(1, settings.EMBED_MAX_RETRIES + 1):
            try:
                response = client.post(
                    url, json={"model": settings.EMBED_OLLAMA_MODEL, "input": window}
                )
                response.raise_for_status()
                vectors = response.json().get("embeddings") or []
                if len(vectors) != len(window):
                    raise RuntimeError(
                        f"asked for {len(window)} vectors, received {len(vectors)}"
                    )
                out.extend([list(map(float, v)) for v in vectors])
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < settings.EMBED_MAX_RETRIES:
                    time.sleep(min(8.0, 2.0 ** attempt))
                    continue
                raise RuntimeError(
                    f"The embedding endpoint at {settings.EMBED_OLLAMA_URL} is not "
                    f"answering ({last_error}). Nothing can be indexed or searched "
                    f"until it is back — the local model produces "
                    f"{384}d vectors, which this {dimension()}d collection cannot use."
                ) from e
    return out


# --- the public surface -----------------------------------------------------
def _embed(texts: Sequence[str], prefix: str) -> List[List[float]]:
    if not texts:
        return []
    prepared = [f"{prefix}{t}" for t in texts] if prefix else list(texts)

    vectors = _embed_ollama(prepared) if provider() == OLLAMA else _embed_fastembed(prepared)

    # A dimension mismatch means the configured model and EMBED_DIM disagree. Failing at
    # the first write beats writing vectors Qdrant rejects one batch at a time.
    if vectors and len(vectors[0]) != settings.EMBED_DIM:
        raise RuntimeError(
            f"'{model_name()}' produced {len(vectors[0])}d vectors but EMBED_DIM is "
            f"{settings.EMBED_DIM}. Fix the config before indexing."
        )
    return vectors


def embed_documents(texts: Sequence[str]) -> List[List[float]]:
    """Embed passages for storage."""
    return _embed(texts, _prefixes()[0])


def embed_query(text: str) -> List[float]:
    """Embed one question for search."""
    vectors = _embed([text], _prefixes()[1])
    return vectors[0] if vectors else []


def reachable() -> tuple:
    """
    `(ok, detail)` — whether the configured provider can actually answer right now.

    Used by `/health` so an operator sees "the GPU endpoint is asleep" instead of
    discovering it when the first crawl of the day fails.
    """
    try:
        vector = embed_query("probe")
        if len(vector) != dimension():
            return False, f"{describe()} returned {len(vector)}d, expected {dimension()}d"
        return True, ""
    except Exception as e:
        return False, str(e)


def reset_model() -> None:
    """Drop cached clients so the next call re-reads config. Used by tests."""
    global _model, _client
    with _lock:
        _model = None
        _client = None


# Kept so existing call sites and tests that reached for the fastembed handle still work.
def get_model():
    return _fastembed_model()
