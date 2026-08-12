"""
Pluggable embedding engines — local CPU model or GPU-hosted model, chosen per run.

WHY TWO ENGINES
---------------
The platform's retrieval quality is bounded by its embedding model. BGE-large (335M
params, 1024 dims) runs inside the backend process on CPU: offline, and it never
fails. A larger model served from a GPU separates related-but-distinct concepts more
sharply ("database migration" vs "database administration"), but it lives behind a
network call that can be slow or absent.

Rather than choose once, both are available and selectable per operation:

    bge  — local, in-process, 1024 dims. Default. No network, no outage mode.
    gpu  — remote OpenAI/Ollama-compatible endpoint, 768 dims. Higher fidelity.

THE CONSTRAINT THAT SHAPES EVERYTHING
-------------------------------------
Vectors from different models are **not comparable** — different dimensionality, and
even at equal size they describe different spaces. A query embedded with one model
cannot search an index built with the other; doing so returns confident nonsense.
So the engine is part of the index's identity: each engine owns a separate index, and
`corpus_index_service` keys its storage directory on `engine.name`. Switching engine
switches index; it never mixes.

FALLBACK
--------
If the GPU engine is selected but its endpoint is unreachable, callers fall back to
the local engine and report which engine actually served the request — never a silent
substitution, because a silent one would search the wrong index.

Reverting to BGE-only is a single config value (`EMBEDDING_ENGINE=bge`); the local
index is always maintained and never discarded.
"""
from __future__ import annotations

import threading
from typing import List, Optional, Protocol

import numpy as np

from app.core.config import settings
from app.core.logging import logger


class EmbeddingEngine(Protocol):
    """What retrieval needs from an embedding model, regardless of where it runs."""

    name: str
    dimension: int
    # Cosine floor below which a chunk is noise FOR THIS MODEL. Every model has its own
    # similarity distribution, so a threshold calibrated on one is meaningless on
    # another — reusing BGE's 0.62 on nomic silently discarded genuine matches.
    min_similarity: float

    def embed_documents(self, texts: List[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def available(self) -> bool: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise so inner product on a FAISS IndexFlatIP equals cosine similarity."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class LocalFastEmbedEngine:
    """
    Any FastEmbed (ONNX) model, running in-process on CPU.

    One class serves every local model because they differ only in weights, width and
    calibrated floor — all data, not behaviour.

    BATCH SIZE IS PART OF THE MODEL'S IDENTITY, not a performance knob. FastEmbed
    defaults to 256, which a 335M-parameter model at 512 tokens cannot hold: mxbai-large
    asked ONNX Runtime for a 1.15 GB activation buffer and died with a BFCArena
    allocation failure. Each model therefore declares a batch size it can actually run.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        dimension: int,
        min_similarity: float,
        batch_size: int = 64,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.dimension = dimension
        self._min_similarity = min_similarity
        self.batch_size = batch_size
        self._model: Optional[object] = None
        self._lock = threading.Lock()

    @property
    def min_similarity(self) -> float:
        return self._min_similarity

    def _embedder(self):
        """Loaded once, on first use — a 0.64 GB model must not cost every process start."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    logger.info(f"Loading local embedding model '{self.model_name}' (FastEmbed/ONNX)...")
                    self._model = TextEmbedding(self.model_name)
                    logger.info(f"Embedding model '{self.model_name}' ready ({self.dimension} dims).")
        return self._model

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vectors = list(self._embedder().embed(texts, batch_size=self.batch_size))
        return _normalize(np.asarray(vectors, dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        vector = next(iter(self._embedder().query_embed([text])))
        return _normalize(np.asarray([vector], dtype="float32"))

    def available(self) -> bool:
        return True  # local and offline; nothing to be unreachable


class LocalBgeEngine(LocalFastEmbedEngine):
    """
    BAAI/bge-large-en-v1.5 — the default local engine.

    Upgraded from bge-small-en-v1.5 (384d) to bge-large-en-v1.5 (1024d) for sharper
    semantic separation. Every existing index must be rebuilt after this change.
    """

    def __init__(self) -> None:
        from app.core.constants import EMBEDDING_MODEL_NAME

        super().__init__(
            name="bge",
            model_name=EMBEDDING_MODEL_NAME,
            dimension=1024,
            # Measured floor updated for BGE-large-en-v1.5 (1024 dims).
            min_similarity=float(settings.RETRIEVAL_MIN_SIMILARITY),
            batch_size=32,
        )

    @property
    def min_similarity(self) -> float:
        # Read live so a config change takes effect without a restart.
        return float(settings.RETRIEVAL_MIN_SIMILARITY)


class GpuEndpointEngine:
    """
    A model served from an Ollama-compatible endpoint (`/api/embed`).

    Ollama exposes embeddings on its native route; the OpenAI-compatible
    `/v1/embeddings` path is only present when the server was started with embeddings
    enabled, so the native route is used and the OpenAI one is tried as a fallback.
    """

    def __init__(self, base_url: str, model: str, dimension: int, timeout: float) -> None:
        self.name = "gpu"
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.dimension = dimension
        self.timeout = timeout
        # Measured on nomic-embed-text: relevant pairs >= 0.592, unrelated <= 0.426 —
        # a +0.165 separation, 2.4x wider than BGE's. Its absolute values sit lower, so
        # BGE's floor would discard genuine matches; this floor sits in ITS gap.
        self.min_similarity = float(settings.EMBEDDING_GPU_MIN_SIMILARITY)
        self._available: Optional[bool] = None
        self._lock = threading.Lock()

    # --- transport --------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        import httpx

        response = httpx.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        # Native Ollama batch route.
        try:
            data = self._post("/api/embed", {"model": self.model, "input": texts})
            vectors = data.get("embeddings")
            if vectors:
                return _normalize(np.asarray(vectors, dtype="float32"))
        except Exception as e:
            logger.debug(f"/api/embed unavailable ({e}); trying the OpenAI-compatible route.")

        # OpenAI-compatible route, when the server was started with embeddings on.
        data = self._post("/v1/embeddings", {"model": self.model, "input": texts})
        vectors = [item["embedding"] for item in data["data"]]
        return _normalize(np.asarray(vectors, dtype="float32"))

    # --- interface --------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        # Chunked so a large corpus does not build one enormous request.
        batch = max(int(settings.EMBEDDING_GPU_BATCH), 1)
        out: List[np.ndarray] = []
        for start in range(0, len(texts), batch):
            out.append(self._embed(texts[start:start + batch]))
        return np.vstack(out) if out else np.zeros((0, self.dimension), dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])

    def available(self) -> bool:
        """Cached reachability probe — one embedding round trip."""
        with self._lock:
            if self._available is None:
                try:
                    vec = self._embed(["ping"])
                    self._available = bool(len(vec)) and vec.shape[1] == self.dimension
                    if len(vec) and vec.shape[1] != self.dimension:
                        logger.error(
                            f"GPU embedding model '{self.model}' returned {vec.shape[1]} dims but "
                            f"EMBEDDING_GPU_DIM is {self.dimension}. Fix the config before indexing."
                        )
                except Exception as e:
                    logger.warning(f"GPU embedding endpoint unreachable: {e}")
                    self._available = False
            return self._available

    def reset_availability(self) -> None:
        with self._lock:
            self._available = None


class LocalMxbaiEngine(LocalFastEmbedEngine):
    """
    mixedbread-ai/mxbai-embed-large-v1 — 1024 dims, 0.64 GB, in-process on CPU.

    Added as a THIRD comparable engine rather than a replacement: it is ten times the
    size of BGE-large (mxbai at 1024d too, though different architecture) and measurably slower (~32 chunks/s on a laptop CPU vs BGE's
    hundreds), but it is local, so unlike the GPU engine it can always index. That
    matters for a fair comparison — a model that cannot index when the endpoint is down
    cannot be compared on the same resumes.

    THE FLOOR IS MEASURED, NOT GUESSED. Cosine values are not comparable across models,
    so a shared threshold would hand the win to whichever model happens to score higher
    in absolute terms. 0.537 is the value at which mxbai admits the same FRACTION of
    query-chunk pairs that BGE admits at 0.62, measured over 279 chunks of this
    project's real resumes with 12 recruiter-style probes. Equal selectivity is what
    makes the three-way comparison about ranking quality instead of threshold luck.
    """

    def __init__(self) -> None:
        super().__init__(
            name="mxbai",
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            dimension=1024,
            min_similarity=float(settings.EMBEDDING_MXBAI_MIN_SIMILARITY),
            # 8, not the library default of 256: see LocalFastEmbedEngine.
            batch_size=int(settings.EMBEDDING_MXBAI_BATCH),
        )


_LOCAL = LocalBgeEngine()
_MXBAI = LocalMxbaiEngine()
_gpu_singleton: Optional[GpuEndpointEngine] = None
_gpu_lock = threading.Lock()


def _gpu_engine() -> Optional[GpuEndpointEngine]:
    global _gpu_singleton
    if not settings.EMBEDDING_GPU_URL or not settings.EMBEDDING_GPU_MODEL:
        return None
    with _gpu_lock:
        if _gpu_singleton is None:
            _gpu_singleton = GpuEndpointEngine(
                base_url=settings.EMBEDDING_GPU_URL,
                model=settings.EMBEDDING_GPU_MODEL,
                dimension=int(settings.EMBEDDING_GPU_DIM),
                timeout=float(settings.EMBEDDING_GPU_TIMEOUT_SECONDS),
            )
        return _gpu_singleton


def resolve_engine(requested: Optional[str] = None) -> EmbeddingEngine:
    """
    Return the engine to use, falling back to local when the GPU one cannot serve.

    `requested` overrides `settings.EMBEDDING_ENGINE` for a single operation, which is
    what lets a recruiter choose per run. Anything unrecognised resolves to local
    rather than raising — retrieval must never hard-fail on a config typo.
    """
    choice = (requested or settings.EMBEDDING_ENGINE or "bge").strip().lower()
    if choice in ("bge", "local", "cpu", ""):
        return _LOCAL

    if choice in ("mxbai", "mixedbread", "large"):
        return _MXBAI

    if choice in ("gpu", "remote", "nomic"):
        engine = _gpu_engine()
        if engine is None:
            logger.warning("GPU embedding engine requested but not configured; using local BGE.")
            return _LOCAL
        if not engine.available():
            logger.warning("GPU embedding endpoint unreachable; falling back to local BGE.")
            return _LOCAL
        return engine

    logger.warning(f"Unknown embedding engine '{choice}'; using local BGE.")
    return _LOCAL


# Every engine the platform can index or search with, in display order.
ENGINE_NAMES = ("bge", "mxbai", "gpu")


def all_engines() -> List[EmbeddingEngine]:
    """Every configured engine — the set an "index with all models" run covers."""
    engines: List[EmbeddingEngine] = [_LOCAL, _MXBAI]
    gpu = _gpu_engine()
    if gpu is not None:
        engines.append(gpu)
    return engines


def engine_status() -> dict:
    """
    Report every engine for the UI, without forcing a probe on the local ones.

    `available` is what the indexing screen needs: a model that cannot serve right now
    cannot be indexed with, and the recruiter must be able to see that rather than
    discover it as a silent gap in coverage later.
    """
    gpu = _gpu_engine()
    return {
        "configured_default": (settings.EMBEDDING_ENGINE or "bge").lower(),
        "engines": [
            {"name": "bge", "model": _LOCAL.model_name, "dimension": _LOCAL.dimension,
             "location": "in-process (CPU)", "available": True, "configured": True,
             "min_similarity": _LOCAL.min_similarity},
            {"name": "mxbai", "model": _MXBAI.model_name, "dimension": _MXBAI.dimension,
             "location": "in-process (CPU)", "available": True, "configured": True,
             "min_similarity": _MXBAI.min_similarity},
            {"name": "gpu", "model": gpu.model if gpu else None,
             "dimension": gpu.dimension if gpu else None,
             "location": gpu.base_url if gpu else None,
             "available": bool(gpu and gpu.available()),
             "configured": gpu is not None,
             "min_similarity": gpu.min_similarity if gpu else None},
        ],
    }
