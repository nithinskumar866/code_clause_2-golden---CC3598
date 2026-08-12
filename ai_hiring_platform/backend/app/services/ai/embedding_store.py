"""
The one place resume vectors live — shared by ranking, analysis and the chatbot.

WHY THIS REPLACES TWO SEPARATE INDEXES
--------------------------------------
The platform used to embed every resume twice, into two unrelated structures:

    vector_store_service   one FAISS index PER RESUME, hardcoded to BGE (d=384),
                           used by AI Analysis and Ranking.
    corpus_index_service   one pool-wide index PER ENGINE, used by the chatbot.

Three problems followed from that, and all three are the same problem:

  1. **Duplicated work.** Each resume was embedded twice with BGE. A third model would
     have meant six passes per resume.
  2. **The model selector was a lie outside the chatbot.** Analysis and Ranking could
     only ever run on BGE, because their index hardcoded 384 dimensions.
  3. **Models silently diverged.** 300 resumes uploaded while the GPU endpoint was down
     entered the BGE index and nothing else, so the two models were answering questions
     about different talent pools — and nothing in the system said so.

THE SHAPE THAT FIXES IT
-----------------------
Parsing and chunking are **model-independent**. So they happen once, into a document
layer that every model shares:

    documents.db     resumes (fingerprint + profile) and chunks (section, page, text)
    vectors-<model>.faiss + rows-<model>.json    one vector set per model, same chunks

That single fact is what makes a fair three-way comparison possible at all: every model
is asked to index *the same chunk set*, so a difference in results is a difference in
the model rather than in what it was shown. `coverage()` reports each model's share of
that set, so a gap is visible instead of silent.

Indexing is **explicit and per model**. When the GPU endpoint is down you index the two
local models and carry on; the remote one is filled in later and the store tells you it
is behind until you do. Nothing is embedded implicitly at upload, because an implicit
step is exactly what went wrong before — it succeeded for one model, failed for another,
and said nothing.

Retrieval is one operation with an optional filter:

    search(engine, query, top_k)                  -> the whole pool   (chatbot)
    search(engine, query, top_k, resume_ids={7})  -> one candidate    (analysis, ranking)

A single-resume search scores that resume's own rows directly rather than taking a
global top-N and filtering, because a global cut can return nothing for the candidate
being evaluated — which is how an evaluation silently loses its evidence.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import faiss
import numpy as np
from sqlalchemy.orm import Session

from app.core.constants import RESUME_UPLOAD_DIR, VECTOR_STORE_DIR
from app.core.logging import logger
from app.models.database import Resume
from app.services.ai import (
    document_loader,
    embedding_engines,
    resume_structuring_service,
    vector_store_service,
)

# --- Layout -----------------------------------------------------------------
def store_dir() -> str:
    path = os.path.join(VECTOR_STORE_DIR, "shared")
    os.makedirs(path, exist_ok=True)
    return path


def _documents_db() -> str:
    return os.path.join(store_dir(), "documents.db")


def _vectors_file(model: str) -> str:
    return os.path.join(store_dir(), f"vectors-{model}.faiss")


def _rows_file(model: str) -> str:
    return os.path.join(store_dir(), f"rows-{model}.json")


# `ordinal` preserves the order chunks came out of the structuring pass, so a resume
# reads in document order regardless of how rows ended up laid out in any model's index.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS resumes (
    resume_id   INTEGER PRIMARY KEY,
    filename    TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    profile     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id  INTEGER NOT NULL,
    ordinal    INTEGER NOT NULL,
    section    TEXT NOT NULL,
    page       INTEGER NOT NULL,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_resume ON chunks(resume_id);
"""

_LOCK = threading.RLock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_documents_db())
    conn.executescript(_SCHEMA)
    return conn


# --- Document layer ---------------------------------------------------------
@dataclass
class DocumentSyncResult:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    total_resumes: int = 0
    total_chunks: int = 0
    failed: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added, "updated": self.updated, "removed": self.removed,
            "unchanged": self.unchanged, "total_resumes": self.total_resumes,
            "total_chunks": self.total_chunks, "failed": self.failed,
        }


def _resume_disk_path(resume: Resume) -> str:
    """Uploads are stored as `<id>_<original filename>` (see services/resume.py)."""
    return os.path.join(RESUME_UPLOAD_DIR, f"{resume.id}_{resume.filename}")


def sync_documents(
    db: Session, force: bool = False, resume_ids: Optional[Sequence[int]] = None
) -> DocumentSyncResult:
    """
    Parse and chunk any resume the document layer does not already hold.

    Content-fingerprinted, so a re-uploaded or edited file is re-chunked and an
    unchanged one costs nothing. This runs BEFORE any model indexes, and it is the
    reason every model sees an identical chunk set.

    `resume_ids` narrows the pass to specific resumes. Evaluating one candidate must not
    fingerprint all 600 files on disk to discover that nothing else changed.
    """
    from app.services.ai.corpus_index_service import _profile_for  # deterministic profile

    result = DocumentSyncResult()
    with _LOCK:
        conn = _connect()
        try:
            known = {
                int(rid): fp
                for rid, fp in conn.execute("SELECT resume_id, fingerprint FROM resumes")
            }

            query = db.query(Resume)
            if resume_ids is not None:
                wanted_ids = list(resume_ids)
                if not wanted_ids:
                    return result
                query = query.filter(Resume.id.in_(wanted_ids))

            wanted: Dict[int, Tuple[str, str]] = {}
            for resume in query.all():
                path = _resume_disk_path(resume)
                if not os.path.exists(path):
                    logger.warning(f"Resume {resume.id} has no file at {path}; skipped.")
                    continue
                wanted[resume.id] = (path, resume.filename)

            # A narrowed pass says nothing about resumes it did not look at, so it must
            # never conclude they were deleted.
            removed = [] if resume_ids is not None else [rid for rid in known if rid not in wanted]
            to_parse: List[int] = []
            for rid, (path, _fn) in wanted.items():
                fingerprint = vector_store_service.compute_fingerprint(path)
                if force or known.get(rid) != fingerprint:
                    to_parse.append(rid)

            result.unchanged = len(wanted) - len(to_parse)
            result.removed = len(removed)

            with conn:
                for rid in removed:
                    conn.execute("DELETE FROM chunks WHERE resume_id = ?", (rid,))
                    conn.execute("DELETE FROM resumes WHERE resume_id = ?", (rid,))

                for rid in to_parse:
                    path, filename = wanted[rid]
                    try:
                        text = document_loader.load_document(path)
                        nodes = resume_structuring_service.structure_resume_to_nodes(
                            text=text, candidate_id=rid, resume_id=rid, filename=filename
                        )
                        profile = _profile_for(rid, path, text, filename)
                    except Exception as e:  # one broken file must not sink the batch
                        logger.error(f"Document sync failed for resume {rid} ({filename}): {e}",
                                     exc_info=True)
                        result.failed.append(filename)
                        continue

                    conn.execute("DELETE FROM chunks WHERE resume_id = ?", (rid,))
                    conn.executemany(
                        "INSERT INTO chunks (resume_id, ordinal, section, page, text)"
                        " VALUES (?,?,?,?,?)",
                        [
                            (rid, i, n.metadata.get("section", "Summary"),
                             int(n.metadata.get("page", 1)), n.text.strip())
                            for i, n in enumerate(nodes)
                        ],
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO resumes (resume_id, filename, fingerprint, profile)"
                        " VALUES (?,?,?,?)",
                        (rid, filename, profile.get("fingerprint", ""), json.dumps(profile)),
                    )
                    if rid in known:
                        result.updated += 1
                    else:
                        result.added += 1

            result.total_resumes = conn.execute("SELECT COUNT(*) FROM resumes").fetchone()[0]
            result.total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()

    logger.info(f"Document sync: {result.as_dict()}")
    return result


def read_chunks(resume_ids: Optional[Iterable[int]] = None) -> List[Dict[str, Any]]:
    """Every stored chunk, in (resume_id, ordinal) order."""
    conn = _connect()
    try:
        sql = ("SELECT chunk_id, resume_id, ordinal, section, page, text FROM chunks")
        params: Sequence[Any] = ()
        if resume_ids is not None:
            ids = list(resume_ids)
            if not ids:
                return []
            sql += f" WHERE resume_id IN ({','.join('?' * len(ids))})"
            params = ids
        sql += " ORDER BY resume_id, ordinal"
        rows = conn.execute(sql, params).fetchall()
        names = dict(conn.execute("SELECT resume_id, filename FROM resumes"))
    finally:
        conn.close()
    return [
        {"chunk_id": r[0], "resume_id": r[1], "ordinal": r[2], "section": r[3],
         "page": r[4], "text": r[5], "filename": names.get(r[1], "")}
        for r in rows
    ]


def read_profiles() -> Dict[int, Dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT resume_id, profile FROM resumes").fetchall()
    finally:
        conn.close()
    return {int(r[0]): json.loads(r[1]) for r in rows}


def document_totals() -> Dict[str, int]:
    conn = _connect()
    try:
        return {
            "resumes": conn.execute("SELECT COUNT(*) FROM resumes").fetchone()[0],
            "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        }
    finally:
        conn.close()


# --- One model's vectors ----------------------------------------------------
class ModelIndex:
    """
    A single model's view of the shared chunk set.

    `row_chunk_ids[i]` is the chunk at FAISS row `i`, and `chunks[i]` is that chunk's
    record — the two are kept in lockstep so a search result maps straight to text
    without a second lookup.
    """

    def __init__(self, model: str, engine: Any, index: Optional[faiss.Index],
                 chunks: List[Dict[str, Any]]) -> None:
        self.model = model
        self.engine = engine
        self.index = index
        self.chunks = chunks
        self._rows_by_resume: Optional[Dict[int, List[int]]] = None
        self._resumes: Optional[Dict[int, Dict[str, Any]]] = None
        self._bm25: Optional[Any] = None
        self._lexicon: Optional[Any] = None

    @property
    def dimension(self) -> int:
        return int(getattr(self.engine, "dimension", 0))

    @property
    def engine_name(self) -> str:
        return str(getattr(self.engine, "name", self.model))

    def is_empty(self) -> bool:
        return self.index is None or not self.chunks

    # --- Pool-level views -------------------------------------------------
    # The chatbot needs three things beyond raw vectors: who the candidates ARE, a
    # sparse index for exact term matches, and the corpus lexicon. All three derive
    # from the shared document layer this index already points at, so they live here
    # rather than in a parallel structure — a second copy is what let the chatbot and
    # the rest of the platform drift onto different pools in the first place.

    @property
    def resumes(self) -> Dict[int, Dict[str, Any]]:
        """{resume_id -> deterministic profile record} for everything this model holds."""
        if self._resumes is None:
            indexed = self.indexed_resume_ids
            self._resumes = {
                rid: profile for rid, profile in read_profiles().items() if rid in indexed
            }
        return self._resumes

    @property
    def bm25(self) -> Any:
        """
        Sparse statistics over this index's chunks, computed once and cached.

        Row numbers are shared with the dense index, so a fused ranking needs no
        translation between the two.
        """
        from app.services.ai.corpus_index_service import _Bm25Stats

        if self._bm25 is None:
            logger.info(f"Precomputing BM25 statistics over {len(self.chunks)} chunks [{self.model}]...")
            self._bm25 = _Bm25Stats([c["text"] for c in self.chunks])
        return self._bm25

    @property
    def lexicon(self) -> Any:
        """
        What this pool's own text says its words mean — skills, places, people.

        Built lazily from the chunk text already in memory; see `chat_lexicon` for why
        a corpus-relative answer is the only correct one.
        """
        from app.services.ai.chat_lexicon import build_lexicon

        if self._lexicon is None:
            self._lexicon = build_lexicon(self.chunks, list(self.resumes.values()))
        return self._lexicon

    @property
    def rows_by_resume(self) -> Dict[int, List[int]]:
        if self._rows_by_resume is None:
            mapping: Dict[int, List[int]] = {}
            for row, chunk in enumerate(self.chunks):
                mapping.setdefault(chunk["resume_id"], []).append(row)
            self._rows_by_resume = mapping
        return self._rows_by_resume

    @property
    def indexed_resume_ids(self) -> Set[int]:
        return set(self.rows_by_resume)

    def search(
        self, query: str, top_k: int, resume_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank chunk rows against a query. Returns [(row, cosine)].

        With `resume_ids` the candidates' own rows are scored EXHAUSTIVELY rather than
        filtered out of a global top-N. For a single resume that is both exact and
        cheaper (~15 dot products), and it cannot return an empty result just because
        stronger chunks exist elsewhere in the pool — which is what made a per-candidate
        evaluation lose its evidence.
        """
        if self.is_empty():
            return []
        vector = np.asarray(self.engine.embed_query(query), dtype="float32").reshape(1, -1)

        if resume_ids is not None:
            rows = [r for rid in resume_ids for r in self.rows_by_resume.get(rid, [])]
            if not rows:
                return []
            matrix = np.asarray(
                [self.index.reconstruct(int(r)) for r in rows], dtype="float32"
            )
            scores = matrix @ vector.reshape(-1)
            ranked = sorted(zip(rows, scores.tolist()), key=lambda p: p[1], reverse=True)
            return [(int(r), float(s)) for r, s in ranked[:top_k]]

        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return []
        sims, ids = self.index.search(vector, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0]

    def vectors_for_rows(self, rows: Sequence[int]) -> np.ndarray:
        return np.asarray([self.index.reconstruct(int(r)) for r in rows], dtype="float32")


def _read_rows(model: str) -> Tuple[List[int], Dict[str, Any]]:
    path = _rows_file(model)
    if not os.path.exists(path):
        return [], {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return [int(c) for c in payload.get("chunk_ids", [])], payload
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Row map for '{model}' unreadable ({e}); treating as unindexed.")
        return [], {}


def _write_rows(model: str, chunk_ids: Sequence[int], dimension: int) -> None:
    with open(_rows_file(model), "w", encoding="utf-8") as f:
        json.dump(
            {"model": model, "dimension": dimension, "chunk_ids": list(chunk_ids)},
            f,
        )


# One loaded index per model, reused across requests.
_CACHE: Dict[str, ModelIndex] = {}


def invalidate(model: Optional[str] = None) -> None:
    with _LOCK:
        if model:
            _CACHE.pop(model, None)
        else:
            _CACHE.clear()


def load_index(model: Optional[str] = None) -> ModelIndex:
    """
    Load a model's index, from the in-process cache when possible.

    Never triggers indexing: a screen that reads the store must not start a five-minute
    embedding run as a side effect. `index_model` is the only thing that embeds.
    """
    engine = embedding_engines.resolve_engine(model)
    name = engine.name
    with _LOCK:
        cached = _CACHE.get(name)
        if cached is not None:
            return cached

        chunk_ids, _meta = _read_rows(name)
        if not chunk_ids or not os.path.exists(_vectors_file(name)):
            empty = ModelIndex(name, engine, None, [])
            _CACHE[name] = empty
            return empty

        by_id = {c["chunk_id"]: c for c in read_chunks()}
        # A chunk deleted since this model indexed leaves a hole; drop those rows rather
        # than let row numbers drift out of step with the vectors.
        ordered = [by_id[cid] for cid in chunk_ids if cid in by_id]
        index = faiss.read_index(_vectors_file(name))
        if len(ordered) != index.ntotal:
            logger.warning(
                f"Index '{name}' holds {index.ntotal} vectors but {len(ordered)} chunks "
                f"still exist; re-index this model to realign."
            )
        loaded = ModelIndex(name, engine, index, ordered)
        _CACHE[name] = loaded
        logger.info(f"Loaded '{name}' index: {index.ntotal} vectors, {len(ordered)} chunks.")
        return loaded


# --- Indexing ---------------------------------------------------------------
@dataclass
class IndexProgress:
    """Live state of one model's indexing run, polled by the UI."""

    model: str
    state: str = "idle"          # idle | running | done | error
    done: int = 0
    total: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    error: Optional[str] = None
    embedded: int = 0
    reused: int = 0

    def as_dict(self) -> Dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - self.started_at if self.started_at else 0.0
        return {
            "model": self.model, "state": self.state, "done": self.done,
            "total": self.total, "embedded": self.embedded, "reused": self.reused,
            "elapsed_seconds": round(elapsed, 1), "error": self.error,
            "percent": int(round(100 * self.done / self.total)) if self.total else 0,
        }


_PROGRESS: Dict[str, IndexProgress] = {}


def progress(model: Optional[str] = None) -> Dict[str, Any]:
    with _LOCK:
        if model:
            entry = _PROGRESS.get(model)
            return entry.as_dict() if entry else IndexProgress(model=model).as_dict()
        return {name: p.as_dict() for name, p in _PROGRESS.items()}


def index_model(model: str, batch: int = 0, on_progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """
    Bring one model's vectors in line with the shared chunk set.

    Incremental: chunks this model already holds keep their existing vectors, so adding
    300 resumes costs 300 resumes' worth of embedding and not the whole pool. Chunks the
    document layer no longer has are dropped.
    """
    engine = embedding_engines.resolve_engine(model)
    name = engine.name
    if name != (model or name).lower() and model not in ("", None):
        # resolve_engine falls back to BGE when a remote engine is unreachable. Indexing
        # BGE while the operator asked for the GPU model would quietly leave the
        # requested model empty and report success — refuse instead.
        raise RuntimeError(
            f"Embedding engine '{model}' is not available right now (resolved to "
            f"'{name}'). Index it when the endpoint is reachable."
        )

    tracker = IndexProgress(model=name, state="running", started_at=time.time())
    with _LOCK:
        _PROGRESS[name] = tracker

    try:
        chunks = read_chunks()
        wanted_ids = [c["chunk_id"] for c in chunks]
        tracker.total = len(wanted_ids)

        existing_ids, _meta = _read_rows(name)
        reusable: Dict[int, np.ndarray] = {}
        if existing_ids and os.path.exists(_vectors_file(name)):
            old = faiss.read_index(_vectors_file(name))
            if old.d == engine.dimension:
                keep = set(wanted_ids)
                for row, chunk_id in enumerate(existing_ids):
                    if chunk_id in keep and row < old.ntotal:
                        reusable[chunk_id] = old.reconstruct(int(row))
            else:
                logger.warning(
                    f"Existing '{name}' index has {old.d} dims, engine reports "
                    f"{engine.dimension}; rebuilding from scratch."
                )

        tracker.reused = len(reusable)
        missing = [c for c in chunks if c["chunk_id"] not in reusable]
        tracker.done = len(reusable)
        if on_progress:
            on_progress(tracker.done, tracker.total)

        step = batch or max(int(getattr(engine, "batch_size", 32)), 1) * 4
        fresh: Dict[int, np.ndarray] = {}
        for start in range(0, len(missing), step):
            window = missing[start:start + step]
            vectors = engine.embed_documents([c["text"] for c in window])
            for chunk, vector in zip(window, vectors):
                fresh[chunk["chunk_id"]] = vector
            tracker.done += len(window)
            tracker.embedded += len(window)
            if on_progress:
                on_progress(tracker.done, tracker.total)

        matrix = np.asarray(
            [reusable.get(cid, fresh.get(cid)) for cid in wanted_ids], dtype="float32"
        ) if wanted_ids else np.zeros((0, engine.dimension), dtype="float32")

        index = faiss.IndexFlatIP(engine.dimension)
        if len(matrix):
            index.add(matrix)
        faiss.write_index(index, _vectors_file(name))
        _write_rows(name, wanted_ids, engine.dimension)
        invalidate(name)

        tracker.state = "done"
        tracker.finished_at = time.time()
        logger.info(
            f"Indexed '{name}': {tracker.embedded} embedded, {tracker.reused} reused, "
            f"{index.ntotal} vectors total in {tracker.as_dict()['elapsed_seconds']}s."
        )
        return tracker.as_dict()
    except Exception as e:
        tracker.state = "error"
        tracker.error = str(e)
        tracker.finished_at = time.time()
        logger.error(f"Indexing '{name}' failed: {e}", exc_info=True)
        raise


def index_models_async(models: Sequence[str], db: Session, sync_first: bool = True) -> Dict[str, Any]:
    """
    Kick off indexing for several models on a worker thread.

    A full pass over 600 resumes takes minutes for the larger models, which is far too
    long for a request to hold open — so the call returns immediately and the UI polls
    `progress()`. Models run one after another rather than in parallel: they compete for
    the same CPU, and a 0.64 GB model running beside another is how the ONNX allocator
    runs out of memory.
    """
    requested = [m for m in models if m]
    if not requested:
        raise ValueError("Select at least one model to index.")

    if sync_first:
        sync_documents(db)

    with _LOCK:
        for name in requested:
            _PROGRESS[name] = IndexProgress(model=name, state="queued")

    def worker() -> None:
        for name in requested:
            try:
                index_model(name)
            except Exception:
                continue  # the tracker already records the failure for this model

    threading.Thread(target=worker, daemon=True, name="embedding-indexer").start()
    return {"queued": requested, "documents": document_totals()}


def store_document(resume_id: int, path: str, filename: str, force: bool = False) -> bool:
    """
    Parse, chunk and record ONE resume from a file path. Returns True if it changed.

    Path-based rather than DB-based so the evaluation agents can guarantee a candidate
    is present without taking a database session — agents orchestrate services, they do
    not query tables.
    """
    from app.services.ai.corpus_index_service import _profile_for

    fingerprint = vector_store_service.compute_fingerprint(path)
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT fingerprint FROM resumes WHERE resume_id = ?", (resume_id,)
            ).fetchone()
            if row and row[0] == fingerprint and not force:
                return False

            text = document_loader.load_document(path)
            nodes = resume_structuring_service.structure_resume_to_nodes(
                text=text, candidate_id=resume_id, resume_id=resume_id, filename=filename
            )
            profile = _profile_for(resume_id, path, text, filename)

            with conn:
                conn.execute("DELETE FROM chunks WHERE resume_id = ?", (resume_id,))
                conn.executemany(
                    "INSERT INTO chunks (resume_id, ordinal, section, page, text) VALUES (?,?,?,?,?)",
                    [
                        (resume_id, i, n.metadata.get("section", "Summary"),
                         int(n.metadata.get("page", 1)), n.text.strip())
                        for i, n in enumerate(nodes)
                    ],
                )
                conn.execute(
                    "INSERT OR REPLACE INTO resumes (resume_id, filename, fingerprint, profile)"
                    " VALUES (?,?,?,?)",
                    (resume_id, filename, fingerprint, json.dumps(profile)),
                )
        finally:
            conn.close()
    logger.info(f"Stored document for resume {resume_id} ({filename}).")
    return True


def ensure_resume_searchable(
    resume_id: int, path: str, filename: str, model: Optional[str] = None
) -> ModelIndex:
    """
    Guarantee one resume is searchable by one model, on demand.

    Bulk indexing is explicit and operator-driven, but evaluating a single candidate is
    itself an explicit act — and it cannot proceed without that candidate's vectors. So
    this fills exactly the gap in front of it and nothing else in the pool is touched.
    """
    engine = embedding_engines.resolve_engine(model)
    changed = store_document(resume_id, path, filename)
    if changed:
        invalidate(engine.name)

    index = load_index(engine.name)
    if not changed and resume_id in index.indexed_resume_ids:
        return index

    logger.info(f"Resume {resume_id} needs vectors for '{engine.name}'; indexing now.")
    index_model(engine.name)
    return load_index(engine.name)


def ensure_indexed(db: Session, resume_id: int, model: Optional[str] = None) -> ModelIndex:
    """
    Guarantee one resume is searchable by one model, on demand.

    Bulk indexing is explicit and operator-driven, but evaluating a single candidate is
    itself an explicit act — and it cannot proceed without that candidate's vectors. So
    this fills exactly the gap in front of it: parse the one resume if the document layer
    lacks it, then top up that model. Nothing else in the pool is touched.
    """
    engine = embedding_engines.resolve_engine(model)
    sync_documents(db, resume_ids=[resume_id])

    index = load_index(engine.name)
    if resume_id in index.indexed_resume_ids:
        return index

    logger.info(f"Resume {resume_id} is not in the '{engine.name}' index yet; indexing it now.")
    index_model(engine.name)
    return load_index(engine.name)


# --- Coverage ---------------------------------------------------------------
def coverage(db: Optional[Session] = None) -> Dict[str, Any]:
    """
    What each model actually holds, against the shared document set.

    This is the screen that makes a silent divergence impossible. `comparable_resumes`
    is the intersection across available models — the only set on which a three-way
    comparison is honest, because it is the only set every model has seen.
    """
    totals = document_totals()
    database_resumes = db.query(Resume).count() if db is not None else None

    models: List[Dict[str, Any]] = []
    indexed_sets: List[Set[int]] = []
    for name in embedding_engines.ENGINE_NAMES:
        chunk_ids, meta = _read_rows(name)
        resumes_with_vectors: Set[int] = set()
        if chunk_ids:
            by_id = {c["chunk_id"]: c["resume_id"] for c in read_chunks()}
            resumes_with_vectors = {by_id[c] for c in chunk_ids if c in by_id}
        models.append({
            "model": name,
            "indexed_resumes": len(resumes_with_vectors),
            "indexed_chunks": len(chunk_ids),
            "dimension": meta.get("dimension"),
            "in_sync": len(chunk_ids) == totals["chunks"] and totals["chunks"] > 0,
            "progress": progress(name),
        })
        if resumes_with_vectors:
            indexed_sets.append(resumes_with_vectors)

    comparable = set.intersection(*indexed_sets) if indexed_sets else set()
    return {
        "documents": totals,
        "database_resumes": database_resumes,
        "documents_in_sync": database_resumes is None or database_resumes == totals["resumes"],
        "models": models,
        "comparable_resumes": len(comparable),
        # True only when every model that holds anything holds the SAME resumes.
        "models_aligned": all(
            m["indexed_resumes"] in (0, len(comparable)) for m in models
        ) and len(indexed_sets) > 0,
    }


def comparable_resume_ids(models: Sequence[str]) -> Set[int]:
    """
    Resumes every one of these models has indexed.

    A model comparison run over anything wider is not a comparison — it rewards
    whichever model happens to have seen more resumes.
    """
    sets: List[Set[int]] = []
    all_chunks = read_chunks()
    by_id = {c["chunk_id"]: c["resume_id"] for c in all_chunks}
    for name in models:
        chunk_ids, _ = _read_rows(name)
        sets.append({by_id[c] for c in chunk_ids if c in by_id})
    return set.intersection(*sets) if sets else set()
