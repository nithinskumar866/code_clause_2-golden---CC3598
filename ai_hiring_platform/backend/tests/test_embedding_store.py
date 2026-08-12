"""
The shared embedding store: one document set, several models indexing it.

Every test here guards a failure that actually happened. Three hundred resumes were
uploaded while the remote model's endpoint was down; they entered the local model's
index and nothing else, and no screen reported the gap — so two models were quietly
answering questions about different talent pools. Separately, the evaluation pipeline
kept its own per-resume indexes hardcoded to 384 dimensions, which meant every resume
was embedded twice and choosing a model did nothing outside the chatbot.

The store's job is to make both structurally impossible: parsing and chunking happen
ONCE, every model indexes that same chunk set, and coverage is reportable per model.
"""
from __future__ import annotations

import os
import zlib
from typing import Any, Dict, List

import numpy as np
import pytest

from app.services.ai import embedding_store


class _StubEngine:
    """
    A deterministic stand-in for a real model.

    CRC32 rather than `hash()`: Python salts string hashing per process, so the same
    resume would embed differently on every run and a coverage assertion could pass or
    fail by luck.
    """

    def __init__(self, name: str, dimension: int) -> None:
        self.name = name
        self.dimension = dimension
        self.min_similarity = 0.3
        self.batch_size = 4
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype="float32")
        for tok in (text or "").lower().split():
            v[zlib.crc32(tok.encode()) % self.dimension] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        self.calls += len(texts)
        return np.asarray([self._vec(t) for t in texts], dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text).reshape(1, self.dimension)

    def available(self) -> bool:
        return True


class _Resume:
    def __init__(self, rid: int, filename: str) -> None:
        self.id, self.filename = rid, filename


class _Query:
    def __init__(self, rows): self.rows = rows
    def all(self): return self.rows
    def count(self): return len(self.rows)
    def filter(self, *_a, **_k): return self


class _DB:
    def __init__(self, rows): self.rows = rows
    def query(self, *_a): return _Query(self.rows)


RESUMES = [
    (1, "alice.docx", "Alice Kumar\nBengaluru 560001\nSKILLS\nJava Spring Boot\n"
                      "EXPERIENCE\nBuilt Java microservices and Spring Boot APIs in production."),
    (2, "bob.docx", "Bob Menon\nChennai 600017\nSKILLS\nPython SQL\n"
                    "EXPERIENCE\nPython data pipelines and SQL warehousing for analytics."),
    (3, "cara.docx", "Cara Silva\nPune 411001\nSKILLS\nAzure Docker\n"
                     "EXPERIENCE\nRan Azure DevOps release pipelines and Docker builds."),
]


def _write_docx(path, text: str) -> None:
    """
    The document loader accepts PDF and DOCX only, so fixtures must be real documents.

    Writing plain text here would exercise the loader's rejection path instead of the
    store, and every assertion below would pass vacuously against an empty corpus.
    """
    from docx import Document

    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(str(path))


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A private store on disk, with two stub models and no network."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    vectors = tmp_path / "vectors"
    vectors.mkdir()

    monkeypatch.setattr(embedding_store, "RESUME_UPLOAD_DIR", str(uploads))
    monkeypatch.setattr(embedding_store, "VECTOR_STORE_DIR", str(vectors))
    embedding_store.invalidate()

    for rid, filename, text in RESUMES:
        _write_docx(uploads / f"{rid}_{filename}", text)

    engines = {"small": _StubEngine("small", 32), "large": _StubEngine("large", 64)}
    monkeypatch.setattr(
        embedding_store.embedding_engines, "resolve_engine",
        lambda name=None: engines.get(name or "small", engines["small"]),
    )
    monkeypatch.setattr(embedding_store.embedding_engines, "ENGINE_NAMES", ("small", "large"))

    db = _DB([_Resume(rid, fn) for rid, fn, _ in RESUMES])
    yield {"db": db, "engines": engines, "uploads": uploads}
    embedding_store.invalidate()


# --- The document layer is shared, and parsed exactly once ------------------
def test_documents_are_parsed_once_for_every_model(store):
    """
    Chunking is model-independent, so it must not happen per model. This is what makes
    a multi-model comparison meaningful at all: every model indexes the SAME chunks.
    """
    result = embedding_store.sync_documents(store["db"])
    assert result.added == 3
    assert result.total_chunks > 0
    assert result.failed == []

    again = embedding_store.sync_documents(store["db"])
    assert again.added == 0 and again.updated == 0
    assert again.unchanged == 3, "an unchanged file must not be re-parsed"

    chunks = embedding_store.read_chunks()
    for model in ("small", "large"):
        embedding_store.index_model(model)
        assert len(embedding_store.load_index(model).chunks) == len(chunks)


def test_every_model_indexes_the_identical_chunk_set(store):
    embedding_store.sync_documents(store["db"])
    embedding_store.index_model("small")
    embedding_store.index_model("large")

    small = {c["chunk_id"] for c in embedding_store.load_index("small").chunks}
    large = {c["chunk_id"] for c in embedding_store.load_index("large").chunks}
    assert small == large, "a comparison is only fair over an identical population"


# --- Coverage makes divergence visible -------------------------------------
def test_a_model_left_behind_is_reported_not_hidden(store):
    """
    THE REGRESSION: 300 resumes were indexed by one model only, and nothing said so.
    Coverage must show the gap, and `models_aligned` must go false.
    """
    embedding_store.sync_documents(store["db"])
    embedding_store.index_model("small")

    report = embedding_store.coverage(store["db"])
    by_model = {m["model"]: m for m in report["models"]}
    assert by_model["small"]["indexed_resumes"] == 3
    assert by_model["large"]["indexed_resumes"] == 0
    assert by_model["small"]["in_sync"] is True
    assert by_model["large"]["in_sync"] is False

    embedding_store.index_model("large")
    aligned = embedding_store.coverage(store["db"])
    assert aligned["models_aligned"] is True
    assert aligned["comparable_resumes"] == 3


def test_comparable_ids_are_the_intersection(store):
    embedding_store.sync_documents(store["db"])
    embedding_store.index_model("small")
    assert embedding_store.comparable_resume_ids(["small", "large"]) == set()

    embedding_store.index_model("large")
    assert embedding_store.comparable_resume_ids(["small", "large"]) == {1, 2, 3}


# --- Incremental: adding a resume costs one resume -------------------------
def test_indexing_reuses_existing_vectors(store):
    embedding_store.sync_documents(store["db"])
    embedding_store.index_model("small")
    first = store["engines"]["small"].calls
    assert first > 0

    store["engines"]["small"].calls = 0
    out = embedding_store.index_model("small")
    assert store["engines"]["small"].calls == 0, "nothing changed, so nothing should re-embed"
    assert out["reused"] == first and out["embedded"] == 0


def test_a_new_resume_embeds_only_itself(store):
    embedding_store.sync_documents(store["db"])
    embedding_store.index_model("small")
    baseline = embedding_store.load_index("small").index.ntotal

    _write_docx(
        store["uploads"] / "4_dan.docx",
        "Dan Roy\nKochi 682001\nSKILLS\nReact\nEXPERIENCE\nReact interfaces.",
    )
    store["db"].rows.append(_Resume(4, "dan.docx"))

    store["engines"]["small"].calls = 0
    embedding_store.sync_documents(store["db"])
    out = embedding_store.index_model("small")

    assert out["embedded"] < baseline, "adding one resume must not re-embed the pool"
    assert out["reused"] == baseline
    assert embedding_store.load_index("small").indexed_resume_ids == {1, 2, 3, 4}


def test_a_changed_file_is_reindexed(store):
    embedding_store.sync_documents(store["db"])
    embedding_store.index_model("small")

    _write_docx(
        store["uploads"] / "1_alice.docx",
        "Alice Kumar\nBengaluru 560001\nSKILLS\nKubernetes Terraform\n"
        "EXPERIENCE\nRan Kubernetes clusters and Terraform infrastructure.",
    )
    result = embedding_store.sync_documents(store["db"])
    assert result.updated == 1

    embedding_store.index_model("small")
    texts = " ".join(
        c["text"] for c in embedding_store.load_index("small").chunks if c["resume_id"] == 1
    )
    assert "Kubernetes" in texts and "Spring Boot" not in texts


def test_a_deleted_resume_leaves_the_store(store):
    embedding_store.sync_documents(store["db"])
    embedding_store.index_model("small")

    store["db"].rows = [r for r in store["db"].rows if r.id != 2]
    embedding_store.sync_documents(store["db"])
    embedding_store.index_model("small")

    assert 2 not in embedding_store.load_index("small").indexed_resume_ids


# --- Search: whole pool, or exactly one candidate --------------------------
def test_pool_search_and_single_resume_search(store):
    embedding_store.sync_documents(store["db"])
    embedding_store.index_model("small")
    index = embedding_store.load_index("small")

    pool = index.search("Java Spring Boot microservices", top_k=3)
    assert pool, "a pool search must return something"
    assert index.chunks[pool[0][0]]["resume_id"] == 1

    # Filtered search scores the candidate's OWN chunks exhaustively, so it can never
    # come back empty just because stronger chunks exist elsewhere in the pool — which
    # is how a per-candidate evaluation used to lose its evidence.
    only_bob = index.search("Java Spring Boot microservices", top_k=3, resume_ids={2})
    assert only_bob, "a single-candidate search must return that candidate's own chunks"
    assert all(index.chunks[row]["resume_id"] == 2 for row, _ in only_bob)


def test_models_keep_their_own_dimensions(store):
    embedding_store.sync_documents(store["db"])
    embedding_store.index_model("small")
    embedding_store.index_model("large")
    assert embedding_store.load_index("small").index.d == 32
    assert embedding_store.load_index("large").index.d == 64


# --- Indexing an unavailable model must fail loudly ------------------------
def test_indexing_a_model_that_resolved_elsewhere_is_refused(store, monkeypatch):
    """
    `resolve_engine` falls back to the local model when a remote endpoint is down.
    Indexing under that fallback would fill the LOCAL index while reporting success for
    the remote one — the exact silent divergence this store exists to prevent.
    """
    embedding_store.sync_documents(store["db"])
    monkeypatch.setattr(
        embedding_store.embedding_engines, "resolve_engine",
        lambda name=None: store["engines"]["small"],
    )
    with pytest.raises(RuntimeError, match="not available"):
        embedding_store.index_model("large")
