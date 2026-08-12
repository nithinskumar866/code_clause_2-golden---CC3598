"""
Dual embedding engine tests.

The dangerous failure here is silent: searching an index built by one model with a
query embedded by another returns confident nonsense. These tests pin the guarantees
that make that impossible — engine identity is part of the index path, an unreachable
GPU endpoint degrades to the local model rather than erroring or mixing, and the
caller is always told which engine actually served.
"""
import numpy as np
import pytest

from app.core.config import settings
from app.services.ai import corpus_index_service, embedding_engines


@pytest.fixture(autouse=True)
def reset_gpu_engine():
    embedding_engines._gpu_singleton = None
    yield
    embedding_engines._gpu_singleton = None


def test_local_engine_is_the_default_and_always_available():
    engine = embedding_engines.resolve_engine(None)
    assert engine.name == "bge"
    assert engine.dimension == 384
    assert engine.available()


@pytest.mark.parametrize("alias", ["bge", "local", "cpu", "BGE", " bge "])
def test_local_aliases_resolve(alias):
    assert embedding_engines.resolve_engine(alias).name == "bge"


def test_unknown_engine_falls_back_rather_than_raising(monkeypatch):
    """A config typo must not take retrieval down."""
    assert embedding_engines.resolve_engine("gpt-embeddings-9000").name == "bge"


def test_gpu_requested_but_unconfigured_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_GPU_URL", "")
    monkeypatch.setattr(settings, "EMBEDDING_GPU_MODEL", "")
    assert embedding_engines.resolve_engine("gpu").name == "bge"


def test_gpu_requested_but_unreachable_falls_back(monkeypatch):
    """A RunPod outage must degrade to the local index, not error out."""
    monkeypatch.setattr(settings, "EMBEDDING_GPU_URL", "http://127.0.0.1:9")  # closed port
    monkeypatch.setattr(settings, "EMBEDDING_GPU_MODEL", "nomic-embed-text")
    monkeypatch.setattr(settings, "EMBEDDING_GPU_TIMEOUT_SECONDS", 1.0)
    assert embedding_engines.resolve_engine("gpu").name == "bge"


def test_each_engine_gets_its_own_index_directory():
    """Vectors from different models must never share an index."""
    assert corpus_index_service.corpus_dir("bge") != corpus_index_service.corpus_dir("gpu")
    assert corpus_index_service.corpus_dir("bge").endswith("corpus-bge")


def test_status_reports_fallback_explicitly(monkeypatch):
    """A silent substitution would mean searching the wrong index unknowingly."""
    monkeypatch.setattr(settings, "EMBEDDING_GPU_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(settings, "EMBEDDING_GPU_MODEL", "nomic-embed-text")
    monkeypatch.setattr(settings, "EMBEDDING_GPU_TIMEOUT_SECONDS", 1.0)
    status = corpus_index_service.corpus_status(db=None, engine_name="gpu")
    assert status["engine"] == "bge"
    assert status["engine_requested"] == "gpu"
    assert status["engine_fell_back"] is True


def test_no_fallback_flag_when_local_was_asked_for():
    status = corpus_index_service.corpus_status(db=None, engine_name="bge")
    assert status["engine_fell_back"] is False


def test_embeddings_are_unit_normalised():
    """Inner-product search only equals cosine similarity on normalised vectors."""
    engine = embedding_engines.resolve_engine("bge")
    vectors = engine.embed_documents(["java microservices", "python data pipelines"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    assert vectors.shape == (2, engine.dimension)


def test_query_and_document_embeddings_share_a_dimension():
    engine = embedding_engines.resolve_engine("bge")
    assert engine.embed_query("java").shape[1] == engine.embed_documents(["java"]).shape[1]


def test_empty_input_returns_an_empty_matrix_of_the_right_width():
    engine = embedding_engines.resolve_engine("bge")
    out = engine.embed_documents([])
    assert out.shape == (0, engine.dimension)


def test_engine_status_lists_every_engine():
    status = embedding_engines.engine_status()
    names = {e["name"] for e in status["engines"]}
    assert names == {"bge", "mxbai", "gpu"}
    local = next(e for e in status["engines"] if e["name"] == "bge")
    assert local["available"] is True


def test_local_engines_are_always_available():
    """
    Both local models must be indexable with the network down. That is what lets a
    three-way comparison stay honest during a GPU outage: the two local models can be
    brought level on the same resumes, and the remote one is visibly behind rather
    than silently absent.
    """
    for name in ("bge", "mxbai"):
        assert embedding_engines.resolve_engine(name).name == name
        assert embedding_engines.resolve_engine(name).available() is True


def test_each_model_carries_its_own_similarity_floor():
    """
    Cosine values are not comparable across models, so a shared threshold would reward
    whichever model scores higher in absolute terms rather than the one that ranks
    better. Each floor is calibrated to the same SELECTIVITY on this project's resumes.
    """
    floors = {e["name"]: e["min_similarity"] for e in embedding_engines.engine_status()["engines"]}
    assert floors["bge"] != floors["mxbai"]
    assert 0.0 < floors["mxbai"] < 1.0


def test_mxbai_batches_small_enough_to_run():
    """
    FastEmbed defaults to 256; a 335M-parameter model at 512 tokens then asks ONNX
    Runtime for a 1.15 GB activation buffer and dies. The batch size is a correctness
    setting, not a tuning knob.
    """
    assert embedding_engines.resolve_engine("mxbai").batch_size <= 16
