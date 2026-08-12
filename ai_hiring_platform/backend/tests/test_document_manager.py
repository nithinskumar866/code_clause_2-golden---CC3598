"""
The working set — which resumes are chunked and embedded.

At 300+ CVs, uploading and embedding stop being the same decision: the slow model
embeds at ~1.9 chunks/second, so a full pass is roughly 40 minutes. These tests pin the
behaviour that makes that cost controllable and reversible.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from app.core.config import settings
from app.services import document_manager, indexing_service
from app.services.ai import embedding_store


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """An empty shared store on disk, isolated from the operator's real one."""
    monkeypatch.setattr(embedding_store, "store_dir", lambda: str(tmp_path))
    embedding_store.invalidate()
    yield tmp_path
    embedding_store.invalidate()


def _seed_working_set(rows):
    """Write chunk/resume rows straight into the document layer."""
    conn = sqlite3.connect(embedding_store._documents_db())
    try:
        conn.executescript(embedding_store._SCHEMA)
        with conn:
            for rid, chunks in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO resumes (resume_id, filename, fingerprint, profile)"
                    " VALUES (?,?,?,?)",
                    (rid, f"{rid}.pdf", f"fp{rid}", '{"name": "Person %d"}' % rid),
                )
                for i in range(chunks):
                    conn.execute(
                        "INSERT INTO chunks (resume_id, ordinal, section, page, text)"
                        " VALUES (?,?,?,?,?)",
                        (rid, i, "Experience", 1, f"resume {rid} chunk {i}"),
                    )
    finally:
        conn.close()


# --- The working set is narrower than "uploaded" ----------------------------
def test_removal_keeps_the_file_and_is_reversible(store, monkeypatch):
    """
    "Remove from index" must not lose the resume.

    Taking an expensive model off a candidate is a cost decision; losing the candidate
    is not the same thing, which is why this is a different operation from deletion.
    """
    _seed_working_set([(1, 3), (2, 2)])
    assert len(document_manager._working_set()) == 2

    # No model holds vectors in this fixture, so no realignment work is needed.
    result = document_manager.remove_from_working_set([1], rebuild=False)

    assert result["removed"] == 1
    assert result["working_set"] == 1
    remaining = document_manager._working_set()
    assert 1 not in remaining and 2 in remaining


def test_removing_drops_that_resumes_chunks_only(store):
    _seed_working_set([(1, 3), (2, 2)])
    document_manager.remove_from_working_set([1], rebuild=False)

    conn = sqlite3.connect(embedding_store._documents_db())
    try:
        owners = [r[0] for r in conn.execute("SELECT DISTINCT resume_id FROM chunks")]
    finally:
        conn.close()
    assert owners == [2]


def test_removing_nothing_is_a_no_op(store):
    _seed_working_set([(1, 2)])
    assert document_manager.remove_from_working_set([])["removed"] == 0
    assert len(document_manager._working_set()) == 1


# --- Large uploads must not index themselves --------------------------------
def test_a_large_upload_is_not_auto_indexed(monkeypatch):
    """
    A 300-file backlog must wait for a selection.

    Auto-indexing it would spend ~40 minutes of CPU deciding, on the recruiter's behalf,
    that every CV in a client dump is worth embedding.
    """
    monkeypatch.setattr(settings, "AUTO_INDEX_ON_UPLOAD", True)
    monkeypatch.setattr(settings, "AUTO_INDEX_MAX_BATCH", 20)
    started = []
    monkeypatch.setattr(indexing_service.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: started.append(kw)})())

    assert indexing_service.schedule(list(range(1, 51)), reason="test") is False
    assert not started, "a batch over the threshold must not start an indexing run"


def test_a_small_upload_still_indexes_itself(monkeypatch):
    """The friction this removes is real — a handful of CVs should just work."""
    monkeypatch.setattr(settings, "AUTO_INDEX_ON_UPLOAD", True)
    monkeypatch.setattr(settings, "AUTO_INDEX_MAX_BATCH", 20)
    started = []
    monkeypatch.setattr(indexing_service.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: started.append(kw)})())

    assert indexing_service.schedule([1, 2, 3], reason="test") is True
    assert started


def test_startup_catch_up_never_grows_the_working_set(monkeypatch):
    """
    Startup fills models in over what is ALREADY searchable.

    If it expanded the working set instead, every uploaded resume would be chunked on
    boot — exactly the surprise the threshold exists to prevent.
    """
    monkeypatch.setattr(settings, "AUTO_INDEX_ON_STARTUP", True)
    monkeypatch.setattr(settings, "AUTO_INDEX_ON_UPLOAD", True)
    captured = {}
    monkeypatch.setattr(indexing_service.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: captured.update(kw)})())

    indexing_service.backfill_on_startup()
    # args = (resume_ids, expand)
    assert captured["args"][1] is False, "startup must not expand the working set"


# --- The API surface --------------------------------------------------------
def test_endpoints_respond_over_the_real_app(store, client, db_session):
    """
    Every management endpoint answers, and an empty selection is a no-op rather than
    an error — a recruiter pressing a button with nothing selected must not see a 500.
    """
    from app.models.database import JobDescription, Resume

    db_session.add(Resume(filename="a.pdf", status="Uploaded"))
    db_session.add(JobDescription(filename="jd.docx", status="Uploaded"))
    db_session.commit()

    listed = client.get("/api/v1/documents/resumes")
    assert listed.status_code == 200
    body = listed.json()["data"]
    assert body["total"] == 1
    # Uploaded but never chunked: the whole point of the working set being narrower.
    assert body["resumes"][0]["in_working_set"] is False
    assert body["resumes"][0]["chunks"] == 0

    jobs = client.get("/api/v1/documents/jobs")
    assert jobs.status_code == 200
    assert jobs.json()["data"]["total"] == 1

    for path in (
        "/api/v1/documents/resumes/working-set/add",
        "/api/v1/documents/resumes/working-set/remove",
        "/api/v1/documents/resumes/delete",
        "/api/v1/documents/jobs/delete",
    ):
        r = client.post(path, json={"ids": []})
        assert r.status_code == 200, f"{path} -> {r.status_code}"
        assert r.json()["success"] is True


def test_deleting_a_job_removes_the_row(store, client, db_session):
    from app.models.database import JobDescription

    job = JobDescription(filename="jd.docx", status="Uploaded")
    db_session.add(job)
    db_session.commit()
    job_id = job.id

    r = client.post("/api/v1/documents/jobs/delete", json={"ids": [job_id]})
    assert r.status_code == 200
    assert r.json()["data"]["deleted"] == 1
    assert client.get("/api/v1/documents/jobs").json()["data"]["total"] == 0


# --- Inventory --------------------------------------------------------------
def test_listing_reports_the_funnel(store):
    """
    The screen must be able to say "300 uploaded, 50 chunked, 20 through the slow model".
    A gap between those numbers is information, not an error.
    """
    _seed_working_set([(1, 3)])
    working = document_manager._working_set()
    assert working[1]["chunks"] == 3
    assert working[1]["name"] == "Person 1"


def test_models_report_per_resume_coverage(store):
    """A model that has indexed nothing reports nothing, rather than failing."""
    _seed_working_set([(1, 2)])
    by_model = document_manager._resumes_by_model()
    assert set(by_model) >= {"bge", "mxbai", "gpu"}
    assert all(ids == set() for ids in by_model.values())
