"""
The working set — deciding WHICH resumes are chunked and embedded.

WHY THIS IS A SEPARATE DECISION
-------------------------------
Uploading a resume and paying to index it are two different acts, and a pool of 300+
CVs makes the difference expensive rather than theoretical: mxbai embeds at roughly 1.9
chunks/second on CPU, so a full pass over 300 resumes is around 40 minutes. Indexing
everything the moment it arrives is the wrong default at that scale — a recruiter
loading a client's backlog wants to choose the 40 people worth searching today.

So the platform has a **working set**: the resumes currently parsed into the shared
document layer (`documents.db`). It is the answer to "what is searchable", and it is
deliberately smaller than "what has been uploaded".

    uploaded          every file in the resumes table + on disk
      └── working set  parsed + chunked into documents.db      <- YOU CHOOSE THIS
            └── per-model vectors  vectors-<model>.faiss       <- and which models run

Adding to the working set costs parsing (fast, model-independent). Embedding costs one
pass per model over the working set's chunks. Removing from the working set drops the
chunks and the vectors, but never the file — so it is fully reversible, and taking an
expensive model off a candidate does not lose the candidate.

WHY REMOVAL DOES NOT RE-EMBED ANYTHING
--------------------------------------
`index_model` rebuilds a model's index from whatever chunks currently exist, reusing
every vector it already holds. So dropping resumes from the working set and re-running
it costs a file write and no embedding at all — the removed chunks simply stop being
included. That is why "remove from index" is instant while "add to index" is not.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from sqlalchemy.orm import Session

from app.core.constants import JOB_UPLOAD_DIR, RESUME_UPLOAD_DIR
from app.core.logging import logger
from app.models.database import JobDescription, Resume
from app.services.ai import embedding_engines, embedding_store


# --- Inventory --------------------------------------------------------------
def _working_set() -> Dict[int, Dict[str, Any]]:
    """{resume_id -> {chunks, filename}} for everything currently in the document layer."""
    conn = sqlite3.connect(embedding_store._documents_db())
    try:
        conn.executescript(embedding_store._SCHEMA)
        counts = dict(
            conn.execute("SELECT resume_id, COUNT(*) FROM chunks GROUP BY resume_id").fetchall()
        )
        rows = conn.execute("SELECT resume_id, filename, profile FROM resumes").fetchall()
    finally:
        conn.close()

    out: Dict[int, Dict[str, Any]] = {}
    for rid, filename, profile_json in rows:
        try:
            profile = json.loads(profile_json)
        except (TypeError, ValueError):
            profile = {}
        out[int(rid)] = {
            "chunks": int(counts.get(rid, 0)),
            "filename": filename,
            "name": profile.get("name"),
            "title": profile.get("title"),
            "total_years": profile.get("total_years"),
            "location": profile.get("location"),
        }
    return out


def _resumes_by_model() -> Dict[str, Set[int]]:
    """Which resumes each model actually holds vectors for."""
    chunk_owner = {c["chunk_id"]: c["resume_id"] for c in embedding_store.read_chunks()}
    out: Dict[str, Set[int]] = {}
    for name in embedding_engines.ENGINE_NAMES:
        chunk_ids, _meta = embedding_store._read_rows(name)
        out[name] = {chunk_owner[c] for c in chunk_ids if c in chunk_owner}
    return out


def list_resumes(db: Session) -> Dict[str, Any]:
    """
    Every uploaded resume with its indexing state — the data the selection screen shows.

    Reports per resume whether it is in the working set, how many chunks it produced,
    and which models hold vectors for it. A recruiter can then see at a glance that 300
    are uploaded, 50 are searchable, and only 20 of those have been through the slow
    model — which is the whole point of making the working set visible.
    """
    working = _working_set()
    by_model = _resumes_by_model()

    items: List[Dict[str, Any]] = []
    for resume in db.query(Resume).order_by(Resume.upload_time.desc()).all():
        entry = working.get(resume.id)
        path = os.path.join(RESUME_UPLOAD_DIR, f"{resume.id}_{resume.filename}")
        items.append({
            "id": resume.id,
            "filename": resume.filename,
            "upload_time": resume.upload_time,
            "status": resume.status,
            "file_present": os.path.exists(path),
            "in_working_set": entry is not None,
            "chunks": entry["chunks"] if entry else 0,
            "name": entry.get("name") if entry else None,
            "title": entry.get("title") if entry else None,
            "total_years": entry.get("total_years") if entry else None,
            "location": entry.get("location") if entry else None,
            "models": {name: resume.id in ids for name, ids in by_model.items()},
        })

    return {
        "resumes": items,
        "total": len(items),
        "in_working_set": sum(1 for i in items if i["in_working_set"]),
        "models": {name: len(ids) for name, ids in by_model.items()},
    }


def list_jobs(db: Session) -> Dict[str, Any]:
    """
    Every uploaded job description.

    JDs carry no indexing state because they are never chunked or embedded: a JD is
    parsed for its requirements at analysis time and discarded. There is therefore no
    working-set decision to make here — only listing and deletion.
    """
    items = []
    for job in db.query(JobDescription).order_by(JobDescription.upload_time.desc()).all():
        path = os.path.join(JOB_UPLOAD_DIR, f"{job.id}_{job.filename}")
        items.append({
            "id": job.id,
            "filename": job.filename,
            "upload_time": job.upload_time,
            "status": job.status,
            "file_present": os.path.exists(path),
            "analyses": len(job.analyses),
        })
    return {"jobs": items, "total": len(items)}


def read_resume(resume_id: int) -> Dict[str, Any]:
    """
    One resume's parsed content, section by section — what the "view resume" panel shows.

    Served from the document layer rather than by re-parsing the file: this is exactly
    the text the platform indexed, so what a recruiter reads is what the search actually
    matched on. Re-parsing could show something subtly different from what was scored,
    which would make the evidence trail a lie.
    """
    conn = sqlite3.connect(embedding_store._documents_db())
    try:
        conn.executescript(embedding_store._SCHEMA)
        row = conn.execute(
            "SELECT filename, profile FROM resumes WHERE resume_id = ?", (resume_id,)
        ).fetchone()
        chunks = conn.execute(
            "SELECT ordinal, section, page, text FROM chunks WHERE resume_id = ? ORDER BY ordinal",
            (resume_id,),
        ).fetchall()
    finally:
        conn.close()

    if not row:
        return {"resume_id": resume_id, "in_working_set": False, "sections": [], "profile": {}}

    try:
        profile = json.loads(row[1])
    except (TypeError, ValueError):
        profile = {}

    # Group consecutive chunks by section so the panel reads like a document rather
    # than like a list of retrieval fragments.
    sections: List[Dict[str, Any]] = []
    for ordinal, section, page, text in chunks:
        if sections and sections[-1]["section"] == section:
            sections[-1]["text"] += "\n" + text
        else:
            sections.append({"section": section, "page": page, "text": text})

    return {
        "resume_id": resume_id,
        "filename": row[0],
        "in_working_set": True,
        "profile": profile,
        "sections": sections,
        "chunks": len(chunks),
    }


def resume_file_path(db: Session, resume_id: int) -> Optional[str]:
    """Disk path of the ORIGINAL upload, for download. None when the file is gone."""
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if resume is None:
        return None
    path = os.path.join(RESUME_UPLOAD_DIR, f"{resume.id}_{resume.filename}")
    return path if os.path.exists(path) else None


# --- Changing the working set ----------------------------------------------
def add_to_working_set(db: Session, resume_ids: Sequence[int]) -> Dict[str, Any]:
    """
    Parse and chunk the selected resumes into the shared document layer.

    Model-independent and comparatively cheap, so it is safe to run synchronously for a
    reasonable selection. Embedding is a separate, explicit step: adding 200 resumes
    here costs seconds, embedding them costs minutes per model, and the recruiter should
    be able to make those two decisions separately.
    """
    ids = [int(r) for r in resume_ids]
    if not ids:
        return {"added": 0, "chunks": 0, "working_set": len(_working_set())}

    result = embedding_store.sync_documents(db, resume_ids=ids)
    embedding_store.invalidate()
    working = _working_set()
    logger.info(f"Working set: added {len(ids)} resume(s); now {len(working)} total.")
    return {
        "added": result.added + result.updated,
        "unchanged": result.unchanged,
        "failed": result.failed,
        "chunks": sum(w["chunks"] for w in working.values()),
        "working_set": len(working),
    }


def remove_from_working_set(resume_ids: Sequence[int], rebuild: bool = True) -> Dict[str, Any]:
    """
    Drop the selected resumes' chunks — and therefore their vectors — keeping the files.

    Reversible by design: the upload and its database row are untouched, so re-adding
    costs one parse. Every model is realigned afterwards, which is a file write rather
    than an embedding run because `index_model` reuses the vectors it already holds.
    """
    ids = [int(r) for r in resume_ids]
    if not ids:
        return {"removed": 0, "working_set": len(_working_set())}

    with embedding_store._LOCK:
        conn = sqlite3.connect(embedding_store._documents_db())
        try:
            conn.executescript(embedding_store._SCHEMA)
            placeholders = ",".join("?" * len(ids))
            with conn:
                conn.execute(f"DELETE FROM chunks WHERE resume_id IN ({placeholders})", ids)
                conn.execute(f"DELETE FROM resumes WHERE resume_id IN ({placeholders})", ids)
        finally:
            conn.close()
    embedding_store.invalidate()

    realigned: List[str] = []
    if rebuild:
        # Only models that actually hold something need realigning, and none of them
        # re-embeds: the removed chunks simply stop being included.
        for name, held in _resumes_by_model().items():
            if held:
                try:
                    embedding_store.index_model(name)
                    realigned.append(name)
                except Exception as e:
                    logger.error(f"Realigning '{name}' after removal failed: {e}", exc_info=True)

    working = _working_set()
    logger.info(f"Working set: removed {len(ids)} resume(s); now {len(working)} total.")
    return {"removed": len(ids), "working_set": len(working), "realigned": realigned}


# --- Permanent deletion -----------------------------------------------------
def delete_resumes(db: Session, resume_ids: Sequence[int]) -> Dict[str, Any]:
    """
    Remove resumes completely: vectors, chunks, database row and the file on disk.

    Destructive and not reversible, which is why it is a different call from
    `remove_from_working_set` rather than a flag on it. Analyses referencing the resume
    are removed by the model's cascade.
    """
    ids = [int(r) for r in resume_ids]
    if not ids:
        return {"deleted": 0, "files_removed": 0}

    # Chunks and vectors first: a row deleted from the database while its chunks remain
    # would leave the document layer describing a resume that no longer exists.
    remove_from_working_set(ids, rebuild=True)

    files_removed = 0
    deleted = 0
    for resume in db.query(Resume).filter(Resume.id.in_(ids)).all():
        path = os.path.join(RESUME_UPLOAD_DIR, f"{resume.id}_{resume.filename}")
        if os.path.exists(path):
            try:
                os.remove(path)
                files_removed += 1
            except OSError as e:
                logger.error(f"Could not delete file for resume {resume.id}: {e}")
        db.delete(resume)
        deleted += 1
    db.commit()

    logger.info(f"Deleted {deleted} resume(s) permanently ({files_removed} file(s) removed).")
    return {"deleted": deleted, "files_removed": files_removed}


def delete_jobs(db: Session, job_ids: Sequence[int]) -> Dict[str, Any]:
    """Remove job descriptions completely. JDs hold no vectors, so this is row + file."""
    ids = [int(j) for j in job_ids]
    if not ids:
        return {"deleted": 0, "files_removed": 0}

    files_removed = 0
    deleted = 0
    for job in db.query(JobDescription).filter(JobDescription.id.in_(ids)).all():
        path = os.path.join(JOB_UPLOAD_DIR, f"{job.id}_{job.filename}")
        if os.path.exists(path):
            try:
                os.remove(path)
                files_removed += 1
            except OSError as e:
                logger.error(f"Could not delete file for job {job.id}: {e}")
        db.delete(job)
        deleted += 1
    db.commit()

    logger.info(f"Deleted {deleted} job description(s) ({files_removed} file(s) removed).")
    return {"deleted": deleted, "files_removed": files_removed}
