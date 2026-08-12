"""
Automatic indexing — a resume becomes searchable by every model, on upload.

WHY THIS EXISTS
---------------
Indexing used to be a button. That was a deliberate choice, and it was defensible: an
implicit step had once succeeded for the local model, failed for the remote one while
its endpoint was down, and said nothing — so 300 resumes existed for one model and not
the other, and no screen admitted it.

But the cure created a worse disease in daily use. A recruiter uploads resumes on the
Resume page, then finds the Recruiter Assistant asking them to "index new resumes", and
Model Lab reporting a different count again. Nothing is actually shared until somebody
remembers to press a button on a third screen, and it *feels* like the same CVs are
being embedded over and over.

So indexing is automatic again — but with the visibility the first version lacked:

  * **Fan out over every AVAILABLE model.** A resume that arrives while the GPU
    endpoint is down is still embedded by the local models, and the gap for the remote
    one stays visible in `/embeddings/coverage` until it is filled.
  * **Never silent.** Every run records progress and failures against the same tracker
    the explicit path uses, so the coverage screen tells the same story either way.
  * **Never blocking.** Upload returns as soon as the file is saved; embedding happens
    on a worker thread. A 100-file bulk upload must not hold a request open for minutes.
  * **Never duplicated.** It writes to the one shared store, and the document layer is
    content-fingerprinted, so re-running costs nothing for resumes that have not changed.

WHAT "EMBEDDED ONCE" MEANS
--------------------------
Parsing and chunking happen ONCE per resume, into `documents.db`. Each model then
embeds those same chunks ONCE, into `vectors-<model>.faiss`. Nothing re-embeds on a
later visit to any screen: the Recruiter Assistant, AI Analysis, Ranking and Model Lab
all *read* those vectors. Re-embedding only ever happens when the file content changes
(a new fingerprint) or a model is deliberately rebuilt.
"""
from __future__ import annotations

import threading
from typing import List, Optional, Sequence

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logging import logger
from app.services.ai import embedding_engines, embedding_store

# One indexing run at a time, process-wide. The models are CPU-bound and a 0.64 GB
# ONNX model running beside another is how the allocator runs out of memory — the same
# reason the explicit path runs them in sequence rather than together.
_RUN_LOCK = threading.Lock()
_pending = threading.Event()


def available_model_names() -> List[str]:
    """
    Models that can actually embed right now.

    A remote model whose endpoint is unreachable is skipped rather than failed: its
    resumes are picked up by the next run, and `coverage()` shows it behind in the
    meantime. Indexing it under a fallback engine would put BGE vectors in the remote
    model's index and quietly corrupt every later comparison.
    """
    names: List[str] = []
    for engine in embedding_engines.all_engines():
        try:
            if engine.available():
                names.append(engine.name)
            else:
                logger.info(f"Auto-index: '{engine.name}' is not reachable; skipping this pass.")
        except Exception as e:
            logger.warning(f"Auto-index: availability check failed for '{engine.name}': {e}")
    return names


def _run(resume_ids: Optional[Sequence[int]], expand: bool) -> None:
    """
    Top up every available model, optionally growing the working set first.

    `expand` is the difference between "make these new resumes searchable" and "catch
    the models up on what is already searchable". Startup must never expand: adding
    every uploaded resume to the working set is precisely the 40-minute surprise the
    threshold exists to prevent.
    """
    db = SessionLocal()
    try:
        if expand:
            # The document layer is model-independent and must be current BEFORE any
            # model embeds, so every model is shown an identical chunk set.
            result = embedding_store.sync_documents(
                db, resume_ids=list(resume_ids) if resume_ids else None
            )
            if result.failed:
                logger.warning(
                    f"Auto-index: {len(result.failed)} file(s) could not be parsed: {result.failed}"
                )

        for name in available_model_names():
            try:
                embedding_store.index_model(name)
            except Exception as e:
                # One model failing must not stop the others — a partially covered pool
                # is strictly better than an unindexed one, and the gap stays visible.
                logger.error(f"Auto-index: '{name}' failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Auto-index run failed: {e}", exc_info=True)
    finally:
        db.close()


def _worker(resume_ids: Optional[Sequence[int]], expand: bool) -> None:
    with _RUN_LOCK:
        _pending.clear()
        _run(resume_ids, expand)


def schedule(
    resume_ids: Optional[Sequence[int]] = None,
    reason: str = "upload",
    expand: bool = True,
) -> bool:
    """
    Queue an indexing pass on a worker thread. Returns False when nothing was scheduled.

    `resume_ids` narrows the PARSE step to the resumes that just arrived — fingerprinting
    600 files to discover that 599 are unchanged is the slowest part of a small upload.
    The embed step is always incremental across the whole store anyway, so a narrowed
    parse costs nothing in coverage.

    A LARGE upload is deliberately NOT indexed. Above `AUTO_INDEX_MAX_BATCH` the files
    are stored and left out of the working set, because deciding which of 300 CVs is
    worth ~40 minutes of embedding is the recruiter's call, not a default. They pick
    them on the Documents screen instead.

    If a run is already in flight the request is coalesced rather than queued: the run
    in progress re-reads the store when it reaches each model, so a second identical
    pass would only repeat work.
    """
    if not settings.AUTO_INDEX_ON_UPLOAD:
        logger.info("Auto-indexing is disabled (AUTO_INDEX_ON_UPLOAD=false); skipping.")
        return False

    if resume_ids is not None and len(resume_ids) > settings.AUTO_INDEX_MAX_BATCH:
        logger.info(
            f"Auto-index skipped for {len(resume_ids)} resumes ({reason}): batches over "
            f"{settings.AUTO_INDEX_MAX_BATCH} are chosen on the Documents screen so the "
            f"embedding cost is deliberate."
        )
        return False

    if _RUN_LOCK.locked() and _pending.is_set():
        logger.info(f"Auto-index already queued behind a running pass ({reason}); coalescing.")
        return True

    _pending.set()
    threading.Thread(
        target=_worker, args=(resume_ids, expand), daemon=True, name="auto-indexer"
    ).start()
    logger.info(f"Auto-index scheduled ({reason}) for {resume_ids or 'all new resumes'}.")
    return True


def backfill_on_startup() -> None:
    """
    Catch every model up on the EXISTING working set. Never grows it.

    On a store that is already current this costs nothing. On one that predates a model
    — or where a model's endpoint was down when its resumes were selected — it fills
    that model in without anyone pressing a button. What is searchable does not change;
    only which models can answer about it.
    """
    if not settings.AUTO_INDEX_ON_STARTUP:
        return
    schedule(None, reason="startup catch-up", expand=False)
