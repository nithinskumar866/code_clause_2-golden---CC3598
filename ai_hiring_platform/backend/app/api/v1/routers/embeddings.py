"""
Embedding store management and the model comparison.

Thin by design (Golden Rule 7): every endpoint delegates to a service and wraps the
result in the standard `{success, message, data}` envelope.

Indexing is EXPLICIT here rather than automatic on upload. An implicit step is exactly
what let two models drift apart: it succeeded for the local model, failed for the
remote one when its endpoint was down, and reported nothing — so 300 resumes existed
for one model and not the other, and no screen said so.
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.embeddings import (
    ComparisonRequest,
    ComparisonResponse,
    IndexAccepted,
    IndexRequest,
    StoreCoverage,
)
from app.schemas.response import ApiResponse
from app.services.ai import embedding_engines, embedding_store, model_comparison_service

router = APIRouter()


@router.get("/coverage", response_model=ApiResponse[StoreCoverage])
def store_coverage(db: Session = Depends(get_db)):
    """
    What each model holds, against the one shared document set.

    The screen that makes a silent divergence impossible: it reports per-model resume
    and chunk counts, and the intersection every model has actually seen.
    """
    data = embedding_store.coverage(db)
    data["engines"] = embedding_engines.engine_status()["engines"]
    return ApiResponse[StoreCoverage](
        success=True, message="Embedding coverage retrieved.", data=StoreCoverage(**data)
    )


@router.post("/index", response_model=ApiResponse[IndexAccepted])
def index_models(payload: IndexRequest, db: Session = Depends(get_db)):
    """
    Index the shared chunk set with the selected models.

    Returns immediately and works on a background thread: a full pass over hundreds of
    resumes takes minutes for the larger models, far longer than a request should be
    held open. Poll `/embeddings/progress` for live state.

    Models run one after another rather than together — they compete for the same CPU,
    and a 0.64 GB model running beside another is how the ONNX allocator runs out of
    memory.
    """
    logger.info(f"Indexing requested for models: {payload.models}")
    data = embedding_store.index_models_async(
        payload.models, db, sync_first=payload.sync_documents
    )
    return ApiResponse[IndexAccepted](
        success=True,
        message=f"Indexing started for {', '.join(data['queued'])}.",
        data=IndexAccepted(**data),
    )


@router.get("/progress", response_model=ApiResponse[dict])
def index_progress(model: Optional[str] = Query(None, description="Limit to one model.")):
    """Live indexing state. Cheap enough to poll every second."""
    return ApiResponse[dict](
        success=True, message="Indexing progress retrieved.",
        data=embedding_store.progress(model),
    )


@router.post("/documents/sync", response_model=ApiResponse[dict])
def sync_documents(
    force: bool = Query(False, description="Re-parse every resume, not only changed ones."),
    db: Session = Depends(get_db),
):
    """
    Parse and chunk new resumes without embedding anything.

    Model-independent, and the reason a three-way comparison is meaningful: every model
    indexes the SAME chunk set, so a difference in results is a difference in the model
    rather than in what it was shown.
    """
    return ApiResponse[dict](
        success=True, message="Document layer synchronised.",
        data=embedding_store.sync_documents(db, force=force).as_dict(),
    )


@router.post("/compare", response_model=ApiResponse[ComparisonResponse])
def compare_models(payload: ComparisonRequest, db: Session = Depends(get_db)):
    """
    Ask several models the same question and report the answers side by side.

    Deterministic retrieval by default — that is what differs between embedding models.
    `use_llm` additionally generates each model's written answer, which is useful to
    read but adds wording variance between runs, so it is opt-in rather than the
    measurement itself.
    """
    logger.info(f"Model comparison [{payload.models}]: {payload.message[:120]!r}")
    data = model_comparison_service.compare(
        db=db,
        message=payload.message,
        models=payload.models,
        limit=payload.limit,
        use_llm=payload.use_llm,
        fair_mode=payload.fair_mode,
    )
    return ApiResponse[ComparisonResponse](
        success=True, message="Comparison complete.", data=ComparisonResponse(**data)
    )
