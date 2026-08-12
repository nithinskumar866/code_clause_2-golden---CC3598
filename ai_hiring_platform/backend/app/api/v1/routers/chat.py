"""
Recruiter chatbot endpoints.

Thin by design (Golden Rule 7): every endpoint delegates to a service and wraps the
result in the standard `{success, message, data}` envelope.
"""
import json
import queue
import threading
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CorpusStatus,
    CorpusSyncResult,
)
from app.schemas.response import ApiResponse
from app.services.ai import chat_service, embedding_engines, llm_service

router = APIRouter()


@router.post("/query", response_model=ApiResponse[ChatResponse])
def chat_query(payload: ChatRequest, db: Session = Depends(get_db)):
    """Answer one recruiter question against the whole indexed resume pool."""
    logger.info(f"Chat query [session={payload.session_id}]: {payload.message[:160]!r}")
    result = chat_service.answer(
        db=db,
        message=payload.message,
        session_id=payload.session_id,
        limit=payload.limit,
        use_llm=payload.use_llm,
        embedding_engine=payload.embedding_engine,
    )
    return ApiResponse[ChatResponse](
        success=True,
        message="Refused by guardrails." if result["refused"] else "Answer generated.",
        data=ChatResponse(**result),
    )


@router.post("/stream")
def chat_query_stream(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    Answer one question, streaming REAL pipeline progress as Server-Sent Events.

    The work runs on a worker thread and pushes a `stage` event as each actual step
    begins — guardrails, understanding, retrieval, scoring, reasoning — then a final
    `result` event carrying the same payload as `/query`. The UI therefore reports what
    the backend is genuinely doing instead of animating a timer, which matters when one
    question takes 30 ms and the next takes 3 s.
    """
    logger.info(f"Chat stream [session={payload.session_id}]: {payload.message[:160]!r}")
    events: "queue.Queue[Optional[str]]" = queue.Queue()

    def on_progress(stage: str, detail: str) -> None:
        events.put("event: stage\ndata: " + json.dumps({"stage": stage, "detail": detail}) + "\n\n")

    def work() -> None:
        try:
            result = chat_service.answer(
                db=db, message=payload.message, session_id=payload.session_id,
                limit=payload.limit, use_llm=payload.use_llm,
                embedding_engine=payload.embedding_engine, progress=on_progress,
            )
            body = ChatResponse(**result).model_dump()
            events.put("event: result\ndata: " + json.dumps(body, default=str) + "\n\n")
        except Exception as e:  # surface the failure to the client rather than hanging
            logger.error(f"Chat stream failed: {e}", exc_info=True)
            events.put("event: error\ndata: " + json.dumps({"message": str(e)}) + "\n\n")
        finally:
            events.put(None)  # sentinel: closes the stream

    threading.Thread(target=work, daemon=True).start()

    def generate():
        while True:
            item = events.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/reset", response_model=ApiResponse[dict])
def chat_reset(session_id: str = Query("default")):
    """Forget a conversation's follow-up memory."""
    chat_service.reset_session(session_id)
    return ApiResponse[dict](success=True, message="Conversation reset.", data={"session_id": session_id})


@router.get("/corpus/status", response_model=ApiResponse[CorpusStatus])
def corpus_status(
    engine: Optional[str] = Query(None, description="'bge', 'mxbai' or 'gpu'. Defaults to EMBEDDING_ENGINE."),
    db: Session = Depends(get_db),
):
    """
    What THIS model can currently see, read from the shared store. Never rebuilds.

    The chatbot no longer keeps an index of its own. It reads the same
    `documents.db` + `vectors-<model>.faiss` that upload, AI Analysis, Ranking and
    Model Lab use, so "indexed" means the same thing on every screen and a resume is
    never embedded twice for the same model.
    """
    return ApiResponse[CorpusStatus](
        success=True,
        message="Corpus status retrieved.",
        data=CorpusStatus(**chat_service.pool_status(db, engine_name=engine)),
    )


@router.post("/corpus/sync", response_model=ApiResponse[CorpusSyncResult])
def corpus_sync(
    force: bool = Query(False, description="Re-embed every chunk instead of only new/changed ones."),
    engine: Optional[str] = Query(None, description="Which model to top up: 'bge', 'mxbai' or 'gpu'."),
    db: Session = Depends(get_db),
):
    """
    Top up the shared store for one model — a manual catch-up, not the normal path.

    Uploads index themselves in the background (see `services/indexing_service.py`), so
    this exists for the cases automation cannot cover: a model whose endpoint was down
    when its resumes arrived, or a deliberate rebuild after a model change. It writes to
    the SAME store every other screen reads, so nothing is duplicated.
    """
    return ApiResponse[CorpusSyncResult](
        success=True,
        message="Store synchronised.",
        data=CorpusSyncResult(**chat_service.pool_sync(db, force=force, engine_name=engine)),
    )


@router.get("/engines", response_model=ApiResponse[dict])
def embedding_engines_status():
    """
    Which embedding engines exist, and whether each can serve right now.

    Both run on the operator's own infrastructure — the local model inside this
    process, the GPU model on the endpoint configured in EMBEDDING_GPU_URL. No
    third-party vector service is involved either way.
    """
    return ApiResponse[dict](
        success=True,
        message="Embedding engines retrieved.",
        data=embedding_engines.engine_status(),
    )


@router.get("/health", response_model=ApiResponse[dict])
def chat_health():
    """Report which reasoning engine the chatbot will use, and whether it responds."""
    llm = llm_service.get_llm()
    if llm is None:
        return ApiResponse[dict](
            success=True,
            message="Chat health retrieved.",
            data={"engine": "deterministic", "llm_configured": False, "llm_reachable": False},
        )
    reachable = llm.health() if hasattr(llm, "health") else True
    return ApiResponse[dict](
        success=True,
        message="Chat health retrieved.",
        data={
            "engine": "llm" if reachable else "deterministic",
            "llm_configured": True,
            "llm_reachable": reachable,
            "model": getattr(llm, "model", None),
        },
    )
