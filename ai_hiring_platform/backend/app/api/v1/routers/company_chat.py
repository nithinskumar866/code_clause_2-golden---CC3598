"""
Company knowledge base endpoints (`/api/v1/company-chat`).

Thin by design (Golden Rule 7): every endpoint delegates to `services/company/*` and
wraps the result in the standard `{success, message, data}` envelope. Nothing here
touches the resume pipeline — this router is reached only when the chat UI is in
company mode.
"""
import json
import queue
import threading
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.core.logging import logger
from app.schemas.company_chat import (
    CompanyChatRequest,
    CompanyChatResponse,
    CompanyStoreStatus,
)
from app.schemas.response import ApiResponse
from app.services.company import company_chat, company_retrieval, qdrant_store

router = APIRouter()


@router.post("/query", response_model=ApiResponse[CompanyChatResponse])
def company_query(payload: CompanyChatRequest):
    """Answer one question against the company knowledge base."""
    logger.info(f"Company query [session={payload.session_id}]: {payload.message[:160]!r}")
    result = company_chat.answer(
        message=payload.message,
        session_id=payload.session_id,
        limit=payload.limit,
        use_llm=payload.use_llm,
    )
    return ApiResponse[CompanyChatResponse](
        success=True,
        message="Refused." if result["refused"] else "Answer generated.",
        data=CompanyChatResponse(**result),
    )


@router.post("/stream")
def company_query_stream(payload: CompanyChatRequest):
    """
    The same answer, streaming real pipeline stages as Server-Sent Events.

    Mirrors `/chat/stream` so the UI can drive both modes with one component: the work
    runs on a worker thread and pushes a `stage` event as each step actually begins,
    then a final `result` event carrying the same payload as `/query`.
    """
    logger.info(f"Company stream [session={payload.session_id}]: {payload.message[:160]!r}")
    events: "queue.Queue[Optional[str]]" = queue.Queue()

    def on_progress(stage: str, detail: str) -> None:
        events.put("event: stage\ndata: " + json.dumps({"stage": stage, "detail": detail}) + "\n\n")

    def work() -> None:
        try:
            result = company_chat.answer(
                message=payload.message,
                session_id=payload.session_id,
                limit=payload.limit,
                use_llm=payload.use_llm,
                progress=on_progress,
            )
            body = CompanyChatResponse(**result).model_dump()
            events.put("event: result\ndata: " + json.dumps(body, default=str) + "\n\n")
        except Exception as e:
            logger.error(f"Company stream failed: {e}", exc_info=True)
            events.put("event: error\ndata: " + json.dumps({"message": str(e)}) + "\n\n")
        finally:
            events.put(None)

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
def company_reset(session_id: str = Query("default")):
    """Forget a company conversation's follow-up memory."""
    company_chat.reset_session(session_id)
    return ApiResponse[dict](
        success=True, message="Conversation reset.", data={"session_id": session_id}
    )


@router.get("/status", response_model=ApiResponse[CompanyStoreStatus])
def company_status():
    """
    Whether the company database is configured, reachable and populated.

    The UI reads this before offering the company toggle, so a recruiter is told the
    store is empty rather than being handed an assistant that answers nothing.
    """
    return ApiResponse[CompanyStoreStatus](
        success=True,
        message="Company store status retrieved.",
        data=CompanyStoreStatus(**company_chat.store_status()),
    )


@router.get("/companies", response_model=ApiResponse[list])
def company_list():
    """Every company in the store, for a picker or a sanity check after loading."""
    if not qdrant_store.configured():
        return ApiResponse[list](success=True, message="Not configured.", data=[])
    return ApiResponse[list](
        success=True,
        message="Companies retrieved.",
        data=qdrant_store.list_companies(),
    )


@router.post("/refresh", response_model=ApiResponse[dict])
def company_refresh():
    """
    Re-read the name index after a load, without restarting the API.

    The loader runs as a separate script, so a running server would otherwise keep a
    stale list of company and people names for up to the cache TTL.
    """
    company_retrieval.refresh_index()
    return ApiResponse[dict](
        success=True, message="Company index refreshed.", data=qdrant_store.status()
    )
