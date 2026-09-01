"""
HTTP endpoints. Thin by construction: every route validates, delegates and wraps.

No crawling logic, no retrieval logic and no scoring lives here. When a route grows
past a handful of lines it is doing something that belongs in a service — that rule is
what keeps the pipeline testable without an HTTP client.
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ApiResponse,
    ChatRequest,
    ChatResponse,
    Company,
    CompanyCreate,
    CompanyStatus,
    CrawlRequest,
    Health,
    RefreshRequest,
)
from app.chat import answer as answer_service
from app.core.config import settings
from app.embed import engine
from app.core.logging import logger
from app.ingest import pipeline, refresh
from app.sources import registry, state
from app.store import qdrant

router = APIRouter()


def require_store() -> None:
    """
    Refuse work that needs Qdrant when Qdrant is not configured.

    A 503 with a sentence an operator can act on, rather than the RuntimeError from the
    client factory surfacing as an opaque 500. `/health` deliberately does not depend on
    this — its whole job is to report the outage.
    """
    if not qdrant.configured():
        raise HTTPException(
            status_code=503,
            detail="Qdrant is not configured. Set QDRANT_URL and QDRANT_API_KEY in .env.",
        )

# Crawls started with `background=true`, keyed by company. A crawl is long and a second
# one for the same company would fetch the same pages concurrently and race on the same
# points, so one at a time per company is enforced here.
_running: Dict[str, threading.Thread] = {}
_running_lock = threading.Lock()


# --- health -----------------------------------------------------------------
@router.get("/health", response_model=ApiResponse[Health])
def health() -> ApiResponse[Health]:
    store = qdrant.health()
    companies = 0
    pages_due = 0
    if store.get("reachable"):
        try:
            companies = len(registry.list_all())
            pages_due = refresh.pending_count()
        except Exception as e:
            logger.warning(f"Health counts unavailable: {e}")

    # Only probed for a remote provider: the local model is a file on disk, and loading
    # it just to answer a health check would make /health the slowest route in the app.
    if settings.embed_provider == "ollama":
        embed_ok, embed_detail = engine.reachable()
    else:
        embed_ok, embed_detail = True, ""

    healthy = bool(store.get("reachable")) and embed_ok
    return ApiResponse[Health](
        success=healthy,
        message=(
            "Ready."
            if healthy
            else "Qdrant is not reachable." if not store.get("reachable")
            else "The embedding endpoint is not reachable."
        ),
        data=Health(
            status="ok" if healthy else "degraded",
            qdrant=store,
            embedding_model=engine.model_name(),
            embedding_dim=engine.dimension(),
            embedding_provider=settings.embed_provider,
            embedding_reachable=embed_ok,
            embedding_detail=embed_detail,
            llm_provider=(settings.ANSWER_PROVIDER or "none").strip().lower(),
            llm_configured=settings.llm_configured,
            companies=companies,
            pages_due=pages_due,
        ),
    )


# --- companies --------------------------------------------------------------
@router.post("/companies", response_model=ApiResponse[Company], dependencies=[Depends(require_store)])
def create_company(payload: CompanyCreate) -> ApiResponse[Company]:
    try:
        company = registry.register(**payload.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ApiResponse[Company](message="Company registered.", data=Company(**company))


@router.get("/companies", response_model=ApiResponse[List[Company]], dependencies=[Depends(require_store)])
def list_companies(enabled_only: bool = Query(False)) -> ApiResponse[List[Company]]:
    companies = [Company(**c) for c in registry.list_all(enabled_only=enabled_only)]
    return ApiResponse[List[Company]](message=f"{len(companies)} company(ies).", data=companies)


@router.get("/companies/{company_id}", response_model=ApiResponse[CompanyStatus], dependencies=[Depends(require_store)])
def company_status(company_id: str) -> ApiResponse[CompanyStatus]:
    company = registry.get(company_id)
    if not company:
        raise HTTPException(status_code=404, detail=f"No company with id {company_id!r}.")
    summary = state.summarize(company_id)
    return ApiResponse[CompanyStatus](
        message="Status.",
        data=CompanyStatus(company=Company(**company), **summary),
    )


@router.delete("/companies/{company_id}", response_model=ApiResponse[Dict[str, int]], dependencies=[Depends(require_store)])
def delete_company(company_id: str) -> ApiResponse[Dict[str, int]]:
    if not registry.get(company_id):
        raise HTTPException(status_code=404, detail=f"No company with id {company_id!r}.")
    removed = registry.delete(company_id)
    return ApiResponse[Dict[str, int]](message="Company and all its data removed.", data=removed)


# --- ingestion --------------------------------------------------------------
@router.post("/companies/{company_id}/crawl", response_model=ApiResponse[Dict[str, Any]], dependencies=[Depends(require_store)])
def crawl(company_id: str, payload: CrawlRequest) -> ApiResponse[Dict[str, Any]]:
    if not registry.get(company_id):
        raise HTTPException(status_code=404, detail=f"No company with id {company_id!r}.")

    with _running_lock:
        active = _running.get(company_id)
        if active and active.is_alive():
            raise HTTPException(status_code=409, detail="A crawl is already running for this company.")

    def work() -> None:
        try:
            pipeline.crawl_company(
                company_id,
                max_pages=payload.max_pages,
                max_depth=payload.max_depth,
                force=payload.force,
            )
        except Exception as e:
            logger.error(f"Crawl failed for {company_id}: {e}")

    if payload.background:
        thread = threading.Thread(target=work, name=f"crawl:{company_id}", daemon=True)
        with _running_lock:
            _running[company_id] = thread
        thread.start()
        return ApiResponse[Dict[str, Any]](
            message="Crawl started. Poll GET /companies/{id} for progress.",
            data={"company_id": company_id, "started": True},
        )

    report = pipeline.crawl_company(
        company_id, max_pages=payload.max_pages, max_depth=payload.max_depth, force=payload.force
    )
    return ApiResponse[Dict[str, Any]](message="Crawl complete.", data=report.as_dict())


@router.post("/refresh/run", response_model=ApiResponse[Dict[str, Any]], dependencies=[Depends(require_store)])
def run_refresh(payload: RefreshRequest) -> ApiResponse[Dict[str, Any]]:
    report = refresh.run(limit=payload.limit, company_id=payload.company_id)
    return ApiResponse[Dict[str, Any]](message="Refresh complete.", data=report.as_dict())


# --- chat -------------------------------------------------------------------
@router.post("/chat/stream", dependencies=[Depends(require_store)])
def chat_stream(payload: ChatRequest):
    """
    The same answer, streaming real pipeline stages as Server-Sent Events.

    The stages are emitted as each step actually begins, not on a timer — a progress
    indicator that invents its own steps is worse than none, because it tells the user
    something false about where the time is going.

    The work runs on a worker thread and pushes frames into a queue, because the answer
    path is synchronous (the embedding model and the Qdrant client both block) and
    running it inline would hold the event loop for the whole request.
    """
    events: "queue.Queue[Optional[str]]" = queue.Queue()

    def _frame(event: str, payload: Dict[str, Any]) -> str:
        """One SSE frame: an event name, a JSON data line, and a blank line to close it."""
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    def on_progress(stage: str, detail: str) -> None:
        events.put(_frame("stage", {"stage": stage, "detail": detail}))

    def work() -> None:
        try:
            result = answer_service.ask(
                payload.message,
                company_id=payload.company_id,
                limit=payload.limit,
                use_llm=payload.use_llm,
                on_progress=on_progress,
            )
            events.put(_frame("result", result))
        except Exception as e:
            logger.error(f"Streamed answer failed: {e}")
            events.put(_frame("error", {"message": str(e)}))
        finally:
            events.put(None)  # sentinel: the generator below stops on it

    threading.Thread(target=work, name="chat-stream", daemon=True).start()

    def frames():
        while True:
            frame = events.get()
            if frame is None:
                return
            yield frame

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/query", response_model=ApiResponse[ChatResponse], dependencies=[Depends(require_store)])
def chat(payload: ChatRequest) -> ApiResponse[ChatResponse]:
    result = answer_service.ask(
        payload.message,
        company_id=payload.company_id,
        limit=payload.limit,
        use_llm=payload.use_llm,
    )
    return ApiResponse[ChatResponse](
        success=not result["refused"],
        message="Refused." if result["refused"] else "Answer generated.",
        data=ChatResponse(**result),
    )
