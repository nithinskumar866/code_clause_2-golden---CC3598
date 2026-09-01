"""
Application entry point.

Startup does three things and none of them are expensive: create the collections if
missing, warn if the live vector size disagrees with the configured model, and start
the refresh scheduler. The embedding model is loaded lazily on first use rather than
here, so a container starts fast and a service that is only serving `/health` never
pays for weights it will not use.

    uvicorn app.main:app --reload --port 8080
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logging import logger
from app.ingest import refresh
from app.store import qdrant

_scheduler = None


def _start_scheduler() -> None:
    """
    Hourly refresh of whatever is due.

    Hourly is the *check* interval, not the crawl interval — a page's own cadence
    decides when it actually comes due, so most of these runs find nothing and cost one
    filtered scroll.
    """
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        _scheduler = BackgroundScheduler(daemon=True)
        _scheduler.add_job(
            refresh.run,
            "interval",
            hours=1,
            id="refresh",
            max_instances=1,          # never let a slow run overlap the next tick
            coalesce=True,            # after downtime, catch up once, not once per missed hour
        )
        _scheduler.start()
        logger.info("Refresh scheduler started (hourly).")
    except Exception as e:
        logger.warning(f"Scheduler not started: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if qdrant.configured():
        try:
            qdrant.ensure_collections()
            mismatch = qdrant.verify_content_dimension()
            if mismatch:
                # Loud, but not fatal: /health should still answer so an operator can
                # see what is wrong instead of reading a crash loop.
                logger.error(f"VECTOR DIMENSION MISMATCH — {mismatch}")
            _start_scheduler()
        except Exception as e:
            logger.error(f"Startup could not reach Qdrant: {e}")
    else:
        logger.warning("QDRANT_URL is not set — copy .env.example to .env before use.")

    yield

    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
    from app.crawl import fetcher

    fetcher.close_client()


app = FastAPI(
    title="company_intel",
    description=(
        "Company knowledge from their own websites: crawled at ingestion time, stored "
        "in Qdrant, answered from the index. The chat path never crawls."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "service": "company_intel",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
