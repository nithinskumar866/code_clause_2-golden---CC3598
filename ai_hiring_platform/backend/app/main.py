import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger
from app.core.database import init_db
from app.core.exceptions import register_exception_handlers
from app.api.v1.routers import health, resume, job, analysis, dashboard, notes, workflow, analytics, export, chat, embeddings, documents, prompt_lab, company_chat

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the database
    logger.info("Starting up FastAPI application...")
    init_db()
    # Bring the shared embedding store level with the database, in the background.
    # Incremental and non-blocking: startup is not delayed, and a deployment that
    # predates a model fills that model in without anyone pressing a button.
    from app.services import indexing_service

    indexing_service.backfill_on_startup()
    yield
    # Shutdown logic if any
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

# Register CORS Middleware
# Allow local dev frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exception handlers
register_exception_handlers(app)

# Include Routers
app.include_router(health.router, prefix=f"{settings.API_V1_STR}/health", tags=["Health"])
app.include_router(resume.router, prefix=f"{settings.API_V1_STR}/resume", tags=["Resumes"])
app.include_router(job.router, prefix=f"{settings.API_V1_STR}/job", tags=["Job Descriptions"])
app.include_router(analysis.router, prefix=f"{settings.API_V1_STR}/analysis", tags=["Analyses"])
app.include_router(dashboard.router, prefix=f"{settings.API_V1_STR}/dashboard", tags=["Dashboard"])
app.include_router(analytics.router, prefix=f"{settings.API_V1_STR}/analytics", tags=["Analytics"])
# Recruiter chatbot — pool-wide candidate Q&A over the unified corpus index.
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["Recruiter Chat"])
app.include_router(embeddings.router, prefix=f"{settings.API_V1_STR}/embeddings", tags=["Embedding Store"])
app.include_router(documents.router, prefix=f"{settings.API_V1_STR}/documents", tags=["Document Management"])
# Prompt Lab — measure a system prompt against a fixed case set. A test harness, not a
# runtime agent: it never joins the hiring workflow.
app.include_router(prompt_lab.router, prefix=f"{settings.API_V1_STR}/prompt-lab", tags=["Prompt Lab"])
# Company knowledge base (Qdrant). Opt-in module: nothing else in the app calls it.
app.include_router(company_chat.router, prefix=f"{settings.API_V1_STR}/company-chat", tags=["Company Intelligence"])
# Notes routes span /analysis/{id}/notes and /notes/{id}, so mount at the API root.
app.include_router(notes.router, prefix=settings.API_V1_STR, tags=["Notes"])
# Workflow status routes live under /analysis/{id}/status, so mount at the API root.
app.include_router(workflow.router, prefix=settings.API_V1_STR, tags=["Workflow Status"])
# Export routes live under /analysis/{id}/export/*, so mount at the API root.
app.include_router(export.router, prefix=settings.API_V1_STR, tags=["Report Export"])

# Serve the built frontend (single-container deploy) when a dist directory is present.
# Mounted LAST so all /api/v1/* routes take precedence; unknown paths fall back to the
# SPA. Absent in local/dev/test (no dist) — then the JSON welcome route is used instead.
_FRONTEND_DIST = os.environ.get("FRONTEND_DIST", "")
if _FRONTEND_DIST and os.path.isdir(_FRONTEND_DIST):
    logger.info(f"Serving frontend static files from {_FRONTEND_DIST}")
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {
            "message": f"Welcome to the {settings.PROJECT_NAME} API. Access /docs for API documentation."
        }
