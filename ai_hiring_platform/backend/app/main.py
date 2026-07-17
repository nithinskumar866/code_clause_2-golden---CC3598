from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.core.database import init_db
from app.core.exceptions import register_exception_handlers
from app.api.v1.routers import health, resume, job, analysis, dashboard, notes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the database
    logger.info("Starting up FastAPI application...")
    init_db()
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
# Notes routes span /analysis/{id}/notes and /notes/{id}, so mount at the API root.
app.include_router(notes.router, prefix=settings.API_V1_STR, tags=["Notes"])

@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {settings.PROJECT_NAME} API. Access /docs for API documentation."
    }
