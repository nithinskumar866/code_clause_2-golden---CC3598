"""
The API contract.

Request and response shapes live here and nowhere else, so a client can be written
against this file alone. Every endpoint returns the same `{success, message, data}`
envelope — a caller checks one field to know whether to look at `data`, rather than
learning a different shape per route.
"""
from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = ""
    data: Optional[T] = None


# --- companies --------------------------------------------------------------
class CompanyCreate(BaseModel):
    name: str = Field("", description="Display name. Defaults to the domain.")
    domain: str = Field(..., description="Company domain, e.g. acme.com")
    seed_urls: List[str] = Field(
        default_factory=list, description="Where the crawl starts. Defaults to the domain root."
    )
    linkedin_urls: List[str] = Field(
        default_factory=list,
        description=(
            "LinkedIn profiles of officials. STORED AND SURFACED AS LINKS ONLY — "
            "never fetched. See docs/linkedin.md."
        ),
    )
    allow_patterns: List[str] = Field(
        default_factory=list, description="If set, only URLs containing one of these are crawled."
    )
    deny_patterns: List[str] = Field(
        default_factory=list, description="URLs containing any of these are never crawled."
    )
    enabled: bool = True


class Company(BaseModel):
    company_id: str
    name: str
    domain: str
    seed_urls: List[str] = []
    linkedin_urls: List[str] = []
    allow_patterns: List[str] = []
    deny_patterns: List[str] = []
    enabled: bool = True
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class CompanyStatus(BaseModel):
    company: Company
    pages_known: int = 0
    pages_by_status: Dict[str, int] = {}
    pages_by_type: Dict[str, int] = {}
    chunks: int = 0
    last_crawled_at: Optional[float] = None
    next_due_at: Optional[float] = None
    failures: List[Dict[str, Any]] = []


# --- ingestion --------------------------------------------------------------
class CrawlRequest(BaseModel):
    max_pages: Optional[int] = Field(None, description="Overrides CRAWL_MAX_PAGES for this run.")
    max_depth: Optional[int] = Field(None, description="Overrides CRAWL_MAX_DEPTH for this run.")
    force: bool = Field(
        False,
        description=(
            "Re-embed even when ETag and content hash say nothing changed. For a chunker "
            "or model change, not for routine crawling."
        ),
    )
    background: bool = Field(True, description="Return immediately and crawl on a worker thread.")


class RefreshRequest(BaseModel):
    limit: Optional[int] = None
    company_id: Optional[str] = None


# --- chat -------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    company_id: Optional[str] = Field(
        None, description="Narrow to one company. Otherwise inferred from the question."
    )
    limit: int = 8
    use_llm: bool = True


class Citation(BaseModel):
    n: int
    company_id: str
    company_name: str
    page_url: str
    page_title: str = ""
    page_type: str = ""
    section: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    refused: bool = False
    # 'company' when the question was narrowed to one, 'pool' for a search across all,
    # 'none' when the question was refused before retrieval ran.
    scope: str = "none"
    company: Optional[Dict[str, Any]] = None
    companies: List[Dict[str, Any]] = []
    citations: List[Citation] = []
    # How many passages the answer was WRITTEN from. `companies` and `citations`
    # describe exactly this set — never a wider one.
    evidence_count: int = 0
    # How many cleared retrieval before the prompt budget trimmed them. Reported for
    # diagnostics only; it is not what the answer stands on.
    retrieved_count: int = 0
    intent: str = ""
    llm_used: bool = False
    duration_seconds: float = 0.0


# --- health -----------------------------------------------------------------
class Health(BaseModel):
    status: str
    qdrant: Dict[str, Any]
    embedding_model: str
    embedding_dim: int
    # 'fastembed' (local) or 'ollama' (remote GPU). Reported because a remote provider
    # can be asleep, and "the GPU endpoint is down" is a different problem from "Qdrant
    # is down" even though both stop the same features.
    embedding_provider: str = "fastembed"
    embedding_reachable: bool = True
    embedding_detail: str = ""
    llm_provider: str = "none"
    llm_configured: bool = False
    companies: int = 0
    pages_due: int = 0
