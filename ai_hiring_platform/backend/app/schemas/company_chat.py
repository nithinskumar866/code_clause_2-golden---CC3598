"""
API contract for the company knowledge base (`/api/v1/company-chat`).

Backend owns this contract; `frontend/src/types/index.ts` mirrors it. Kept separate
from `schemas/chat.py` on purpose: a company and a candidate share no fields, and one
union type covering both would force every UI branch to check which half it holds.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CompanyFieldMatch(BaseModel):
    """One retrieved field of one company — the audit trail behind a claim."""

    field: str
    label: str
    text: str
    # Cosine similarity against the question, 0-1. Named `score` to match what the
    # store returns, so the value crosses every layer under one name.
    score: float


class CompanyResult(BaseModel):
    """One company, reassembled from the field-level hits that matched."""

    company_id: str
    company_name: str
    relevance: float
    best_field: str = ""
    matches: List[CompanyFieldMatch] = []
    # The whole stored row, so the UI can show any field without another request.
    fields: Dict[str, str] = {}
    industries: List[str] = []
    people: List[Dict[str, Any]] = []


class CompanyChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    limit: int = 5
    use_llm: bool = True


class CompanyChatResponse(BaseModel):
    answer: str
    companies: List[CompanyResult] = []
    # 'company' when the question named one, 'open' for a pool-wide search, 'none' when
    # the question was refused before retrieval ran.
    mode: str = "none"
    # 'llm' when a model phrased the answer, 'deterministic' otherwise. The facts are
    # identical either way — only the prose differs.
    engine: str = "deterministic"
    refused: bool = False
    refusal_category: Optional[str] = None
    is_followup: bool = False
    stats: Dict[str, Any] = {}
    elapsed_ms: int = 0


class CompanyStoreStatus(BaseModel):
    configured: bool
    reachable: bool
    collection: str
    points: int
    companies: int
    vector_size: int
    detail: str = ""
