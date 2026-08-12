"""
API contract for the shared embedding store and the model comparison.

Backend owns this contract; `frontend/src/types/index.ts` mirrors it.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EngineInfo(BaseModel):
    """One embedding model the platform can index and search with."""

    name: str
    model: Optional[str] = None
    dimension: Optional[int] = None
    location: Optional[str] = None
    # False when a remote endpoint is unreachable. A model that cannot serve cannot be
    # indexed with, and the operator must see that rather than discover it as a gap.
    available: bool = False
    configured: bool = True
    # Each model carries its OWN cosine floor, calibrated to equal selectivity on this
    # project's resumes. A shared threshold would reward whichever model scores higher
    # in absolute terms rather than the one that ranks better.
    min_similarity: Optional[float] = None


class IndexProgressInfo(BaseModel):
    model: str
    state: str = "idle"          # idle | queued | running | done | error
    done: int = 0
    total: int = 0
    embedded: int = 0
    reused: int = 0
    percent: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


class ModelCoverage(BaseModel):
    """How much of the shared chunk set one model actually holds."""

    model: str
    indexed_resumes: int = 0
    indexed_chunks: int = 0
    dimension: Optional[int] = None
    in_sync: bool = False
    progress: IndexProgressInfo


class StoreCoverage(BaseModel):
    """
    The whole picture: one document set, several models indexing it.

    `comparable_resumes` is the intersection across models — the only population on
    which a multi-model comparison is honest, because it is the only one every model
    has actually seen. `models_aligned` is false the moment they diverge, which is the
    condition that used to go unnoticed.
    """

    documents: Dict[str, int]
    database_resumes: Optional[int] = None
    documents_in_sync: bool = True
    models: List[ModelCoverage] = []
    comparable_resumes: int = 0
    models_aligned: bool = False
    engines: List[EngineInfo] = []


class IndexRequest(BaseModel):
    models: List[str] = Field(
        ..., description="Which models to index, e.g. ['bge','mxbai','gpu']."
    )
    sync_documents: bool = Field(
        True, description="Parse and chunk new resumes first. Model-independent."
    )


class IndexAccepted(BaseModel):
    queued: List[str]
    documents: Dict[str, int]


# --- Model comparison -------------------------------------------------------
class ComparisonCandidate(BaseModel):
    """A single result inside one model's column."""

    resume_id: int
    name: Optional[str] = None
    title: Optional[str] = None
    match_percentage: int = 0
    total_years: Optional[float] = None
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    top_evidence: Optional[str] = None
    section: Optional[str] = None
    similarity: Optional[float] = None


class ModelAnswer(BaseModel):
    """What one model made of the question."""

    model: str
    dimension: Optional[int] = None
    available: bool = True
    answer: str = ""
    # Present only when the LLM toggle is on; the deterministic answer is always there.
    llm_answer: Optional[str] = None
    answer_type: str = "candidates"
    candidates: List[ComparisonCandidate] = []
    elapsed_ms: int = 0
    indexed_resumes: int = 0
    error: Optional[str] = None


class ComparisonAgreement(BaseModel):
    """
    How much the models agree — the number that makes this a comparison.

    Eyeballing three lists tells you little; knowing that two models return the same
    top candidate and share 4 of 5 results tells you whether the upgrade is buying
    anything.
    """

    top1_unanimous: bool = False
    top1_by_model: Dict[str, Optional[str]] = {}
    # Jaccard overlap of the returned resume sets, per model pair.
    overlap: Dict[str, float] = {}
    # Resume ids every model returned.
    common_resume_ids: List[int] = []


class ComparisonRequest(BaseModel):
    message: str = Field(..., description="The question every model answers.")
    models: List[str] = Field(default_factory=lambda: ["bge", "mxbai", "gpu"])
    limit: int = Field(5, ge=1, le=20)
    use_llm: bool = Field(
        False,
        description="Also generate each model's natural-language answer. Off by default: "
                    "the LLM's wording varies run to run and would blur the retrieval "
                    "differences this page exists to show.",
    )
    fair_mode: bool = Field(
        True,
        description="Restrict every model to the resumes ALL selected models have "
                    "indexed. With it off, a model that has seen more resumes can win "
                    "simply by having more to choose from.",
    )


class ComparisonResponse(BaseModel):
    question: str
    models: List[ModelAnswer] = []
    agreement: ComparisonAgreement
    # How many resumes the comparison actually ran over, and whether that was forced.
    compared_over_resumes: int = 0
    fair_mode: bool = True
    warnings: List[str] = []
