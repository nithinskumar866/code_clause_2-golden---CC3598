from datetime import date
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.constants import STATUS_ANALYSED
from app.core.logging import logger
from app.schemas.response import ApiResponse
from app.schemas.analysis import AnalysisHistoryItem, AnalysisHistoryDetail
from app.schemas.history import RecommendationFilter, HistorySort
from app.schemas.ranking import RankRequest, RankingResponse
from app.schemas.interview import InterviewSimulation
from app.services import analysis as history_service
from app.services import ranking as ranking_service
from app.services.ai import interview_service

router = APIRouter()

@router.post("/evaluate", response_model=ApiResponse[dict])
def evaluate_candidate(resume_id: int, jd_id: int, db: Session = Depends(get_db)):
    """Run the full LangGraph pipeline for one resume × one JD and return the report."""
    logger.info(f"LangGraph State Evaluation Pipeline triggered for Resume ID: {resume_id}, JD ID: {jd_id}")
    analysis_id, final_report = ranking_service.run_evaluation(db, resume_id, jd_id)
    return ApiResponse[dict](
        success=True,
        message="Candidate evaluation completed successfully via LangGraph workflow.",
        data={
            "analysis_id": analysis_id,
            "status": STATUS_ANALYSED,
            "report": final_report,
        },
    )


@router.post("/rank", response_model=ApiResponse[RankingResponse])
def rank_candidates(payload: RankRequest, db: Session = Depends(get_db)):
    """
    Evaluate several resumes against a single JD and return a ranked leaderboard
    (best overall score first). Reuses the same pipeline as /evaluate per resume,
    so ranks are real explainable scores. Failed resumes are still listed.
    """
    logger.info(f"Ranking {len(payload.resume_ids)} resumes against JD ID {payload.jd_id}")
    result = ranking_service.rank_candidates(db, payload.jd_id, payload.resume_ids)
    return ApiResponse[RankingResponse](
        success=True,
        message=f"Ranked {result.evaluated_count} of {result.candidate_count} candidates.",
        data=result,
    )


@router.post("/{analysis_id}/interview", response_model=ApiResponse[InterviewSimulation])
def simulate_interview(analysis_id: int):
    """
    AI Recruiter: for a completed analysis, generate — for each interview question —
    the ideal answer the candidate's resume can support, the supporting evidence,
    a confidence score, missing information, follow-ups, and a recruiter verdict.
    Uses the configured LLM (Gemini recommended) over the retrieved evidence, with
    a deterministic fallback when no LLM is configured.
    """
    logger.info(f"AI Recruiter interview simulation requested for analysis {analysis_id}.")
    result = interview_service.simulate_interview(analysis_id)
    return ApiResponse[InterviewSimulation](
        success=True,
        message=f"Interview simulated ({result.generated_by}).",
        data=result,
    )


# --- Analysis History (read/manage over already-persisted evaluations) ---
# These endpoints never run the pipeline; they only expose and manage reports
# that POST /evaluate has already persisted. /evaluate is intentionally untouched.

@router.get("/history", response_model=ApiResponse[List[AnalysisHistoryItem]])
def get_analysis_history(
    resume_filename: Optional[str] = Query(None, description="Case-insensitive substring match on resume filename"),
    jd_filename: Optional[str] = Query(None, description="Case-insensitive substring match on job description filename"),
    recommendation: Optional[RecommendationFilter] = Query(None, description="Selected / Borderline / Rejected"),
    min_score: Optional[int] = Query(None, ge=0, le=100, description="Minimum overall score"),
    max_score: Optional[int] = Query(None, ge=0, le=100, description="Maximum overall score"),
    date_from: Optional[date] = Query(None, description="Only analyses created on/after this date (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="Only analyses created on/before this date (YYYY-MM-DD)"),
    sort: HistorySort = Query(HistorySort.newest, description="Result ordering"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: Optional[int] = Query(None, ge=1, le=100, description="Items per page; omit to return all matches"),
    db: Session = Depends(get_db),
):
    """
    Return completed analyses, filtered/sorted/paginated. With no query params
    this returns all analyses newest-first (unchanged, backward-compatible).
    Pagination metadata is reported in the response envelope's `meta` field.
    """
    # Cross-field validation (single-field bounds are enforced by Query above).
    if min_score is not None and max_score is not None and min_score > max_score:
        raise HTTPException(status_code=422, detail="min_score cannot be greater than max_score.")
    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from cannot be after date_to.")

    items, meta = history_service.search_history(
        db,
        resume_filename=resume_filename,
        jd_filename=jd_filename,
        recommendation=recommendation.value if recommendation else None,
        min_score=min_score,
        max_score=max_score,
        date_from=date_from,
        date_to=date_to,
        sort=sort.value,
        page=page,
        page_size=page_size,
    )
    return ApiResponse[List[AnalysisHistoryItem]](
        success=True,
        message=f"Retrieved {len(items)} of {meta.total_count} analyses from history.",
        data=items,
        meta=meta.model_dump(),
    )


@router.get("/history/{analysis_id}", response_model=ApiResponse[AnalysisHistoryDetail])
def get_analysis_history_detail(analysis_id: int, db: Session = Depends(get_db)):
    """Return one full stored hiring report by analysis ID."""
    detail = history_service.get_history_detail(db, analysis_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found in history.")
    return ApiResponse[AnalysisHistoryDetail](
        success=True,
        message="Analysis report retrieved.",
        data=detail
    )


@router.delete("/history/{analysis_id}", response_model=ApiResponse[dict])
def delete_analysis_history_item(analysis_id: int, db: Session = Depends(get_db)):
    """Delete a single analysis from history (report file + record)."""
    deleted = history_service.delete_history_item(db, analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found in history.")
    return ApiResponse[dict](
        success=True,
        message="Analysis deleted from history.",
        data={"analysis_id": analysis_id}
    )


@router.delete("/history", response_model=ApiResponse[dict])
def clear_analysis_history(db: Session = Depends(get_db)):
    """Clear all analysis history (all report files + records)."""
    count = history_service.clear_history(db)
    return ApiResponse[dict](
        success=True,
        message=f"Cleared {count} analyses from history.",
        data={"deleted": count}
    )
