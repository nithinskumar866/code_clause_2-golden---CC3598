from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.core.logging import logger
from app.schemas.response import ApiResponse
from app.schemas.analytics import (
    OverallStatistics,
    ScoreDistribution,
    RecommendationDistribution,
    TrendData,
    TopItem,
    RecentAnalysisItem,
    SkillFrequency,
)
from app.services import analytics as analytics_service

router = APIRouter()


@router.get("/overview", response_model=ApiResponse[OverallStatistics])
def get_overview(db: Session = Depends(get_db)):
    """Overall statistics: totals, recommendation counts, average sub-scores."""
    logger.info("Analytics overview endpoint accessed.")
    data = analytics_service.get_overall_statistics(db)
    return ApiResponse[OverallStatistics](success=True, message="Overall statistics retrieved.", data=data)


@router.get("/score-distribution", response_model=ApiResponse[ScoreDistribution])
def get_score_distribution(db: Session = Depends(get_db)):
    """Distribution of overall scores across fixed bands."""
    logger.info("Analytics score-distribution endpoint accessed.")
    data = analytics_service.get_score_distribution(db)
    return ApiResponse[ScoreDistribution](success=True, message="Score distribution retrieved.", data=data)


@router.get("/recommendation-distribution", response_model=ApiResponse[RecommendationDistribution])
def get_recommendation_distribution(db: Session = Depends(get_db)):
    """Selected/Borderline/Rejected counts and percentages."""
    logger.info("Analytics recommendation-distribution endpoint accessed.")
    data = analytics_service.get_recommendation_distribution(db)
    return ApiResponse[RecommendationDistribution](
        success=True, message="Recommendation distribution retrieved.", data=data
    )


@router.get("/trends", response_model=ApiResponse[TrendData])
def get_trends(db: Session = Depends(get_db)):
    """Daily, weekly and monthly analysis counts."""
    logger.info("Analytics trends endpoint accessed.")
    data = analytics_service.get_trends(db)
    return ApiResponse[TrendData](success=True, message="Trend data retrieved.", data=data)


@router.get("/top-resumes", response_model=ApiResponse[List[TopItem]])
def get_top_resumes(
    limit: int = Query(settings.ANALYTICS_TOP_LIMIT, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Most-analysed resume filenames."""
    logger.info("Analytics top-resumes endpoint accessed.")
    data = analytics_service.get_top_resumes(db, limit)
    return ApiResponse[List[TopItem]](success=True, message="Top resumes retrieved.", data=data)


@router.get("/top-jobs", response_model=ApiResponse[List[TopItem]])
def get_top_jobs(
    limit: int = Query(settings.ANALYTICS_TOP_LIMIT, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Most-analysed job description filenames."""
    logger.info("Analytics top-jobs endpoint accessed.")
    data = analytics_service.get_top_job_descriptions(db, limit)
    return ApiResponse[List[TopItem]](success=True, message="Top job descriptions retrieved.", data=data)


@router.get("/skill-frequency", response_model=ApiResponse[SkillFrequency])
def get_skill_frequency(
    limit: int = Query(settings.ANALYTICS_TOP_LIMIT, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Most frequently matched requirements and most common missing skills."""
    logger.info("Analytics skill-frequency endpoint accessed.")
    data = analytics_service.get_skill_frequency(db, limit)
    return ApiResponse[SkillFrequency](success=True, message="Skill frequency retrieved.", data=data)


@router.get("/recent", response_model=ApiResponse[List[RecentAnalysisItem]])
def get_recent_activity(
    limit: int = Query(settings.ANALYTICS_RECENT_LIMIT, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Most recent completed analyses."""
    logger.info("Analytics recent endpoint accessed.")
    data = analytics_service.get_recent_activity(db, limit)
    return ApiResponse[List[RecentAnalysisItem]](success=True, message="Recent activity retrieved.", data=data)
