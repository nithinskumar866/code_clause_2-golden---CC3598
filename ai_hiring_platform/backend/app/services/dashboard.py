from datetime import datetime, timedelta
from typing import List

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.constants import SCORE_DISTRIBUTION_BANDS
from app.core.logging import logger
from app.services import analysis as history_service
from app.schemas.analysis import AnalysisHistoryItem
from app.schemas.dashboard import (
    DashboardOverview,
    DashboardTrends,
    TrendPoint,
    DashboardDistribution,
    ScoreRange,
)

# ---------------------------------------------------------------------------
# Dashboard analytics service.
#
# Read-only aggregation over the existing Analysis History (completed
# evaluations only). It reuses history_service.list_history so there is a single
# source of truth and no duplicated persistence access. The evaluation pipeline
# and LangGraph workflow are not involved.
# ---------------------------------------------------------------------------


def _mean(values: List[int], total: int) -> float:
    """Rounded mean, guarding against an empty history (returns 0.0)."""
    return round(sum(values) / total, 2) if total else 0.0


def get_overview(db: Session) -> DashboardOverview:
    """Aggregate counts and average sub-scores across all completed analyses."""
    items: List[AnalysisHistoryItem] = history_service.list_history(db)
    total = len(items)

    selected = sum(1 for i in items if i.overall_score >= settings.DASHBOARD_SELECTED_MIN)
    rejected = sum(1 for i in items if i.overall_score < settings.DASHBOARD_BORDERLINE_MIN)
    borderline = total - selected - rejected

    overview = DashboardOverview(
        total_analyses=total,
        selected_count=selected,
        borderline_count=borderline,
        rejected_count=rejected,
        average_overall_score=_mean([i.overall_score for i in items], total),
        average_skill_score=_mean([i.coverage_score for i in items], total),
        average_experience_score=_mean([i.experience_score for i in items], total),
        average_project_score=_mean([i.project_score for i in items], total),
        average_quality_score=_mean([i.quality_score for i in items], total),
    )
    logger.info(f"Dashboard overview computed over {total} analyses.")
    return overview


def get_trends(db: Session) -> DashboardTrends:
    """Return per-day analysis counts over the last N days (zero-filled, oldest first)."""
    days = settings.DASHBOARD_TRENDS_DAYS
    items: List[AnalysisHistoryItem] = history_service.list_history(db)

    today = datetime.utcnow().date()
    start = today - timedelta(days=days - 1)

    counts: dict = {}
    for item in items:
        day = item.timestamp.date()
        if start <= day <= today:
            counts[day] = counts.get(day, 0) + 1

    trends = [
        TrendPoint(date=(start + timedelta(days=offset)).isoformat(),
                   count=counts.get(start + timedelta(days=offset), 0))
        for offset in range(days)
    ]
    logger.info(f"Dashboard trends computed for last {days} days ({sum(counts.values())} analyses in window).")
    return DashboardTrends(period_days=days, trends=trends)


def get_distribution(db: Session) -> DashboardDistribution:
    """Bucket completed analyses by overall_score into fixed score bands."""
    items: List[AnalysisHistoryItem] = history_service.list_history(db)

    ranges = [
        ScoreRange(
            label=label,
            min=low,
            max=high,
            count=sum(1 for i in items if low <= i.overall_score <= high),
        )
        for label, low, high in SCORE_DISTRIBUTION_BANDS
    ]
    logger.info(f"Dashboard distribution computed over {len(items)} analyses.")
    return DashboardDistribution(total_analyses=len(items), ranges=ranges)
