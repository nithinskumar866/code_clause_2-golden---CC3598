from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.repositories import analytics_repository
from app.services import analytics as analytics_service
from app.services.analysis import sync_history_scores
from app.schemas.dashboard import (
    DashboardOverview,
    DashboardTrends,
    TrendPoint,
    DashboardDistribution,
    ScoreRange,
)

# ---------------------------------------------------------------------------
# Dashboard service.
#
# The dashboard's calculations are delegated to the analytics service/repository
# (SQL aggregation) so there is a single source of truth and no duplicated
# calculation logic. This module only adapts analytics results into the existing
# dashboard response shapes, which remain unchanged.
# ---------------------------------------------------------------------------


def get_overview(db: Session) -> DashboardOverview:
    """Counts and average sub-scores (delegated to analytics)."""
    stats = analytics_service.get_overall_statistics(db)
    overview = DashboardOverview(
        total_analyses=stats.total_analyses,
        selected_count=stats.selected,
        borderline_count=stats.borderline,
        rejected_count=stats.rejected,
        average_overall_score=stats.average_overall_score,
        average_skill_score=stats.average_coverage_score,
        average_experience_score=stats.average_experience_score,
        average_project_score=stats.average_project_score,
        average_quality_score=stats.average_quality_score,
    )
    logger.info(f"Dashboard overview computed over {overview.total_analyses} analyses.")
    return overview


def get_distribution(db: Session) -> DashboardDistribution:
    """Score-band distribution (delegated to analytics)."""
    dist = analytics_service.get_score_distribution(db)
    return DashboardDistribution(
        total_analyses=dist.total_analyses,
        ranges=[ScoreRange(label=b.label, min=b.min, max=b.max, count=b.count) for b in dist.ranges],
    )


def get_trends(db: Session) -> DashboardTrends:
    """Per-day counts over the last N days, zero-filled (uses the analytics SQL grouping)."""
    days = settings.DASHBOARD_TRENDS_DAYS
    sync_history_scores(db)
    counts = dict(analytics_repository.daily_counts(db, days))  # {'YYYY-MM-DD': count}

    today = datetime.utcnow().date()
    start = today - timedelta(days=days - 1)
    trends = [
        TrendPoint(
            date=(start + timedelta(days=offset)).isoformat(),
            count=counts.get((start + timedelta(days=offset)).isoformat(), 0),
        )
        for offset in range(days)
    ]
    logger.info(f"Dashboard trends computed for last {days} days ({sum(counts.values())} analyses in window).")
    return DashboardTrends(period_days=days, trends=trends)
