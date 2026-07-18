import os
import json
import glob
import time
from collections import Counter
from typing import Callable

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.constants import SCORE_DISTRIBUTION_BANDS, REPORT_DIR
from app.core.logging import logger
from app.repositories import analytics_repository as repo
from app.services.analysis import sync_history_scores
from app.schemas.analytics import (
    OverallStatistics,
    ScoreBucket,
    ScoreDistribution,
    RecommendationBucket,
    RecommendationDistribution,
    TrendPoint,
    TrendData,
    TopItem,
    RecentAnalysisItem,
    SkillCount,
    SkillFrequency,
)

# ---------------------------------------------------------------------------
# Analytics service.
#
# Composes SQL aggregates from analytics_repository into API schemas. Score
# columns are backfilled once via sync_history_scores so aggregation stays
# SQL-native. An optional process-level TTL cache (disabled by default) avoids
# recomputing identical aggregates on repeated calls.
# ---------------------------------------------------------------------------

_cache: dict = {}


def clear_cache() -> None:
    _cache.clear()


def _cached(key: str, producer: Callable):
    """Return a cached aggregate when ANALYTICS_CACHE_TTL_SECONDS > 0, else compute."""
    ttl = settings.ANALYTICS_CACHE_TTL_SECONDS
    if ttl <= 0:
        return producer()
    now = time.monotonic()
    entry = _cache.get(key)
    if entry and entry[0] > now:
        return entry[1]
    value = producer()
    _cache[key] = (now + ttl, value)
    return value


def _round(value, total: int) -> float:
    """Round an SQL AVG (which is None on an empty set) to 2 dp, 0.0 when empty."""
    return round(float(value), 2) if (total and value is not None) else 0.0


def classify_recommendation(overall_score) -> str:
    """Map an overall score to a recruiter recommendation bucket."""
    if overall_score is None:
        return "Rejected"
    if overall_score >= settings.DASHBOARD_SELECTED_MIN:
        return "Selected"
    if overall_score >= settings.DASHBOARD_BORDERLINE_MIN:
        return "Borderline"
    return "Rejected"


def get_overall_statistics(db: Session) -> OverallStatistics:
    sync_history_scores(db)

    def _compute():
        stats = repo.overall_statistics(db)
        selected, borderline, rejected = repo.recommendation_counts(db)
        total = int(stats["total"])
        return OverallStatistics(
            total_analyses=total,
            selected=selected,
            borderline=borderline,
            rejected=rejected,
            average_overall_score=_round(stats["avg_overall"], total),
            average_experience_score=_round(stats["avg_experience"], total),
            average_project_score=_round(stats["avg_project"], total),
            average_quality_score=_round(stats["avg_quality"], total),
            average_coverage_score=_round(stats["avg_coverage"], total),
        )

    result = _cached("overall_statistics", _compute)
    logger.info(f"Analytics overall statistics computed ({result.total_analyses} analyses).")
    return result


def get_score_distribution(db: Session) -> ScoreDistribution:
    sync_history_scores(db)

    def _compute():
        counts = repo.score_distribution(db)
        ranges = [
            ScoreBucket(label=label, min=low, max=high, count=count)
            for (label, low, high), count in zip(SCORE_DISTRIBUTION_BANDS, counts)
        ]
        return ScoreDistribution(total_analyses=sum(counts), ranges=ranges)

    return _cached("score_distribution", _compute)


def get_recommendation_distribution(db: Session) -> RecommendationDistribution:
    sync_history_scores(db)

    def _compute():
        selected, borderline, rejected = repo.recommendation_counts(db)
        total = selected + borderline + rejected

        def pct(n: int) -> float:
            return round(n / total * 100, 2) if total else 0.0

        distribution = [
            RecommendationBucket(label="Selected", count=selected, percentage=pct(selected)),
            RecommendationBucket(label="Borderline", count=borderline, percentage=pct(borderline)),
            RecommendationBucket(label="Rejected", count=rejected, percentage=pct(rejected)),
        ]
        return RecommendationDistribution(total_analyses=total, distribution=distribution)

    return _cached("recommendation_distribution", _compute)


def get_trends(db: Session) -> TrendData:
    sync_history_scores(db)

    def _compute():
        daily = [TrendPoint(period=p, count=c) for p, c in repo.daily_counts(db, settings.ANALYTICS_DAILY_DAYS)]
        weekly = [TrendPoint(period=p, count=c) for p, c in repo.weekly_counts(db, settings.ANALYTICS_WEEKLY_WEEKS)]
        monthly = [TrendPoint(period=p, count=c) for p, c in repo.monthly_counts(db, settings.ANALYTICS_MONTHLY_MONTHS)]
        return TrendData(daily=daily, weekly=weekly, monthly=monthly)

    return _cached("trends", _compute)


def get_top_resumes(db: Session, limit: int) -> list:
    sync_history_scores(db)
    return [TopItem(name=name, count=count) for name, count in repo.top_resumes(db, limit)]


def get_top_job_descriptions(db: Session, limit: int) -> list:
    sync_history_scores(db)
    return [TopItem(name=name, count=count) for name, count in repo.top_job_descriptions(db, limit)]


def get_skill_frequency(db: Session, limit: int) -> SkillFrequency:
    """Aggregate matched requirements and missing skills across every stored report.

    Reads the persisted `analysis_<id>.json` files (the source of truth for the full
    report) and tallies frequencies — this is the one analytic that needs report
    bodies, not just the denormalized score columns. Corrupt/absent files are skipped.
    """
    def _compute():
        matched: Counter = Counter()
        missing: Counter = Counter()
        for path in glob.glob(os.path.join(REPORT_DIR, "analysis_*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    report = json.load(f)
            except Exception:
                continue
            for skill in report.get("missing_skills", []) or []:
                key = str(skill).strip()
                if key:
                    missing[key] += 1
            for req in report.get("requirements", []) or []:
                if req.get("status") == "Matched":
                    key = str(req.get("requirement", "")).strip()
                    if key:
                        matched[key] += 1

        def _top(counter: Counter) -> list:
            return [SkillCount(skill=k, count=c) for k, c in counter.most_common(limit)]

        return SkillFrequency(top_matched=_top(matched), top_missing=_top(missing))

    return _cached(f"skill_frequency_{limit}", _compute)


def get_recent_activity(db: Session, limit: int) -> list:
    sync_history_scores(db)
    rows = repo.recent_activity(db, limit)
    return [
        RecentAnalysisItem(
            analysis_id=analysis_id,
            timestamp=created_at,
            resume_filename=resume_filename or "",
            jd_filename=jd_filename or "",
            overall_score=overall_score,
            recommendation=classify_recommendation(overall_score),
            workflow_status=workflow_status or "Applied",
        )
        for (analysis_id, created_at, resume_filename, jd_filename, overall_score, workflow_status) in rows
    ]
