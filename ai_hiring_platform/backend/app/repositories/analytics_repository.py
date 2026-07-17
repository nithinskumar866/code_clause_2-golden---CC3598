from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

from sqlalchemy import func, case
from sqlalchemy.orm import Session

from app.models.database import Analysis, Resume, JobDescription
from app.core.config import settings
from app.core.constants import SCORE_DISTRIBUTION_BANDS

# ---------------------------------------------------------------------------
# Analytics repository.
#
# All history analytics are computed here with SQL aggregation (COUNT / AVG /
# SUM(CASE) / GROUP BY). Aggregates return a handful of rows at most -- the full
# analysis set is never loaded into memory. "Completed" = Analysis.overall_score
# is not NULL (denormalized from a finished report).
# ---------------------------------------------------------------------------

_COMPLETED = Analysis.overall_score.isnot(None)


def overall_statistics(db: Session) -> dict:
    """Total count and average of each denormalized score, in one query."""
    row = (
        db.query(
            func.count(Analysis.id),
            func.avg(Analysis.overall_score),
            func.avg(Analysis.coverage_score),
            func.avg(Analysis.experience_score),
            func.avg(Analysis.project_score),
            func.avg(Analysis.quality_score),
        )
        .filter(_COMPLETED)
        .one()
    )
    return {
        "total": row[0] or 0,
        "avg_overall": row[1],
        "avg_coverage": row[2],
        "avg_experience": row[3],
        "avg_project": row[4],
        "avg_quality": row[5],
    }


def recommendation_counts(db: Session) -> Tuple[int, int, int]:
    """(selected, borderline, rejected) counts via score-threshold buckets."""
    selected_min = settings.DASHBOARD_SELECTED_MIN
    borderline_min = settings.DASHBOARD_BORDERLINE_MIN
    row = (
        db.query(
            func.coalesce(func.sum(case((Analysis.overall_score >= selected_min, 1), else_=0)), 0),
            func.coalesce(
                func.sum(
                    case(
                        ((Analysis.overall_score >= borderline_min) & (Analysis.overall_score < selected_min), 1),
                        else_=0,
                    )
                ),
                0,
            ),
            func.coalesce(func.sum(case((Analysis.overall_score < borderline_min, 1), else_=0)), 0),
        )
        .filter(_COMPLETED)
        .one()
    )
    return int(row[0]), int(row[1]), int(row[2])


def score_distribution(db: Session) -> List[int]:
    """Counts per fixed score band (parallel to SCORE_DISTRIBUTION_BANDS)."""
    columns = [
        func.coalesce(func.sum(case((Analysis.overall_score.between(low, high), 1), else_=0)), 0)
        for _, low, high in SCORE_DISTRIBUTION_BANDS
    ]
    row = db.query(*columns).filter(_COMPLETED).one()
    return [int(c) for c in row]


def _grouped_counts(db: Session, expr, since: Optional[datetime]) -> List[Tuple[str, int]]:
    """GROUP BY an SQL date expression, optionally windowed by `since`."""
    query = db.query(expr.label("period"), func.count(Analysis.id)).filter(_COMPLETED)
    if since is not None:
        query = query.filter(Analysis.created_at >= since)
    rows = query.group_by("period").order_by("period").all()
    return [(str(period), int(count)) for period, count in rows]


def daily_counts(db: Session, days: int) -> List[Tuple[str, int]]:
    since = datetime.utcnow() - timedelta(days=days - 1)
    since = datetime(since.year, since.month, since.day)
    return _grouped_counts(db, func.date(Analysis.created_at), since)


def weekly_counts(db: Session, weeks: int) -> List[Tuple[str, int]]:
    since = datetime.utcnow() - timedelta(weeks=weeks)
    return _grouped_counts(db, func.strftime("%Y-W%W", Analysis.created_at), since)


def monthly_counts(db: Session, months: int) -> List[Tuple[str, int]]:
    since = datetime.utcnow() - timedelta(days=months * 31)
    return _grouped_counts(db, func.strftime("%Y-%m", Analysis.created_at), since)


def top_resumes(db: Session, limit: int) -> List[Tuple[str, int]]:
    rows = (
        db.query(Resume.filename, func.count(Analysis.id))
        .join(Analysis, Analysis.resume_id == Resume.id)
        .filter(_COMPLETED)
        .group_by(Resume.filename)
        .order_by(func.count(Analysis.id).desc(), Resume.filename.asc())
        .limit(limit)
        .all()
    )
    return [(name, int(count)) for name, count in rows]


def top_job_descriptions(db: Session, limit: int) -> List[Tuple[str, int]]:
    rows = (
        db.query(JobDescription.filename, func.count(Analysis.id))
        .join(Analysis, Analysis.jd_id == JobDescription.id)
        .filter(_COMPLETED)
        .group_by(JobDescription.filename)
        .order_by(func.count(Analysis.id).desc(), JobDescription.filename.asc())
        .limit(limit)
        .all()
    )
    return [(name, int(count)) for name, count in rows]


def recent_activity(db: Session, limit: int) -> List[Tuple]:
    """Latest analyses (joined for filenames) — LIMITed, never the full set."""
    return (
        db.query(
            Analysis.id,
            Analysis.created_at,
            Resume.filename,
            JobDescription.filename,
            Analysis.overall_score,
            Analysis.workflow_status,
        )
        .join(Resume, Analysis.resume_id == Resume.id)
        .join(JobDescription, Analysis.jd_id == JobDescription.id)
        .filter(_COMPLETED)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(limit)
        .all()
    )
