from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Query, Session

from app.models.database import Analysis, Resume, JobDescription
from app.core.config import settings

# ---------------------------------------------------------------------------
# History repository.
#
# Single source of truth for the Analysis-History query. Builds an efficient
# SQLAlchemy query joining Resume/JobDescription for filenames and filtering on
# the denormalized Analysis.overall_score. All filtering, ordering, counting and
# pagination happen in SQL -- rows are never loaded into memory to be filtered.
# ---------------------------------------------------------------------------


def _sort_columns(sort: str):
    """Map a sort key to ORDER BY columns (with a stable id tie-breaker)."""
    mapping = {
        "newest": (Analysis.created_at.desc(), Analysis.id.desc()),
        "oldest": (Analysis.created_at.asc(), Analysis.id.asc()),
        "highest_score": (Analysis.overall_score.desc(), Analysis.id.desc()),
        "lowest_score": (Analysis.overall_score.asc(), Analysis.id.asc()),
    }
    return mapping.get(sort, mapping["newest"])


def _build_filtered_query(
    db: Session,
    *,
    resume_filename: Optional[str] = None,
    jd_filename: Optional[str] = None,
    recommendation: Optional[str] = None,
    min_score: Optional[int] = None,
    max_score: Optional[int] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> Query:
    """Build the joined, filtered (but unordered, unpaginated) history query."""
    query = (
        db.query(
            Analysis,
            Resume.filename.label("resume_filename"),
            JobDescription.filename.label("jd_filename"),
        )
        .join(Resume, Analysis.resume_id == Resume.id)
        .join(JobDescription, Analysis.jd_id == JobDescription.id)
        # Only completed analyses (score denormalized from a finished report).
        .filter(Analysis.overall_score.isnot(None))
    )

    if resume_filename:
        query = query.filter(func.lower(Resume.filename).like(f"%{resume_filename.lower()}%"))
    if jd_filename:
        query = query.filter(func.lower(JobDescription.filename).like(f"%{jd_filename.lower()}%"))

    if recommendation:
        selected_min = settings.DASHBOARD_SELECTED_MIN
        borderline_min = settings.DASHBOARD_BORDERLINE_MIN
        if recommendation == "Selected":
            query = query.filter(Analysis.overall_score >= selected_min)
        elif recommendation == "Borderline":
            query = query.filter(
                Analysis.overall_score >= borderline_min,
                Analysis.overall_score < selected_min,
            )
        elif recommendation == "Rejected":
            query = query.filter(Analysis.overall_score < borderline_min)

    if min_score is not None:
        query = query.filter(Analysis.overall_score >= min_score)
    if max_score is not None:
        query = query.filter(Analysis.overall_score <= max_score)

    if date_from is not None:
        start = datetime(date_from.year, date_from.month, date_from.day)
        query = query.filter(Analysis.created_at >= start)
    if date_to is not None:
        # Inclusive of the whole `date_to` day.
        end = datetime(date_to.year, date_to.month, date_to.day) + timedelta(days=1)
        query = query.filter(Analysis.created_at < end)

    return query


def search(
    db: Session,
    *,
    filters: dict,
    sort: str = "newest",
    limit: Optional[int] = None,
    offset: int = 0,
) -> Tuple[List, int]:
    """
    Return (rows, total_count) for the given filters.

    - total_count is computed in SQL over the filtered set (COUNT, no rows loaded).
    - rows are ordered and, when limit is given, paginated via LIMIT/OFFSET.
    - each row is (Analysis, resume_filename, jd_filename).
    """
    base = _build_filtered_query(db, **filters)
    total = base.count()

    query = base.order_by(*_sort_columns(sort))
    if limit is not None:
        query = query.limit(limit).offset(offset)

    return query.all(), total
