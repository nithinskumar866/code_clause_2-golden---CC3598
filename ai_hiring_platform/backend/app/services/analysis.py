import os
import json
import math
from datetime import date
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.models.database import Analysis, Resume, JobDescription
from app.core.constants import REPORT_DIR
from app.core.logging import logger
from app.repositories import history_repository
from app.schemas.analysis import (
    HiringReport,
    AnalysisHistoryItem,
    AnalysisHistoryDetail,
)
from app.schemas.history import HistoryPageMeta

# ---------------------------------------------------------------------------
# Analysis History service.
#
# History is NOT a new store: it is a read/manage layer over the persistence the
# evaluation pipeline already writes -- the Analysis DB row (timestamp + ids),
# the Resume/JobDescription rows (filenames), and the per-analysis report file at
# storage/reports/analysis_<id>.json (the full hiring report). Evaluation is left
# untouched; an entry appears here only once its report file has been written.
# ---------------------------------------------------------------------------


def _report_path(analysis_id: int) -> str:
    return os.path.join(REPORT_DIR, f"analysis_{analysis_id}.json")


def _load_report(analysis_id: int) -> Optional[dict]:
    """Read the persisted report JSON for an analysis, or None if unavailable."""
    path = _report_path(analysis_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read report file for analysis {analysis_id}: {e}", exc_info=True)
        return None


def _is_completed(report: Optional[dict]) -> bool:
    """A history entry exists only after evaluation completed and wrote scores."""
    return bool(report) and "overall_score" in report


def _build_item(
    analysis: Analysis,
    report: dict,
    resume_filename: str,
    jd_filename: str,
) -> AnalysisHistoryItem:
    # overall_score is authoritative from the denormalized column; fall back to
    # the report file when the column is not yet backfilled.
    overall = analysis.overall_score
    if overall is None:
        overall = report.get("overall_score", 0)
    return AnalysisHistoryItem(
        analysis_id=analysis.id,
        timestamp=analysis.created_at,
        resume_id=analysis.resume_id,
        job_description_id=analysis.jd_id,
        resume_filename=resume_filename or "",
        jd_filename=jd_filename or "",
        workflow_status=getattr(analysis, "workflow_status", None) or "Applied",
        overall_score=overall,
        coverage_score=report.get("coverage_score", 0),
        experience_score=report.get("experience_score", 0),
        project_score=report.get("project_score", 0),
        quality_score=report.get("quality_score", 0),
        recruiter_recommendation=report.get("recruiter_recommendation", ""),
        summary=report.get("summary", ""),
    )


def _row_to_item(row) -> AnalysisHistoryItem:
    """Enrich one repository row (Analysis, resume_filename, jd_filename)."""
    analysis, resume_filename, jd_filename = row
    report = _load_report(analysis.id) or {}
    return _build_item(analysis, report, resume_filename, jd_filename)


def sync_history_scores(db: Session) -> None:
    """
    Backfill the denormalized Analysis score columns (overall + sub-scores) from
    persisted report files for any completed analysis not yet denormalized.
    Idempotent: only rows with a NULL overall_score are touched, so this is a
    no-op once caught up. Keeps the evaluation pipeline untouched while enabling
    SQL-native score filtering and aggregation.
    """
    pending = db.query(Analysis).filter(Analysis.overall_score.is_(None)).all()
    updated = 0
    for analysis in pending:
        report = _load_report(analysis.id)
        if report and report.get("overall_score") is not None:
            analysis.overall_score = int(report.get("overall_score") or 0)
            analysis.coverage_score = int(report.get("coverage_score") or 0)
            analysis.experience_score = int(report.get("experience_score") or 0)
            analysis.project_score = int(report.get("project_score") or 0)
            analysis.quality_score = int(report.get("quality_score") or 0)
            updated += 1
    if updated:
        db.commit()
        logger.info(f"Backfilled denormalized scores for {updated} analyses.")


def _empty_filters() -> dict:
    return {
        "resume_filename": None,
        "jd_filename": None,
        "recommendation": None,
        "min_score": None,
        "max_score": None,
        "date_from": None,
        "date_to": None,
    }


def list_history(db: Session) -> List[AnalysisHistoryItem]:
    """Return all completed analyses as summary rows, newest first (unpaged)."""
    sync_history_scores(db)
    rows, _ = history_repository.search(db, filters=_empty_filters(), sort="newest", limit=None)
    return [_row_to_item(row) for row in rows]


def search_history(
    db: Session,
    *,
    resume_filename: Optional[str] = None,
    jd_filename: Optional[str] = None,
    recommendation: Optional[str] = None,
    min_score: Optional[int] = None,
    max_score: Optional[int] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    sort: str = "newest",
    page: int = 1,
    page_size: Optional[int] = None,
) -> Tuple[List[AnalysisHistoryItem], HistoryPageMeta]:
    """
    Search/filter/sort/paginate completed analyses. When page_size is None the
    full matching set is returned (backward-compatible with the original
    unpaged endpoint); otherwise LIMIT/OFFSET pagination is applied in SQL.
    """
    sync_history_scores(db)
    filters = {
        "resume_filename": resume_filename,
        "jd_filename": jd_filename,
        "recommendation": recommendation,
        "min_score": min_score,
        "max_score": max_score,
        "date_from": date_from,
        "date_to": date_to,
    }

    if page_size is None:
        rows, total = history_repository.search(db, filters=filters, sort=sort, limit=None)
        meta = HistoryPageMeta(total_count=total, page=1, page_size=total, total_pages=1 if total else 0)
    else:
        offset = (page - 1) * page_size
        rows, total = history_repository.search(db, filters=filters, sort=sort, limit=page_size, offset=offset)
        total_pages = math.ceil(total / page_size) if total else 0
        meta = HistoryPageMeta(total_count=total, page=page, page_size=page_size, total_pages=total_pages)

    items = [_row_to_item(row) for row in rows]
    return items, meta


def get_history_detail(db: Session, analysis_id: int) -> Optional[AnalysisHistoryDetail]:
    """Return one full history entry (metadata + full hiring report) or None."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        return None
    report = _load_report(analysis_id)
    if not _is_completed(report):
        return None
    resume = db.query(Resume).filter(Resume.id == analysis.resume_id).first()
    jd = db.query(JobDescription).filter(JobDescription.id == analysis.jd_id).first()
    item = _build_item(
        analysis,
        report,
        resume.filename if resume else "",
        jd.filename if jd else "",
    )
    # Reuse the existing HiringReport schema; extra evidence keys are ignored.
    return AnalysisHistoryDetail(
        **item.model_dump(),
        report=HiringReport.model_validate(report),
    )


def delete_history_item(db: Session, analysis_id: int) -> bool:
    """Delete one history entry: its report file and its Analysis row.

    Returns False if there was nothing to delete (unknown analysis_id).
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    path = _report_path(analysis_id)
    file_exists = os.path.exists(path)

    if not analysis and not file_exists:
        return False

    if file_exists:
        try:
            os.remove(path)
        except OSError as e:
            logger.error(f"Failed to remove report file for analysis {analysis_id}: {e}", exc_info=True)

    if analysis:
        db.delete(analysis)
        db.commit()

    return True


def clear_history(db: Session) -> int:
    """Delete all history: every analysis's report file and row.

    Returns the count of entries that had a persisted report (true history rows).
    """
    analyses = db.query(Analysis).all()
    cleared = 0
    for analysis in analyses:
        path = _report_path(analysis.id)
        if os.path.exists(path):
            try:
                os.remove(path)
                cleared += 1
            except OSError as e:
                logger.error(f"Failed to remove report file for analysis {analysis.id}: {e}", exc_info=True)
        db.delete(analysis)
    db.commit()
    return cleared
