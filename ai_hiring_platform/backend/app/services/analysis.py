import os
import json
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.database import Analysis, Resume, JobDescription
from app.core.constants import REPORT_DIR
from app.core.logging import logger
from app.schemas.analysis import (
    HiringReport,
    AnalysisHistoryItem,
    AnalysisHistoryDetail,
)

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
    resume: Optional[Resume],
    jd: Optional[JobDescription],
) -> AnalysisHistoryItem:
    return AnalysisHistoryItem(
        analysis_id=analysis.id,
        timestamp=analysis.created_at,
        resume_id=analysis.resume_id,
        job_description_id=analysis.jd_id,
        resume_filename=resume.filename if resume else "",
        jd_filename=jd.filename if jd else "",
        overall_score=report.get("overall_score", 0),
        coverage_score=report.get("coverage_score", 0),
        experience_score=report.get("experience_score", 0),
        project_score=report.get("project_score", 0),
        quality_score=report.get("quality_score", 0),
        recruiter_recommendation=report.get("recruiter_recommendation", ""),
        summary=report.get("summary", ""),
    )


def list_history(db: Session) -> List[AnalysisHistoryItem]:
    """Return all completed analyses as summary rows, newest first."""
    analyses = (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .all()
    )
    items: List[AnalysisHistoryItem] = []
    for analysis in analyses:
        report = _load_report(analysis.id)
        if not _is_completed(report):
            continue  # history begins only after evaluation completes
        resume = db.query(Resume).filter(Resume.id == analysis.resume_id).first()
        jd = db.query(JobDescription).filter(JobDescription.id == analysis.jd_id).first()
        items.append(_build_item(analysis, report, resume, jd))
    return items


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
    item = _build_item(analysis, report, resume, jd)
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
