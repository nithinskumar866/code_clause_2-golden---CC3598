from sqlalchemy.orm import Session

from app.models.database import Analysis
from app.schemas.status import WorkflowStatus
from app.core.exceptions import NotFoundError
from app.core.logging import logger

# ---------------------------------------------------------------------------
# Candidate workflow status service.
#
# Each analysis carries exactly one workflow status (Analysis.workflow_status).
# This reuses the existing Analysis row for persistence; there is no separate
# store. The evaluation pipeline is not involved.
# ---------------------------------------------------------------------------


def _get_analysis_or_404(db: Session, analysis_id: int) -> Analysis:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise NotFoundError(f"Analysis {analysis_id} not found.")
    return analysis


def get_status(db: Session, analysis_id: int) -> Analysis:
    """Return the analysis (carrying its current workflow_status)."""
    return _get_analysis_or_404(db, analysis_id)


def update_status(db: Session, analysis_id: int, new_status: WorkflowStatus) -> Analysis:
    """Set the analysis's workflow status to a validated value."""
    analysis = _get_analysis_or_404(db, analysis_id)
    analysis.workflow_status = new_status.value
    db.commit()
    db.refresh(analysis)
    logger.info(f"Analysis {analysis_id} workflow status set to '{new_status.value}'.")
    return analysis
