from sqlalchemy.orm import Session
from app.models.database import Analysis
from app.core.exceptions import AppException

def create_analysis_placeholder(db: Session, resume_id: int, jd_id: int) -> Analysis:
    """
    Placeholder service to create an analysis job records.
    Throws a placeholder exception or returns a dummy record.
    """
    raise AppException(
        message="AI Analysis is not implemented in Sprint 1.",
        code="NOT_IMPLEMENTED",
        status_code=501,
        details="Complete Sprint 2 to enable AI analysis features."
    )
