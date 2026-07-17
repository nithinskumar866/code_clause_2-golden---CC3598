from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.response import ApiResponse
from app.schemas.status import WorkflowStatusResponse, WorkflowStatusUpdate
from app.services import workflow as workflow_service

router = APIRouter()


@router.get("/analysis/{analysis_id}/status", response_model=ApiResponse[WorkflowStatusResponse])
def get_workflow_status(analysis_id: int, db: Session = Depends(get_db)):
    """Get the candidate workflow status for an analysis."""
    logger.info(f"Get workflow status for analysis {analysis_id}.")
    analysis = workflow_service.get_status(db, analysis_id)
    return ApiResponse[WorkflowStatusResponse](
        success=True,
        message="Workflow status retrieved.",
        data=WorkflowStatusResponse(analysis_id=analysis.id, workflow_status=analysis.workflow_status),
    )


@router.patch("/analysis/{analysis_id}/status", response_model=ApiResponse[WorkflowStatusResponse])
def update_workflow_status(analysis_id: int, payload: WorkflowStatusUpdate, db: Session = Depends(get_db)):
    """Update the candidate workflow status for an analysis."""
    logger.info(f"Update workflow status for analysis {analysis_id} -> {payload.workflow_status.value}.")
    analysis = workflow_service.update_status(db, analysis_id, payload.workflow_status)
    return ApiResponse[WorkflowStatusResponse](
        success=True,
        message="Workflow status updated.",
        data=WorkflowStatusResponse(analysis_id=analysis.id, workflow_status=analysis.workflow_status),
    )
