from enum import Enum

from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    """Candidate hiring workflow stages, in recruiter pipeline order. One per analysis."""
    applied = "Applied"
    screening = "Screening"
    reviewed = "Reviewed"
    interview_scheduled = "Interview Scheduled"
    interview_completed = "Interview Completed"
    selected = "Selected"
    rejected = "Rejected"
    offer_sent = "Offer Sent"


# Ordered list for UI stepper / progress rendering (and a single source of truth).
WORKFLOW_ORDER = [s.value for s in WorkflowStatus]


class WorkflowStatusResponse(BaseModel):
    analysis_id: int = Field(..., description="Analysis the status belongs to")
    workflow_status: WorkflowStatus = Field(..., description="Current candidate workflow status")


class WorkflowStatusUpdate(BaseModel):
    workflow_status: WorkflowStatus = Field(..., description="New candidate workflow status")
