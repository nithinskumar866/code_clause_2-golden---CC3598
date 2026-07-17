from enum import Enum

from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    """Candidate hiring workflow stages. One per analysis."""
    applied = "Applied"
    screening = "Screening"
    technical = "Technical"
    manager = "Manager"
    hr = "HR"
    offer = "Offer"
    joined = "Joined"
    rejected = "Rejected"


class WorkflowStatusResponse(BaseModel):
    analysis_id: int = Field(..., description="Analysis the status belongs to")
    workflow_status: WorkflowStatus = Field(..., description="Current candidate workflow status")


class WorkflowStatusUpdate(BaseModel):
    workflow_status: WorkflowStatus = Field(..., description="New candidate workflow status")
