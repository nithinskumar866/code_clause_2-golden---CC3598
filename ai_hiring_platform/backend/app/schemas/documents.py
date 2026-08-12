"""
API contract for document management (`/api/v1/documents`).

The screen this serves answers one question a 300-CV pool makes expensive: *which*
resumes are worth chunking and embedding right now. Uploading and indexing are separate
acts here, so the contract exposes the working set as first-class state rather than as a
side effect of upload.

Backend owns this contract; `frontend/src/types/index.ts` mirrors it.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ManagedResume(BaseModel):
    """One uploaded resume and everything the selection screen needs to decide about it."""

    id: int
    filename: str
    upload_time: datetime
    status: str
    # False when the row survives but the file behind it does not — such a resume can
    # never be re-indexed, and the screen must be able to say so.
    file_present: bool = True
    # In the working set: parsed and chunked into the shared document layer, and
    # therefore eligible for embedding. Deliberately narrower than "uploaded".
    in_working_set: bool = False
    chunks: int = 0
    # Deterministic profile fields, available only once the resume has been parsed.
    name: Optional[str] = None
    title: Optional[str] = None
    total_years: Optional[float] = None
    location: Optional[str] = None
    # {model -> holds vectors for this resume}. A model can legitimately be behind.
    models: Dict[str, bool] = {}


class ManagedResumeList(BaseModel):
    resumes: List[ManagedResume] = []
    total: int = 0
    in_working_set: int = 0
    # {model -> how many resumes that model holds vectors for}.
    models: Dict[str, int] = {}


class ManagedJob(BaseModel):
    """
    One uploaded job description.

    JDs carry no indexing state: a JD is parsed for its requirements at analysis time
    and never chunked or embedded, so there is no working-set decision to make.
    """

    id: int
    filename: str
    upload_time: datetime
    status: str
    file_present: bool = True
    analyses: int = 0


class ManagedJobList(BaseModel):
    jobs: List[ManagedJob] = []
    total: int = 0


class SelectionRequest(BaseModel):
    """A batch of documents the recruiter selected on screen."""

    ids: List[int] = Field(..., description="Resume (or job) ids the action applies to.")


class WorkingSetResult(BaseModel):
    added: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: List[str] = []
    chunks: int = 0
    working_set: int = 0
    # Models realigned after a removal. No embedding happens — the dropped chunks
    # simply stop being included.
    realigned: List[str] = []


class DeleteResult(BaseModel):
    deleted: int = 0
    files_removed: int = 0


class ResumeSection(BaseModel):
    """One section of a parsed resume, as the viewer renders it."""

    section: str
    page: int = 1
    text: str


class ResumeContent(BaseModel):
    """
    A resume as the platform actually holds it.

    `in_working_set` is false when the resume has been uploaded but not yet chunked —
    there is nothing to show, and saying so is better than an empty panel.
    """

    resume_id: int
    filename: Optional[str] = None
    in_working_set: bool = False
    profile: Dict[str, Any] = {}
    sections: List[ResumeSection] = []
    chunks: int = 0
