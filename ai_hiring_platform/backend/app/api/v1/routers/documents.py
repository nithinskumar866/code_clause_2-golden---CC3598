"""
Document management — choosing what is searchable, and removing what is not wanted.

Thin by design (Golden Rule 7): every endpoint delegates to `services/document_manager`
and wraps the result in the standard `{success, message, data}` envelope.

Two removals exist here on purpose, and they are not the same operation:

    POST /documents/resumes/working-set/remove   un-chunk, drop vectors, KEEP the file
    POST /documents/resumes/delete               file, database row and vectors, gone

(POST rather than DELETE for the destructive one: it carries a batch of ids in its body,
and a DELETE with a request body is unevenly supported by proxies and HTTP clients.)

The first is a cost decision and is fully reversible — re-adding a resume costs one
parse. The second is destructive. Collapsing them into one button would mean a recruiter
could not take an expensive model off a candidate without losing the candidate.
"""
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.documents import (
    DeleteResult,
    ResumeContent,
    ManagedJobList,
    ManagedResumeList,
    SelectionRequest,
    WorkingSetResult,
)
from app.schemas.response import ApiResponse
from app.services import document_manager

router = APIRouter()


@router.get("/resumes", response_model=ApiResponse[ManagedResumeList])
def list_resumes(db: Session = Depends(get_db)):
    """
    Every uploaded resume with its indexing state.

    Reports what is uploaded, what is in the working set, and which models hold vectors
    for each — so a pool of 300 uploaded / 50 searchable / 20 through the slow model is
    visible at a glance rather than inferred.
    """
    return ApiResponse[ManagedResumeList](
        success=True,
        message="Resumes retrieved.",
        data=ManagedResumeList(**document_manager.list_resumes(db)),
    )


@router.get("/jobs", response_model=ApiResponse[ManagedJobList])
def list_jobs(db: Session = Depends(get_db)):
    """Every uploaded job description. JDs are never chunked, so they carry no index state."""
    return ApiResponse[ManagedJobList](
        success=True,
        message="Job descriptions retrieved.",
        data=ManagedJobList(**document_manager.list_jobs(db)),
    )


@router.post("/resumes/working-set/add", response_model=ApiResponse[WorkingSetResult])
def add_to_working_set(payload: SelectionRequest, db: Session = Depends(get_db)):
    """
    Parse and chunk the selected resumes so they can be embedded.

    Model-independent and fast, so it runs synchronously. Embedding is deliberately a
    separate call: adding 200 resumes here costs seconds, embedding them costs minutes
    per model, and those are two decisions a recruiter should make separately.
    """
    logger.info(f"Working set: adding {len(payload.ids)} resume(s).")
    return ApiResponse[WorkingSetResult](
        success=True,
        message=f"{len(payload.ids)} resume(s) added to the working set. Index them to make them searchable.",
        data=WorkingSetResult(**document_manager.add_to_working_set(db, payload.ids)),
    )


@router.post("/resumes/working-set/remove", response_model=ApiResponse[WorkingSetResult])
def remove_from_working_set(payload: SelectionRequest):
    """
    Drop the selected resumes' chunks and vectors. The files and database rows survive.

    Instant: realigning each model is a file write, not an embedding run, because the
    remaining vectors are reused and the removed chunks simply stop being included.
    """
    logger.info(f"Working set: removing {len(payload.ids)} resume(s).")
    return ApiResponse[WorkingSetResult](
        success=True,
        message=f"{len(payload.ids)} resume(s) removed from the index. The files are untouched.",
        data=WorkingSetResult(**document_manager.remove_from_working_set(payload.ids)),
    )


@router.post("/resumes/delete", response_model=ApiResponse[DeleteResult])
def delete_resumes(payload: SelectionRequest, db: Session = Depends(get_db)):
    """
    Remove resumes completely — vectors, chunks, database row and the file on disk.

    Destructive and not reversible. Any analyses referencing the resume go with it via
    the model's cascade, which is why this is a separate endpoint from the working-set
    removal rather than a flag on it.
    """
    logger.info(f"Deleting {len(payload.ids)} resume(s) permanently.")
    return ApiResponse[DeleteResult](
        success=True,
        message=f"{len(payload.ids)} resume(s) deleted permanently.",
        data=DeleteResult(**document_manager.delete_resumes(db, payload.ids)),
    )


@router.post("/jobs/delete", response_model=ApiResponse[DeleteResult])
def delete_jobs(payload: SelectionRequest, db: Session = Depends(get_db)):
    """Remove job descriptions completely. JDs hold no vectors, so this is row + file."""
    logger.info(f"Deleting {len(payload.ids)} job description(s).")
    return ApiResponse[DeleteResult](
        success=True,
        message=f"{len(payload.ids)} job description(s) deleted.",
        data=DeleteResult(**document_manager.delete_jobs(db, payload.ids)),
    )


@router.get("/resumes/{resume_id}/content", response_model=ApiResponse[ResumeContent])
def read_resume(resume_id: int):
    """
    The parsed resume, section by section — what the "view resume" panel renders.

    Served from the document layer, not by re-parsing the file, so a recruiter reads
    exactly the text the search matched on. Re-parsing could show something subtly
    different from what was scored, which would make the evidence trail a lie.
    """
    return ApiResponse[ResumeContent](
        success=True,
        message="Resume content retrieved.",
        data=ResumeContent(**document_manager.read_resume(resume_id)),
    )


@router.get("/resumes/{resume_id}/file")
def download_resume(resume_id: int, db: Session = Depends(get_db)):
    """
    The ORIGINAL uploaded PDF/DOCX.

    The parsed view shows what was indexed; this is the document the candidate actually
    sent, formatting intact. Recruiters need both — one to see why they matched, one to
    forward to a hiring manager.
    """
    path = document_manager.resume_file_path(db, resume_id)
    if not path:
        raise HTTPException(status_code=404, detail="No file on disk for this resume.")
    return FileResponse(path, filename=os.path.basename(path).split("_", 1)[-1])
