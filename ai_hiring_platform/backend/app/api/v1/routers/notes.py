from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.response import ApiResponse
from app.schemas.note import NoteCreate, NoteUpdate, NoteResponse
from app.services import note as note_service

router = APIRouter()


@router.post("/analysis/{analysis_id}/notes", response_model=ApiResponse[NoteResponse])
def create_note(analysis_id: int, payload: NoteCreate, db: Session = Depends(get_db)):
    """Create a recruiter note attached to an analysis."""
    logger.info(f"Create note request for analysis {analysis_id}.")
    note = note_service.create_note(db, analysis_id, payload)
    return ApiResponse[NoteResponse](
        success=True,
        message="Note created.",
        data=NoteResponse.model_validate(note),
    )


@router.get("/analysis/{analysis_id}/notes", response_model=ApiResponse[List[NoteResponse]])
def list_notes(analysis_id: int, db: Session = Depends(get_db)):
    """List all recruiter notes for an analysis, newest first."""
    logger.info(f"List notes request for analysis {analysis_id}.")
    notes = note_service.list_notes(db, analysis_id)
    return ApiResponse[List[NoteResponse]](
        success=True,
        message=f"Retrieved {len(notes)} notes.",
        data=[NoteResponse.model_validate(n) for n in notes],
    )


@router.put("/notes/{note_id}", response_model=ApiResponse[NoteResponse])
def update_note(note_id: int, payload: NoteUpdate, db: Session = Depends(get_db)):
    """Update a recruiter note's text and/or author."""
    logger.info(f"Update note request for note {note_id}.")
    note = note_service.update_note(db, note_id, payload)
    return ApiResponse[NoteResponse](
        success=True,
        message="Note updated.",
        data=NoteResponse.model_validate(note),
    )


@router.delete("/notes/{note_id}", response_model=ApiResponse[dict])
def delete_note(note_id: int, db: Session = Depends(get_db)):
    """Delete a recruiter note by ID."""
    logger.info(f"Delete note request for note {note_id}.")
    note_service.delete_note(db, note_id)
    return ApiResponse[dict](
        success=True,
        message="Note deleted.",
        data={"note_id": note_id},
    )
