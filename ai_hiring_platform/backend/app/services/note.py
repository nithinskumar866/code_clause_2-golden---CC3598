from typing import List

from sqlalchemy.orm import Session

from app.models.database import RecruiterNote, Analysis
from app.schemas.note import NoteCreate, NoteUpdate
from app.core.exceptions import NotFoundError
from app.core.logging import logger

# ---------------------------------------------------------------------------
# Recruiter Notes service.
#
# CRUD over recruiter notes attached to an analysis, persisted in SQLite via the
# existing SQLAlchemy session. Notes are analysis-scoped; deleting an analysis
# cascades to its notes (see RecruiterNote / Analysis.notes relationship).
# ---------------------------------------------------------------------------


def _get_analysis_or_404(db: Session, analysis_id: int) -> Analysis:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise NotFoundError(f"Analysis {analysis_id} not found.")
    return analysis


def _get_note_or_404(db: Session, note_id: int) -> RecruiterNote:
    note = db.query(RecruiterNote).filter(RecruiterNote.id == note_id).first()
    if not note:
        raise NotFoundError(f"Note {note_id} not found.")
    return note


def create_note(db: Session, analysis_id: int, payload: NoteCreate) -> RecruiterNote:
    """Create a note attached to an existing analysis."""
    _get_analysis_or_404(db, analysis_id)
    note = RecruiterNote(analysis_id=analysis_id, text=payload.text, author=payload.author)
    db.add(note)
    db.commit()
    db.refresh(note)
    logger.info(f"Created recruiter note {note.id} on analysis {analysis_id}.")
    return note


def list_notes(db: Session, analysis_id: int) -> List[RecruiterNote]:
    """List all notes for an analysis, newest first."""
    _get_analysis_or_404(db, analysis_id)
    notes = (
        db.query(RecruiterNote)
        .filter(RecruiterNote.analysis_id == analysis_id)
        .order_by(RecruiterNote.created_at.desc(), RecruiterNote.id.desc())
        .all()
    )
    logger.info(f"Retrieved {len(notes)} notes for analysis {analysis_id}.")
    return notes


def update_note(db: Session, note_id: int, payload: NoteUpdate) -> RecruiterNote:
    """Update a note's text and/or author (updated_at is refreshed automatically)."""
    note = _get_note_or_404(db, note_id)
    if payload.text is not None:
        note.text = payload.text
    if payload.author is not None:
        note.author = payload.author
    db.commit()
    db.refresh(note)
    logger.info(f"Updated recruiter note {note_id}.")
    return note


def delete_note(db: Session, note_id: int) -> None:
    """Delete a note by ID."""
    note = _get_note_or_404(db, note_id)
    db.delete(note)
    db.commit()
    logger.info(f"Deleted recruiter note {note_id}.")
