import hashlib
import os
from typing import Optional, Tuple

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.constants import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    RESUME_UPLOAD_DIR,
    STATUS_UPLOADED,
)
from app.core.exceptions import UploadError
from app.core.logging import logger
from app.models.database import Resume


class DuplicateResumeError(UploadError):
    """
    The exact same file content is already in the pool.

    Not a failure of the upload — a correct rejection. Indexing the same CV twice puts
    the candidate in every search result twice and skews any ranking they appear in,
    which is exactly what a recruiter notices first.
    """

    def __init__(self, existing: Resume) -> None:
        super().__init__(
            message=(
                f"This resume is already in the pool as \"{existing.filename}\" "
                f"(ID #{existing.id}), uploaded {existing.upload_time:%d %b %Y}."
            )
        )
        self.existing = existing


def compute_content_hash(data: bytes) -> str:
    """SHA-256 of the file bytes — identical content, identical hash, any filename."""
    return hashlib.sha256(data).hexdigest()


def find_duplicate(db: Session, content_hash: str) -> Optional[Resume]:
    return db.query(Resume).filter(Resume.content_hash == content_hash).first()


def _validate(file: UploadFile) -> None:
    _, ext = os.path.splitext((file.filename or "").lower())
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"File upload rejected. Invalid extension: {ext}")
        raise UploadError(
            message=f"Invalid file extension {ext}. Only PDF and DOCX files are allowed."
        )
    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"File upload rejected. Invalid MIME type: {file.content_type}")
        raise UploadError(
            message=f"Invalid file type {file.content_type}. Only PDF and DOCX files are allowed."
        )


def validate_and_save_resume(
    db: Session, file: UploadFile, allow_duplicate: bool = False
) -> Resume:
    """
    Validate, de-duplicate and store one resume.

    Duplicate detection is by CONTENT, not filename: the same CV saved as
    `resume_final.docx` and `resume_final_v2.docx` is one candidate, and matching on
    name would miss it. Set `allow_duplicate` to keep a second copy deliberately.
    """
    filename = file.filename
    _validate(file)

    try:
        data = file.file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded resume: {e}", exc_info=True)
        raise UploadError(message="Failed to read uploaded resume file.", details=str(e))

    if not data:
        raise UploadError(message="The uploaded file is empty.")

    content_hash = compute_content_hash(data)
    if not allow_duplicate:
        existing = find_duplicate(db, content_hash)
        if existing is not None:
            logger.info(
                f"Duplicate resume rejected: {filename!r} matches existing ID {existing.id}."
            )
            raise DuplicateResumeError(existing)

    try:
        # DB row first so the id can name the file on disk (matches the original flow).
        db_resume = Resume(filename=filename, status=STATUS_UPLOADED, content_hash=content_hash)
        db.add(db_resume)
        db.commit()
        db.refresh(db_resume)

        unique_filename = f"{db_resume.id}_{filename}"
        dest_path = os.path.join(RESUME_UPLOAD_DIR, unique_filename)
        logger.info(f"Saving uploaded resume to {dest_path}")
        with open(dest_path, "wb") as buffer:
            buffer.write(data)

        logger.info(
            f"Resume database record created. ID: {db_resume.id}, on disk: {unique_filename}"
        )
        return db_resume

    except Exception as e:
        logger.error(f"Failed to process and save resume: {e}", exc_info=True)
        db.rollback()
        raise UploadError(message="Failed to save uploaded resume file.", details=str(e))


def backfill_content_hashes(db: Session) -> Tuple[int, int]:
    """
    Fingerprint resumes uploaded before duplicate detection existed.

    Returns (hashed, duplicate_groups). Existing rows are never deleted — the operator
    decides what to remove; this only makes the duplicates visible.
    """
    hashed = 0
    for resume in db.query(Resume).filter(Resume.content_hash.is_(None)).all():
        path = os.path.join(RESUME_UPLOAD_DIR, f"{resume.id}_{resume.filename}")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                resume.content_hash = compute_content_hash(f.read())
            hashed += 1
        except OSError as e:
            logger.warning(f"Could not fingerprint resume {resume.id}: {e}")
    if hashed:
        db.commit()

    seen: dict = {}
    for resume in db.query(Resume).filter(Resume.content_hash.isnot(None)).all():
        seen.setdefault(resume.content_hash, []).append(resume)
    groups = sum(1 for rows in seen.values() if len(rows) > 1)
    return hashed, groups


def find_duplicate_groups(db: Session) -> list:
    """Existing duplicates, newest copies listed after the original."""
    by_hash: dict = {}
    for resume in db.query(Resume).filter(Resume.content_hash.isnot(None)).order_by(Resume.id).all():
        by_hash.setdefault(resume.content_hash, []).append(resume)
    return [
        {
            "content_hash": h,
            "count": len(rows),
            "keep": {"id": rows[0].id, "filename": rows[0].filename},
            "duplicates": [{"id": r.id, "filename": r.filename} for r in rows[1:]],
        }
        for h, rows in by_hash.items()
        if len(rows) > 1
    ]
