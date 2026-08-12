from typing import List
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.response import ApiResponse
from app.schemas.resume import ResumeResponse
from app.services.resume import (
    DuplicateResumeError,
    backfill_content_hashes,
    find_duplicate_groups,
    validate_and_save_resume,
)
from app.models.database import Resume
from app.core.logging import logger

router = APIRouter()

@router.post("/upload", response_model=ApiResponse[ResumeResponse])
def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info(f"Resume upload request received. Filename: {file.filename}")
    
    db_resume = validate_and_save_resume(db, file)
    
    response_data = ResumeResponse.model_validate(db_resume)
    
    return ApiResponse[ResumeResponse](
        success=True,
        message="Resume uploaded successfully.",
        data=response_data
    )


@router.post("/upload-bulk")
def upload_resumes_bulk(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Upload multiple resume files at once. Each file is processed independently."""
    logger.info(f"Bulk resume upload request received. File count: {len(files)}")
    results = []
    success_count = 0
    failed_count = 0

    duplicate_count = 0
    for file in files:
        try:
            db_resume = validate_and_save_resume(db, file)
            results.append({
                "filename": file.filename,
                "success": True,
                "data": ResumeResponse.model_validate(db_resume).model_dump(),
            })
            success_count += 1
        except DuplicateResumeError as e:
            # A duplicate is a correct rejection, not an error. In a 100-file batch
            # these are the rows a recruiter wants counted separately, so a genuine
            # failure is not buried among re-uploads of CVs already in the pool.
            logger.info(f"Bulk upload: {file.filename} is a duplicate; skipped.")
            results.append({
                "filename": file.filename,
                "success": False,
                "duplicate": True,
                "existing_id": e.existing.id,
                "existing_filename": e.existing.filename,
                "error": e.message,
            })
            duplicate_count += 1
        except Exception as e:
            logger.error(f"Bulk upload: failed to process {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "duplicate": False,
                "error": str(e),
            })
            failed_count += 1

    parts = [f"{success_count} uploaded"]
    if duplicate_count:
        parts.append(f"{duplicate_count} already in the pool")
    if failed_count:
        parts.append(f"{failed_count} failed")
    if uploaded_ids and not indexing_started:
        parts.append(
            f"too many to index automatically — choose which to make searchable on the "
            f"Documents screen"
        )
    return {
        "success": True,
        "message": ", ".join(parts) + ".",
        "data": {
            "success_count": success_count,
            "failed_count": failed_count,
            "duplicate_count": duplicate_count,
            "total": len(files),
            "results": results,
        },
    }


@router.get("/duplicates", response_model=ApiResponse[dict])
def list_duplicate_resumes(db: Session = Depends(get_db)):
    """
    Duplicates already sitting in the pool, grouped by content.

    Fingerprints resumes uploaded before duplicate detection existed, then reports the
    groups. Nothing is deleted — which copy to keep is the operator's call.
    """
    hashed, groups = backfill_content_hashes(db)
    detail = find_duplicate_groups(db)
    return ApiResponse[dict](
        success=True,
        message=f"{groups} duplicate group(s) found; fingerprinted {hashed} older resume(s).",
        data={"newly_fingerprinted": hashed, "duplicate_groups": len(detail), "groups": detail},
    )


@router.get("", response_model=ApiResponse[List[ResumeResponse]])
def list_resumes(db: Session = Depends(get_db)):
    logger.info("Listing resumes from database.")
    resumes = db.query(Resume).all()
    response_data = [ResumeResponse.model_validate(r) for r in resumes]
    
    return ApiResponse[List[ResumeResponse]](
        success=True,
        message="Resumes retrieved successfully.",
        data=response_data
    )

