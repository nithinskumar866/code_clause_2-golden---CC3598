from typing import List
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.response import ApiResponse
from app.schemas.job import JobDescriptionResponse
from app.services.job import validate_and_save_job_description
from app.models.database import JobDescription
from app.core.logging import logger

router = APIRouter()

@router.post("/upload", response_model=ApiResponse[JobDescriptionResponse])
def upload_job_description(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info(f"Job Description upload request received. Filename: {file.filename}")
    
    db_jd = validate_and_save_job_description(db, file)
    
    response_data = JobDescriptionResponse.model_validate(db_jd)
    
    return ApiResponse[JobDescriptionResponse](
        success=True,
        message="Job Description uploaded successfully.",
        data=response_data
    )


@router.post("/upload-bulk")
def upload_jds_bulk(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Upload multiple job description files at once. Each file is processed independently."""
    logger.info(f"Bulk JD upload request received. File count: {len(files)}")
    results = []
    success_count = 0
    failed_count = 0

    for file in files:
        try:
            db_jd = validate_and_save_job_description(db, file)
            results.append({
                "filename": file.filename,
                "success": True,
                "data": JobDescriptionResponse.model_validate(db_jd).model_dump(),
            })
            success_count += 1
        except Exception as e:
            logger.error(f"Bulk JD upload: failed to process {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
            })
            failed_count += 1

    return {
        "success": True,
        "message": f"{success_count} uploaded, {failed_count} failed.",
        "data": {
            "success_count": success_count,
            "failed_count": failed_count,
            "total": len(files),
            "results": results,
        },
    }


@router.get("", response_model=ApiResponse[List[JobDescriptionResponse]])
def list_job_descriptions(db: Session = Depends(get_db)):
    logger.info("Listing job descriptions from database.")
    jds = db.query(JobDescription).all()
    response_data = [JobDescriptionResponse.model_validate(j) for j in jds]
    
    return ApiResponse[List[JobDescriptionResponse]](
        success=True,
        message="Job Descriptions retrieved successfully.",
        data=response_data
    )

