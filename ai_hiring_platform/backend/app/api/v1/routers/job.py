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
