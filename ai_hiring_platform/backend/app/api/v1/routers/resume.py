from typing import List
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.response import ApiResponse
from app.schemas.resume import ResumeResponse
from app.services.resume import validate_and_save_resume
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
