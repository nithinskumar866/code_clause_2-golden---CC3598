import os
import shutil
from sqlalchemy.orm import Session
from fastapi import UploadFile
from app.core.constants import JOB_UPLOAD_DIR, ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, STATUS_UPLOADED
from app.core.exceptions import UploadError
from app.models.database import JobDescription
from app.core.logging import logger

def validate_and_save_job_description(db: Session, file: UploadFile) -> JobDescription:
    filename = file.filename
    _, ext = os.path.splitext(filename.lower())
    
    # Check extension
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"File upload rejected. Invalid extension: {ext}")
        raise UploadError(
            message=f"Invalid file extension {ext}. Only PDF and DOCX files are allowed."
        )
    
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"File upload rejected. Invalid MIME type: {file.content_type}")
        raise UploadError(
            message=f"Invalid file type {file.content_type}. Only PDF and DOCX files are allowed."
        )
        
    try:
        # 1. Create database record first to get a unique ID for path naming
        db_jd = JobDescription(filename=filename, status=STATUS_UPLOADED)
        db.add(db_jd)
        db.commit()
        db.refresh(db_jd)
        
        # 2. Save the file to JOB_UPLOAD_DIR using index key to prevent collision
        # Example: 1_job.docx
        unique_filename = f"{db_jd.id}_{filename}"
        dest_path = os.path.join(JOB_UPLOAD_DIR, unique_filename)
        
        logger.info(f"Saving uploaded job description to {dest_path}")
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Job Description database record created. ID: {db_jd.id}, Filename on disk: {unique_filename}")
        return db_jd
        
    except Exception as e:
        logger.error(f"Failed to process and save job description: {e}", exc_info=True)
        db.rollback()
        raise UploadError(
            message="Failed to save uploaded job description file.",
            details=str(e)
        )
