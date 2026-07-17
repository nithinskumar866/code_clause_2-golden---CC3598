import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.constants import RESUME_UPLOAD_DIR, JOB_UPLOAD_DIR, STATUS_ANALYSED, STATUS_FAILED
from app.core.logging import logger
from app.schemas.response import ApiResponse
from app.schemas.analysis import AnalysisHistoryItem, AnalysisHistoryDetail
from app.models.database import Resume, JobDescription, Analysis
from app.workflows.hiring_workflow import execute_hiring_pipeline
from app.services import analysis as history_service

router = APIRouter()

@router.post("/evaluate", response_model=ApiResponse[dict])
def evaluate_candidate(resume_id: int, jd_id: int, db: Session = Depends(get_db)):
    logger.info(f"LangGraph State Evaluation Pipeline triggered for Resume ID: {resume_id}, JD ID: {jd_id}")
    
    # 1. Fetch Resume and Job Description from database
    db_resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not db_resume:
        logger.error(f"Resume ID {resume_id} not found in database.")
        raise HTTPException(status_code=404, detail=f"Resume ID {resume_id} not found.")
        
    db_jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()
    if not db_jd:
        logger.error(f"Job Description ID {jd_id} not found in database.")
        raise HTTPException(status_code=404, detail=f"Job Description ID {jd_id} not found.")
        
    # Reconstruct exact disk file paths
    resume_path = os.path.join(RESUME_UPLOAD_DIR, f"{db_resume.id}_{db_resume.filename}")
    jd_path = os.path.join(JOB_UPLOAD_DIR, f"{db_jd.id}_{db_jd.filename}")
    
    if not os.path.exists(resume_path):
        logger.error(f"Resume file not found on disk: {resume_path}")
        raise HTTPException(status_code=404, detail="Resume source document not found on disk.")
        
    if not os.path.exists(jd_path):
        logger.error(f"Job Description file not found on disk: {jd_path}")
        raise HTTPException(status_code=404, detail="Job Description source document not found on disk.")
        
    # 2. Check if an Analysis record already exists, or create a new one
    db_analysis = db.query(Analysis).filter(
        Analysis.resume_id == resume_id,
        Analysis.jd_id == jd_id
    ).first()
    
    if not db_analysis:
        db_analysis = Analysis(resume_id=resume_id, jd_id=jd_id, status="Indexed")
        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)
        
    logger.info(f"Active Analysis Database Record ID: {db_analysis.id}")
    
    try:
        # 3. Ingest, embed, index, retrieve, and evaluate sequentially using LangGraph
        final_report = execute_hiring_pipeline(
            resume_id=resume_id,
            resume_path=resume_path,
            jd_id=jd_id,
            jd_path=jd_path,
            analysis_id=db_analysis.id
        )
        
        # 4. Update Analysis status in database
        db_analysis.status = STATUS_ANALYSED
        db.commit()
        db.refresh(db_analysis)
        
        # Update Resume and JD status to show they are now indexed/processed
        db_resume.status = "Indexed"
        db_jd.status = "Indexed"
        db.commit()
        
        return ApiResponse[dict](
            success=True,
            message="Candidate evaluation completed successfully via LangGraph workflow.",
            data={
                "analysis_id": db_analysis.id,
                "status": db_analysis.status,
                "report": final_report
            }
        )
        
    except Exception as e:
        logger.error(f"Failed during LangGraph workflow execution for ID {db_analysis.id}: {e}", exc_info=True)
        db_analysis.status = STATUS_FAILED
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"LangGraph Hiring State Pipeline failed: {str(e)}"
        )


# --- Analysis History (read/manage over already-persisted evaluations) ---
# These endpoints never run the pipeline; they only expose and manage reports
# that POST /evaluate has already persisted. /evaluate is intentionally untouched.

@router.get("/history", response_model=ApiResponse[List[AnalysisHistoryItem]])
def get_analysis_history(db: Session = Depends(get_db)):
    """Return all completed analyses, newest first."""
    items = history_service.list_history(db)
    return ApiResponse[List[AnalysisHistoryItem]](
        success=True,
        message=f"Retrieved {len(items)} analyses from history.",
        data=items
    )


@router.get("/history/{analysis_id}", response_model=ApiResponse[AnalysisHistoryDetail])
def get_analysis_history_detail(analysis_id: int, db: Session = Depends(get_db)):
    """Return one full stored hiring report by analysis ID."""
    detail = history_service.get_history_detail(db, analysis_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found in history.")
    return ApiResponse[AnalysisHistoryDetail](
        success=True,
        message="Analysis report retrieved.",
        data=detail
    )


@router.delete("/history/{analysis_id}", response_model=ApiResponse[dict])
def delete_analysis_history_item(analysis_id: int, db: Session = Depends(get_db)):
    """Delete a single analysis from history (report file + record)."""
    deleted = history_service.delete_history_item(db, analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found in history.")
    return ApiResponse[dict](
        success=True,
        message="Analysis deleted from history.",
        data={"analysis_id": analysis_id}
    )


@router.delete("/history", response_model=ApiResponse[dict])
def clear_analysis_history(db: Session = Depends(get_db)):
    """Clear all analysis history (all report files + records)."""
    count = history_service.clear_history(db)
    return ApiResponse[dict](
        success=True,
        message=f"Cleared {count} analyses from history.",
        data={"deleted": count}
    )
