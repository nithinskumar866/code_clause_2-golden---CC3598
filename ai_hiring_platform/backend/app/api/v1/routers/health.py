from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db
from app.schemas.response import ApiResponse
from app.core.logging import logger

router = APIRouter()

@router.get("", response_model=ApiResponse[dict])
def check_health(db: Session = Depends(get_db)):
    logger.info("Health check endpoint accessed.")
    try:
        # Verify database connectivity
        db.execute(text("SELECT 1"))
        logger.info("Database connectivity verified.")
        return ApiResponse[dict](
            success=True,
            message="Service is healthy",
            data={
                "status": "healthy",
                "database": "connected"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return ApiResponse[dict](
            success=False,
            message="Service is unhealthy",
            data={
                "status": "unhealthy",
                "database": f"error: {str(e)}"
            }
        )
