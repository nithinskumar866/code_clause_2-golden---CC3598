from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.response import ApiResponse
from app.schemas.dashboard import DashboardOverview, DashboardTrends, DashboardDistribution
from app.services import dashboard as dashboard_service

router = APIRouter()


@router.get("/overview", response_model=ApiResponse[DashboardOverview])
def get_dashboard_overview(db: Session = Depends(get_db)):
    """Aggregate counts and average sub-scores across all completed analyses."""
    logger.info("Dashboard overview endpoint accessed.")
    data = dashboard_service.get_overview(db)
    return ApiResponse[DashboardOverview](
        success=True,
        message="Dashboard overview retrieved.",
        data=data,
    )


@router.get("/trends", response_model=ApiResponse[DashboardTrends])
def get_dashboard_trends(db: Session = Depends(get_db)):
    """Per-day analysis counts over the last 30 days."""
    logger.info("Dashboard trends endpoint accessed.")
    data = dashboard_service.get_trends(db)
    return ApiResponse[DashboardTrends](
        success=True,
        message="Dashboard trends retrieved.",
        data=data,
    )


@router.get("/distribution", response_model=ApiResponse[DashboardDistribution])
def get_dashboard_distribution(db: Session = Depends(get_db)):
    """Overall-score distribution across fixed score bands."""
    logger.info("Dashboard distribution endpoint accessed.")
    data = dashboard_service.get_distribution(db)
    return ApiResponse[DashboardDistribution](
        success=True,
        message="Dashboard distribution retrieved.",
        data=data,
    )
