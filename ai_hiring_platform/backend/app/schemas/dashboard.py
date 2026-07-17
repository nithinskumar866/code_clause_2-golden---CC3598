from typing import List
from pydantic import BaseModel, Field

# --- Dashboard Analytics Schemas ---
# All values are aggregated from the existing Analysis History (completed
# evaluations only). No new persistence is introduced.


class DashboardOverview(BaseModel):
    total_analyses: int = Field(..., description="Number of completed analyses")
    selected_count: int = Field(..., description="Analyses with overall_score >= selected threshold")
    borderline_count: int = Field(..., description="Analyses between rejected and selected thresholds")
    rejected_count: int = Field(..., description="Analyses with overall_score below the borderline threshold")
    average_overall_score: float = Field(..., description="Mean overall compatibility score")
    average_skill_score: float = Field(..., description="Mean requirement-coverage (skill) score")
    average_experience_score: float = Field(..., description="Mean experience alignment score")
    average_project_score: float = Field(..., description="Mean project relevance score")
    average_quality_score: float = Field(..., description="Mean resume quality score")


class TrendPoint(BaseModel):
    date: str = Field(..., description="Calendar day (UTC) in YYYY-MM-DD form")
    count: int = Field(..., description="Number of analyses created that day")


class DashboardTrends(BaseModel):
    period_days: int = Field(..., description="Length of the trend window in days")
    trends: List[TrendPoint] = Field(..., description="Per-day analysis counts, oldest first, zero-filled")


class ScoreRange(BaseModel):
    label: str = Field(..., description="Human-readable band label, e.g. '81-100'")
    min: int = Field(..., description="Inclusive lower bound of the band")
    max: int = Field(..., description="Inclusive upper bound of the band")
    count: int = Field(..., description="Number of analyses whose overall_score falls in this band")


class DashboardDistribution(BaseModel):
    total_analyses: int = Field(..., description="Number of completed analyses distributed")
    ranges: List[ScoreRange] = Field(..., description="Overall-score distribution across fixed bands")
