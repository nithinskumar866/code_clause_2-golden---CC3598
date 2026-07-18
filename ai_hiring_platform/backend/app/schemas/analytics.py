from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class OverallStatistics(BaseModel):
    total_analyses: int
    selected: int
    borderline: int
    rejected: int
    average_overall_score: float
    average_experience_score: float
    average_project_score: float
    average_quality_score: float
    average_coverage_score: float


class ScoreBucket(BaseModel):
    label: str
    min: int
    max: int
    count: int


class ScoreDistribution(BaseModel):
    total_analyses: int
    ranges: List[ScoreBucket]


class RecommendationBucket(BaseModel):
    label: str = Field(..., description="Selected / Borderline / Rejected")
    count: int
    percentage: float = Field(..., description="Share of completed analyses, 0-100")


class RecommendationDistribution(BaseModel):
    total_analyses: int
    distribution: List[RecommendationBucket]


class TrendPoint(BaseModel):
    period: str = Field(..., description="Period key, e.g. 2026-07-17, 2026-W29, 2026-07")
    count: int


class TrendData(BaseModel):
    daily: List[TrendPoint]
    weekly: List[TrendPoint]
    monthly: List[TrendPoint]


class TopItem(BaseModel):
    name: str = Field(..., description="Filename")
    count: int


class RecentAnalysisItem(BaseModel):
    analysis_id: int
    timestamp: datetime
    resume_filename: str
    jd_filename: str
    overall_score: Optional[int] = None
    recommendation: str
    workflow_status: str


class SkillCount(BaseModel):
    skill: str
    count: int


class SkillFrequency(BaseModel):
    """Aggregated skill signal across all stored reports: which requirements are
    most often matched, and which skills are most often missing (the hiring gap)."""
    top_matched: List[SkillCount] = Field(default_factory=list, description="Most frequently matched requirements")
    top_missing: List[SkillCount] = Field(default_factory=list, description="Most frequently missing skills")
