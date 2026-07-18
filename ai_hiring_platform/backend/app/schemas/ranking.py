from typing import List, Optional
from pydantic import BaseModel, Field


class RankRequest(BaseModel):
    """Rank many resumes against a single job description."""
    jd_id: int = Field(..., description="Job description to rank against")
    resume_ids: List[int] = Field(..., min_length=1, description="Resumes to evaluate and rank")


class RankingEntry(BaseModel):
    """One candidate's position on the leaderboard, distilled from their full report."""
    rank: int = Field(..., description="1-based leaderboard position (best first)")
    analysis_id: Optional[int] = Field(None, description="Persisted analysis id, if evaluation succeeded")
    resume_id: int = Field(..., description="Source resume id")
    resume_filename: str = Field(..., description="Original resume filename")

    overall_score: int = Field(0, description="Weighted overall compatibility (0-100)")
    coverage_score: int = Field(0, description="Requirement coverage (0-100)")
    experience_score: int = Field(0, description="Experience alignment (0-100)")
    project_score: int = Field(0, description="Project relevance (0-100)")
    quality_score: int = Field(0, description="Resume quality (0-100)")
    confidence_score: int = Field(0, description="Evidence confidence (0-100)")

    recruiter_recommendation: str = Field("", description="Overall hiring recommendation")
    seniority_fit: Optional[str] = Field(None, description="Below | Meets | Exceeds | Unknown")
    credibility_score: Optional[int] = Field(None, description="Authenticity credibility (0-100)")
    keyword_stuffing_risk: Optional[str] = Field(None, description="Low | Medium | High")

    matched_count: int = Field(0, description="Requirements matched")
    partial_count: int = Field(0, description="Requirements partially matched")
    missing_count: int = Field(0, description="Requirements missing")
    top_missing_skills: List[str] = Field(default_factory=list, description="A few notable missing skills")

    error: Optional[str] = Field(None, description="Set if this resume failed to evaluate")


class RankingResponse(BaseModel):
    """A full leaderboard for one JD across the requested resumes."""
    jd_id: int
    jd_filename: str
    candidate_count: int = Field(..., description="Resumes requested")
    evaluated_count: int = Field(..., description="Resumes that evaluated successfully")
    top_candidate: Optional[RankingEntry] = Field(None, description="Highest-scoring successful candidate")
    entries: List[RankingEntry] = Field(default_factory=list, description="All candidates, best first")
