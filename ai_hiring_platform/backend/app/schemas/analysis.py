from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class AnalysisBase(BaseModel):
    resume_id: int
    jd_id: int

class AnalysisCreate(AnalysisBase):
    pass

class AnalysisResponse(AnalysisBase):
    id: int
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# --- Sprint 5 Refined Hiring Report Schemas ---

class RequirementFit(BaseModel):
    requirement: str = Field(..., description="Name of the skill or requirement")
    category: str = Field(..., description="Inferred skill category, e.g. Programming Language, Framework, DevOps")
    status: str = Field(..., description="Matched, Partial, or Missing")
    matched_evidence: str = Field(..., description="Exact supporting text found in candidate resume, or empty if missing")
    explanation: str = Field(..., description="Short explanation of why this evidence matters")
    limitations: str = Field(..., description="What competency was missing, weak, or undocumented for this requirement")
    confidence: int = Field(..., description="Calculated confidence level between 0 and 100 for this claim")

class LearningRoadmapItem(BaseModel):
    skill: str = Field(..., description="Name of the missing skill")
    estimated_time: str = Field(..., description="Realistic dynamic upskilling learning time, e.g. '5-7 days'")
    reason: str = Field(..., description="Transferable skill and dependency reasoning context")

class HiringReport(BaseModel):
    # Algorithmic Scores
    overall_score: int = Field(..., description="Weighted average compatibility score between 0 and 100")
    coverage_score: int = Field(..., description="Requirement Coverage score between 0 and 100")
    experience_score: int = Field(..., description="Experience Alignment score between 0 and 100")
    project_score: int = Field(..., description="Project Relevance score between 0 and 100")
    confidence_score: int = Field(..., description="Evidence Confidence score between 0 and 100")
    quality_score: int = Field(..., description="Resume Quality score between 0 and 100")
    
    summary: str = Field(..., description="Recruiter executive summary of candidate fit")
    requirements: List[RequirementFit] = Field(..., description="Refined requirement-by-requirement assessments")
    
    # Sprint 5 Additions
    strengths: List[str] = Field(..., description="Core candidate strengths derived from high-confidence matches")
    weaknesses: List[str] = Field(..., description="Candidate technical gaps or weak evidence signals")
    skill_relationships: List[str] = Field(..., description="Semantic transferable dependencies between candidate skills and JD requirements")
    
    missing_skills: List[str] = Field(..., description="List of missing skills from the job description")
    learning_roadmap: List[LearningRoadmapItem] = Field(..., description="Upskilling timelines and prerequisite roadmaps")
    interview_questions: List[str] = Field(..., description="Behavioral or technical depth validation questions")
    
    recruiter_recommendation: str = Field(..., description="Overall hiring process recommendation, e.g., Proceed to Interview")
    rejection_email: Optional[str] = Field(None, description="Polite dynamic rejection email draft if compatibility is low")
