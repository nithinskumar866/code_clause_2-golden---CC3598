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
    importance: Optional[str] = Field(None, description="'must' or 'nice' — whether the JD lists this as required or nice-to-have")
    weight: Optional[float] = Field(None, description="Scoring weight derived from importance (must-haves weigh more)")

class LearningRoadmapItem(BaseModel):
    skill: str = Field(..., description="Name of the missing skill")
    estimated_time: str = Field(..., description="Realistic dynamic upskilling learning time, e.g. '5-7 days'")
    reason: str = Field(..., description="Transferable skill and dependency reasoning context")

class CandidateProfile(BaseModel):
    """Lightweight candidate identity + seniority, derived deterministically from the
    resume text and compared against the JD's stated experience requirement. Answers
    'who is this and are they senior enough for the role' — which pure skill-fit misses."""
    name: Optional[str] = Field(None, description="Candidate name, if detectable")
    title: Optional[str] = Field(None, description="Most recent / headline role title")
    total_years: Optional[float] = Field(None, description="Estimated total years of experience")
    seniority_level: Optional[str] = Field(None, description="Junior | Mid | Senior | Lead (from total_years)")
    required_years: Optional[float] = Field(None, description="Years the JD asks for, if stated")
    seniority_fit: Optional[str] = Field(None, description="Below | Meets | Exceeds | Unknown vs the JD requirement")
    explanation: str = Field("", description="Recruiter-readable seniority rationale")

class AuthenticityAssessment(BaseModel):
    """Deterministic credibility signal: how well claimed skills are corroborated by
    concrete Experience/Project evidence, and whether the resume shows keyword stuffing
    (skills listed but never demonstrated). Computed algorithmically from retrieved
    evidence sections — never delegated to the LLM."""
    credibility_score: int = Field(..., description="Corroboration expressed 0-100: share of claimed skills demonstrated in narrative sections")
    keyword_stuffing_risk: str = Field(..., description="Low, Medium, or High — based on the fraction of claimed skills that are listed but not demonstrated")
    over_claimed_skills: List[str] = Field(..., description="Skills the candidate lists but never demonstrates in Experience/Projects")
    corroboration_ratio: float = Field(..., description="Demonstrated claimed skills / total claimed skills (0.0-1.0)")
    explanation: str = Field(..., description="Recruiter-readable rationale for the credibility verdict")

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

    # Deterministic authenticity signal (keyword-stuffing / over-claim detection).
    # Optional so historical reports persisted before this field validate as None.
    authenticity: Optional[AuthenticityAssessment] = Field(None, description="Credibility assessment: corroboration of claims + keyword-stuffing risk")
    # Candidate identity + seniority fit (deterministic). Optional/nullable for backward-compat.
    candidate_profile: Optional[CandidateProfile] = Field(None, description="Candidate name/title/seniority and fit vs the JD's stated experience requirement")

# --- Analysis History Schemas ---
# History is a read/manage view over already-persisted evaluations (the Analysis
# row + storage/reports/analysis_<id>.json). No new storage is introduced.

class AnalysisHistoryItem(BaseModel):
    """Compact summary row for the history list (newest first)."""
    analysis_id: int = Field(..., description="ID of the completed analysis")
    timestamp: datetime = Field(..., description="When the analysis was created")
    resume_id: int = Field(..., description="Source resume ID")
    job_description_id: int = Field(..., description="Source job description ID")
    resume_filename: str = Field(..., description="Original resume filename")
    jd_filename: str = Field(..., description="Original job description filename")
    workflow_status: str = Field("Applied", description="Candidate hiring workflow status")
    overall_score: int = Field(..., description="Weighted overall compatibility score")
    coverage_score: int = Field(..., description="Requirement coverage score")
    experience_score: int = Field(..., description="Experience alignment score")
    project_score: int = Field(..., description="Project relevance score")
    quality_score: int = Field(..., description="Resume quality score")
    recruiter_recommendation: str = Field(..., description="Overall hiring recommendation")
    summary: str = Field(..., description="Recruiter executive summary")

class AnalysisHistoryDetail(AnalysisHistoryItem):
    """A single history entry plus its full stored hiring report."""
    report: HiringReport = Field(..., description="The full persisted hiring report")
