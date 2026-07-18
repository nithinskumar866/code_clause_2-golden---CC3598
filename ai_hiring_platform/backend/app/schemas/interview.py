from typing import List
from pydantic import BaseModel, Field


class InterviewQA(BaseModel):
    """One simulated interview exchange: the AI plays a senior interviewer who both
    asks the question and answers it *as the candidate would*, using ONLY the
    candidate's own resume evidence — so the recruiter sees whether the resume can
    actually justify the skill without interviewing blind."""
    question: str = Field(..., description="Interview question")
    ideal_answer: str = Field(..., description="The strongest answer the resume can support (grounded in evidence, never invented)")
    evidence: str = Field("", description="Resume text that backs the answer")
    confidence: int = Field(..., description="0-100: how well the resume supports a solid answer")
    missing_information: str = Field("", description="What the resume does not show to fully answer this")
    follow_up_questions: List[str] = Field(default_factory=list, description="Probes to run when confidence is low")
    recruiter_evaluation: str = Field(..., description="A recruiter's short verdict on how convincing the answer is")


class InterviewSimulation(BaseModel):
    """A full AI-recruiter pass over one analysis."""
    analysis_id: int
    generated_by: str = Field(..., description="'llm' when a model reasoned, else 'deterministic'")
    items: List[InterviewQA] = Field(default_factory=list)
