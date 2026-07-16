from typing import List, Dict, Any

class CandidateIntelligenceAgentInterface:
    """
    Abstract interface for Candidate Intelligence Agent.
    Responsible for ingesting candidate documents (resumes), parsing, embedding,
    structuring knowledge chunks, and retrieving evidence matching Job Description requirements.
    """
    def ingest_candidate_resume(self, resume_path: str, resume_id: int) -> None:
        raise NotImplementedError("ingest_candidate_resume() is not implemented.")

    def retrieve_evidence(self, resume_id: int, jd_path: str, jd_id: int) -> Dict[str, Any]:
        raise NotImplementedError("retrieve_evidence() is not implemented.")


class HiringDecisionAgentInterface:
    """
    Abstract interface for Hiring Decision Agent.
    Responsible for reasoning over candidate evidence and generating structured fits,
    interview recommendations, and recruiter communications.
    """
    def evaluate_candidate(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("evaluate_candidate() is not implemented.")

    def generate_interview_questions(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("generate_interview_questions() is not implemented.")
