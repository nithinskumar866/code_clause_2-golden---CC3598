import pytest
from app.agents.hiring_decision.agent import HiringDecisionAgent
from app.schemas.analysis import HiringReport

@pytest.fixture
def mock_evidence_data():
    return {
        "analysis_id": 101,
        "candidate_id": 42,
        "resume_id": 42,
        "jd_id": 7,
        "retrieval_results": [
            {
                "requirement": "python",
                "matches": [
                    {
                        "chunk": "Developed backend APIs using Python and FastAPI.",
                        "section": "Experience",
                        "score": 0.92,
                        "page": 1,
                        "filename": "john_doe.pdf",
                        "chunk_id": 1
                    }
                ]
            },
            {
                "requirement": "tensorflow",
                "matches": [
                    {
                        "chunk": "Listed TensorFlow in technical skills list.",
                        "section": "Skills",
                        "score": 0.81,
                        "page": 2,
                        "filename": "john_doe.pdf",
                        "chunk_id": 5
                    }
                ]
            },
            {
                "requirement": "docker",
                "matches": []  # Missing skill
            }
        ]
    }

def test_hiring_decision_agent_fallback_evaluation(mock_evidence_data):
    agent = HiringDecisionAgent()
    
    # We execute evaluate_candidate. Since settings.OPENAI_API_KEY is empty,
    # it will automatically trigger the robust mock engine.
    report = agent.evaluate_candidate(mock_evidence_data)
    
    assert report["analysis_id"] == 101
    assert report["candidate_id"] == 42
    assert report["resume_id"] == 42
    assert report["jd_id"] == 7
    
    # Check enriched fit details and explainable sub-scores
    assert "overall_score" in report
    assert "coverage_score" in report
    assert "experience_score" in report
    assert "project_score" in report
    assert "confidence_score" in report
    assert "quality_score" in report
    
    assert 0 <= report["overall_score"] <= 100
    assert 0 <= report["coverage_score"] <= 100
    assert 0 <= report["experience_score"] <= 100
    assert 0 <= report["project_score"] <= 100
    assert 0 <= report["confidence_score"] <= 100
    assert 0 <= report["quality_score"] <= 100
    
    assert "summary" in report
    assert "requirements" in report
    assert len(report["requirements"]) == 3
    
    # Assert Sprint 5 dynamic report additions
    assert "strengths" in report
    assert len(report["strengths"]) > 0
    assert "weaknesses" in report
    assert len(report["weaknesses"]) > 0
    assert "skill_relationships" in report
    assert "recruiter_recommendation" in report
    assert "technical" in report["recruiter_recommendation"].lower() or "recommend" in report["recruiter_recommendation"].lower()
    
    # Extract python requirements fit
    python_fit = [r for r in report["requirements"] if r["requirement"] == "python"][0]
    assert python_fit["status"] == "Matched"
    assert "Python" in python_fit["matched_evidence"]
    assert python_fit["category"] == "Programming Language"
    assert python_fit["confidence"] >= 75
    assert python_fit["limitations"] == "None"
    
    # Extract tensorflow requirements fit
    tf_fit = [r for r in report["requirements"] if r["requirement"] == "tensorflow"][0]
    assert tf_fit["status"] == "Partial"
    
    # Extract docker requirements fit
    docker_fit = [r for r in report["requirements"] if r["requirement"] == "docker"][0]
    assert docker_fit["status"] == "Missing"
    
    # Check missing skills list
    assert "docker" in report["missing_skills"]
    
    # Check learning roadmap
    assert len(report["learning_roadmap"]) > 0
    docker_roadmap = [r for r in report["learning_roadmap"] if r["skill"] == "docker"][0]
    assert "days" in docker_roadmap["estimated_time"]
    
    # Check questions list
    assert len(report["interview_questions"]) > 0
    
    assert report["overall_score"] < 75
    assert report["rejection_email"] is not None
    assert "rejection" in report["rejection_email"].lower() or "move forward" in report["rejection_email"].lower()
