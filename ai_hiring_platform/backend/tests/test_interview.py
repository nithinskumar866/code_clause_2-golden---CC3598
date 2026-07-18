"""AI Recruiter interview simulation (deterministic path, no LLM key)."""
import os
import json
import pytest
from app.core.constants import REPORT_DIR
from app.services.ai import interview_service


def _write_report(analysis_id):
    report = {
        "analysis_id": analysis_id, "resume_id": 1, "jd_id": 1,
        "overall_score": 60, "coverage_score": 60, "experience_score": 60,
        "project_score": 50, "confidence_score": 55, "quality_score": 60,
        "summary": "s", "requirements": [
            {"requirement": "python", "category": "Programming Language", "status": "Matched",
             "matched_evidence": "Built backend services in Python.", "explanation": "x",
             "limitations": "None", "confidence": 88},
            {"requirement": "kubernetes", "category": "Cloud & DevOps", "status": "Missing",
             "matched_evidence": "", "explanation": "x", "limitations": "gap", "confidence": 0},
        ],
        "strengths": ["s"], "weaknesses": ["w"], "skill_relationships": [], "missing_skills": ["kubernetes"],
        "learning_roadmap": [], "recruiter_recommendation": "Conditional",
        "rejection_email": None,
        "interview_questions": [
            "They are strong in python. Ask about a hard python bug they solved.",
            "The candidate hasn't used kubernetes directly. Probe their container knowledge.",
        ],
        "retrieval_results": [
            {"requirement": "python", "matches": [
                {"chunk": "Built backend services in Python.", "section": "Experience", "score": 0.9, "confidence": 88}]},
            {"requirement": "kubernetes", "matches": []},
        ],
    }
    path = os.path.join(REPORT_DIR, f"analysis_{analysis_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return path


@pytest.fixture
def stored(tmp_path):
    aid = 778899
    path = _write_report(aid)
    yield aid
    if os.path.exists(path):
        os.remove(path)


def test_simulate_interview_deterministic(stored):
    sim = interview_service.simulate_interview(stored)
    assert sim.analysis_id == stored
    assert sim.generated_by == "deterministic"
    assert len(sim.items) == 2
    for qa in sim.items:
        assert qa.question
        assert qa.ideal_answer
        assert 0 <= qa.confidence <= 100
        assert qa.recruiter_evaluation
    # The Matched skill should yield a confident, evidence-backed answer.
    py = next(i for i in sim.items if "python" in i.question.lower())
    assert py.confidence >= 75
    assert "Python" in py.evidence
    # The Missing skill should be flagged and get a low-confidence follow-up.
    k8s = next(i for i in sim.items if "kubernetes" in i.question.lower())
    assert k8s.confidence < 60
    assert len(k8s.follow_up_questions) >= 1


def test_simulate_interview_missing_analysis():
    from app.core.exceptions import NotFoundError
    with pytest.raises(NotFoundError):
        interview_service.simulate_interview(424242)


def test_interview_endpoint(client, stored):
    r = client.post(f"/api/v1/analysis/{stored}/interview")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["data"]["analysis_id"] == stored
    assert len(body["data"]["items"]) == 2
