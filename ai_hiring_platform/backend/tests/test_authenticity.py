"""
Tests for the deterministic authenticity / keyword-stuffing detection service and
its integration into the hiring report. These are pure-algorithm tests over
evidence dicts — no embeddings or LLM required.
"""
import pytest

from app.services.ai import authenticity_service
from app.agents.hiring_decision.agent import HiringDecisionAgent

LONG_CHUNK = (
    "Built and shipped production REST APIs in Python with FastAPI, owning schema "
    "design, authentication, and CI/CD deployment across three microservices for "
    "two years, mentoring two engineers and cutting p95 latency by forty percent."
)  # >150 chars -> top detail-density band


def _evidence(retrieval_results):
    return {
        "analysis_id": 1,
        "candidate_id": 1,
        "resume_id": 1,
        "jd_id": 1,
        "retrieval_results": retrieval_results,
    }


def test_fully_corroborated_resume_is_credible():
    data = _evidence([
        {"requirement": "python", "matches": [
            {"chunk": LONG_CHUNK, "section": "Experience", "score": 0.9, "confidence": 90}]},
        {"requirement": "fastapi", "matches": [
            {"chunk": LONG_CHUNK, "section": "Projects", "score": 0.85, "confidence": 82}]},
    ])
    result = authenticity_service.assess_authenticity(data)

    assert result.assessment.over_claimed_skills == []
    assert result.assessment.corroboration_ratio == 1.0
    assert result.assessment.credibility_score == 100
    assert result.assessment.keyword_stuffing_risk == "Low"
    assert result.quality_score == 100  # 100*0.6 + 100(depth)*0.4


def test_skills_only_claims_flag_keyword_stuffing():
    data = _evidence([
        {"requirement": "python", "matches": [
            {"chunk": LONG_CHUNK, "section": "Experience", "score": 0.9, "confidence": 90}]},
        # Listed in the Skills section only — never demonstrated.
        {"requirement": "tensorflow", "matches": [
            {"chunk": "TensorFlow", "section": "Skills", "score": 0.8, "confidence": 40}]},
        # Absent entirely — Missing, NOT over-claimed.
        {"requirement": "docker", "matches": []},
    ])
    result = authenticity_service.assess_authenticity(data)
    a = result.assessment

    assert a.over_claimed_skills == ["tensorflow"]
    assert "docker" not in a.over_claimed_skills          # missing != over-claimed
    assert a.corroboration_ratio == 0.5                   # 1 of 2 claimed demonstrated
    assert a.credibility_score == 50
    assert a.keyword_stuffing_risk == "High"              # 0.5 over-claimed fraction
    assert "tensorflow" in a.explanation


def test_empty_evidence_is_low_risk_not_stuffing():
    data = _evidence([
        {"requirement": "python", "matches": []},
        {"requirement": "docker", "matches": []},
    ])
    result = authenticity_service.assess_authenticity(data)
    a = result.assessment

    assert a.over_claimed_skills == []
    assert a.corroboration_ratio == 0.0
    assert a.credibility_score == 0
    assert a.keyword_stuffing_risk == "Low"
    assert result.quality_score == 0


def test_summary_only_claim_is_not_demonstrated():
    # A skill mentioned only in the Summary blurb is not a demonstration.
    data = _evidence([
        {"requirement": "kubernetes", "matches": [
            {"chunk": "Experienced with Kubernetes.", "section": "Summary", "score": 0.7, "confidence": 35}]},
    ])
    a = authenticity_service.assess_authenticity(data).assessment
    assert a.over_claimed_skills == ["kubernetes"]
    assert a.keyword_stuffing_risk == "High"


def test_report_carries_authenticity_block():
    # End-to-end through Agent 2's mock path (no LLM key configured in tests).
    data = _evidence([
        {"requirement": "python", "matches": [
            {"chunk": LONG_CHUNK, "section": "Experience", "score": 0.9, "confidence": 90}]},
        {"requirement": "tensorflow", "matches": [
            {"chunk": "TensorFlow", "section": "Skills", "score": 0.8, "confidence": 40}]},
    ])
    report = HiringDecisionAgent().evaluate_candidate(data)

    assert "authenticity" in report
    auth = report["authenticity"]
    assert auth is not None
    assert auth["keyword_stuffing_risk"] in ("Low", "Medium", "High")
    assert 0 <= auth["credibility_score"] <= 100
    assert 0 <= report["quality_score"] <= 100
    assert report["quality_score"] != 85 or auth["credibility_score"] != 0  # no longer a hardcoded constant
