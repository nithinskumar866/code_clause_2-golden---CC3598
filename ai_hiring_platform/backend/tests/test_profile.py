"""
Tests for deterministic candidate profile + seniority extraction and its propagation
into the hiring report.
"""
from app.services.ai.profile_service import extract_profile
from app.services.ai.evaluation_service import run_mock_evaluation

_RESUME = """John Smith
Senior Software Engineer
john@example.com
Summary
Backend engineer with 7 years of experience building APIs.
Experience
Acme Corp 2018 - 2025
Led a team building Python services.
"""


def test_extract_name_title_years_and_seniority():
    p = extract_profile(_RESUME, "Requirements:\n- 5+ years of experience\n")
    assert p.name == "John Smith"
    assert "Engineer" in (p.title or "")
    assert p.total_years == 7.0            # explicit "7 years" claim wins
    assert p.seniority_level == "Senior"   # 5 <= 7 < 8
    assert p.required_years == 5.0
    assert p.seniority_fit == "Exceeds"    # 7 >= 5 + 2


def test_seniority_fit_below_and_meets():
    below = extract_profile("Jane\nDeveloper\n3 years of experience.\n", "5+ years required")
    assert below.total_years == 3.0
    assert below.seniority_fit == "Below"

    meets = extract_profile("Sam\nDeveloper\n5 years of experience.\n", "5+ years required")
    assert meets.seniority_fit == "Meets"


def test_unknown_when_years_absent():
    p = extract_profile("Alex Doe\nProduct Designer\n", "Design role, no years stated")
    assert p.name == "Alex Doe"
    assert p.total_years is None
    assert p.seniority_level is None
    assert p.seniority_fit == "Unknown"


def test_report_carries_candidate_profile():
    evidence = {
        "analysis_id": 1,
        "candidate_profile": {"name": "X Y", "title": "Engineer", "total_years": 6,
                              "seniority_level": "Senior", "required_years": 5,
                              "seniority_fit": "Meets", "explanation": "ok"},
        "retrieval_results": [
            {"requirement": "python", "importance": "must", "weight": 1.0,
             "matches": [{"chunk": "Built APIs in Python within Experience.",
                          "section": "Experience", "score": 0.9, "confidence": 90}]},
        ],
    }
    report = run_mock_evaluation(evidence)
    assert report.candidate_profile is not None
    assert report.candidate_profile.name == "X Y"
    assert report.candidate_profile.seniority_fit == "Meets"


def test_report_profile_is_none_when_absent():
    report = run_mock_evaluation({"analysis_id": 1,
                                  "retrieval_results": [{"requirement": "python", "matches": []}]})
    assert report.candidate_profile is None
