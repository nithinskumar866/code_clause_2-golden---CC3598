"""
Tests for requirement prioritization (must-have vs nice-to-have) and the resulting
importance-weighted coverage scoring.
"""
from app.services.ai.jd_requirement_extractor import classify_priorities
from app.services.ai.evaluation_service import run_mock_evaluation

_MATCH = [{"chunk": "Built production services using this in Experience.",
           "section": "Experience", "score": 0.9, "confidence": 90}]


def _evidence(items):
    return {"analysis_id": 1, "retrieval_results": items}


def test_classify_priorities_reads_jd_wording():
    jd = (
        "Requirements:\n"
        "- Python and Docker\n"
        "Nice to have:\n"
        "- Kubernetes\n"
    )
    p = classify_priorities(jd, ["python", "docker", "kubernetes"])
    assert p["python"]["importance"] == "must"
    assert p["docker"]["importance"] == "must"
    assert p["kubernetes"]["importance"] == "nice"
    assert p["kubernetes"]["weight"] < p["python"]["weight"]


def test_inline_nice_cue_overrides_context():
    jd = "Requirements:\n- Experience with Redis is a plus\n"
    p = classify_priorities(jd, ["redis"])
    assert p["redis"]["importance"] == "nice"


def test_unmentioned_requirement_defaults_to_must():
    # Derived requirements (e.g. seniority) may not appear verbatim -> default must.
    p = classify_priorities("Some job description text.", ["5+ years experience"])
    assert p["5+ years experience"]["importance"] == "must"


def test_missing_must_have_costs_more_than_missing_nice():
    matched_must_missing_nice = run_mock_evaluation(_evidence([
        {"requirement": "python", "matches": _MATCH, "importance": "must", "weight": 1.0},
        {"requirement": "kubernetes", "matches": [], "importance": "nice", "weight": 0.4},
    ]))
    matched_nice_missing_must = run_mock_evaluation(_evidence([
        {"requirement": "python", "matches": _MATCH, "importance": "nice", "weight": 0.4},
        {"requirement": "kubernetes", "matches": [], "importance": "must", "weight": 1.0},
    ]))
    # Same 1-of-2 raw coverage, but weighting makes the must-have miss score far lower.
    assert matched_must_missing_nice.coverage_score > matched_nice_missing_must.coverage_score


def test_requirement_fit_carries_importance_and_missing_must_flagged():
    report = run_mock_evaluation(_evidence([
        {"requirement": "python", "matches": _MATCH, "importance": "must", "weight": 1.0},
        {"requirement": "kafka", "matches": [], "importance": "must", "weight": 1.0},
    ]))
    py = [r for r in report.requirements if r.requirement == "python"][0]
    assert py.importance == "must"
    assert py.weight == 1.0
    # Missing must-have surfaces prominently in recruiter weaknesses.
    assert any("must-have" in w.lower() for w in report.weaknesses)


def test_coverage_backward_compatible_without_priority():
    # Evidence with no importance/weight must score exactly as equal-weighted before.
    report = run_mock_evaluation(_evidence([
        {"requirement": "python", "matches": _MATCH},
        {"requirement": "docker", "matches": []},
    ]))
    assert report.coverage_score == 50  # 1 of 2, equal weights
