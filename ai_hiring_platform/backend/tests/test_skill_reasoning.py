"""Transfer / equivalence reasoning in the deterministic engine.

Verifies the fixes for real recruiter complaints: an equivalent skill (SQL) must
satisfy a requirement (MySQL) instead of reading 'Missing', and an adjacent skill
(Docker) must be noted as transferable toward a gap (Kubernetes) without falsely
claiming the skill. Uses real embeddings (SQL~MySQL ~0.84, Docker~Kubernetes ~0.70)."""
from app.services.ai.evaluation_service import run_mock_evaluation


def _fit(report, name):
    return next(r for r in report.requirements if r.requirement == name)


def _evidence():
    # The candidate lists SQL and Docker (in a hands-on Experience bullet) plus Python.
    return {
        "analysis_id": 1,
        "retrieval_results": [
            {"requirement": "python", "importance": "must", "weight": 1.0, "matches": [
                {"chunk": "Built backend services in Python; used SQL and Docker in production.",
                 "section": "Experience", "score": 0.9, "confidence": 88, "page": 1}]},
            {"requirement": "mysql", "importance": "must", "weight": 1.0, "matches": []},
            {"requirement": "kubernetes", "importance": "must", "weight": 1.0, "matches": []},
            {"requirement": "cobol", "importance": "nice", "weight": 0.4, "matches": []},
        ],
    }


def test_equivalent_skill_rescues_missing_requirement():
    report = run_mock_evaluation(_evidence())
    mysql = _fit(report, "mysql")
    # SQL ≈ MySQL → the requirement must NOT be Missing.
    assert mysql.status in ("Matched", "Partial")
    assert mysql.status != "Missing"
    assert "mysql" not in report.missing_skills
    assert "sql" in mysql.explanation.lower()


def test_adjacent_skill_noted_as_transferable_but_not_claimed():
    report = run_mock_evaluation(_evidence())
    k8s = _fit(report, "kubernetes")
    # Docker is related but NOT equivalent → still Missing, but flagged as a short ramp-up.
    assert k8s.status == "Missing"
    assert "docker" in k8s.explanation.lower()
    assert "kubernetes" in report.missing_skills
    # A transfer-aware interview question should exist for the transferable gap.
    assert any("kubernetes" in q.lower() and "docker" in q.lower() for q in report.interview_questions)


def test_genuine_gap_stays_missing_with_no_false_transfer():
    report = run_mock_evaluation(_evidence())
    cobol = _fit(report, "cobol")
    assert cobol.status == "Missing"
    assert "cobol" in report.missing_skills
    # It is flagged as a genuine gap, not a false "you can transfer into it" claim.
    assert "genuine gap" in cobol.limitations.lower()
    assert "ramp-up" not in cobol.explanation.lower()


def test_plain_language_no_keyword_jargon():
    report = run_mock_evaluation(_evidence())
    text = " ".join(r.explanation + " " + r.limitations for r in report.requirements).lower()
    # The old jargon phrasing must be gone.
    assert "keyword list" not in text
    assert "measurable project detail" not in text
