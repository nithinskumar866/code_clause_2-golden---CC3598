import os
import json

import fitz
import pytest

from app.core.constants import REPORT_DIR
from app.services import report_export
from app.core.exceptions import NotFoundError, CorruptedReportError


def _valid_report(analysis_id, overall=82):
    return {
        "analysis_id": analysis_id,
        "resume_id": 1,
        "jd_id": 1,
        "retrieval_results": [],
        "overall_score": overall,
        "coverage_score": 80,
        "experience_score": 70,
        "project_score": 75,
        "confidence_score": 78,
        "quality_score": 85,
        "summary": "This candidate is a strong match for the backend role.",
        "requirements": [
            {"requirement": "python", "category": "Programming Language", "status": "Matched",
             "matched_evidence": "Built APIs in Python", "explanation": "Direct match",
             "limitations": "None", "confidence": 90},
        ],
        "strengths": ["Strong Python backend experience"],
        "weaknesses": ["Limited Kubernetes exposure"],
        "skill_relationships": [],
        "missing_skills": ["Kubernetes"],
        "learning_roadmap": [{"skill": "Kubernetes", "estimated_time": "10-14 days", "reason": "Container orchestration"}],
        "interview_questions": ["Describe a system you scaled."],
        "recruiter_recommendation": "Highly Recommended - Proceed to Technical Screen",
        "rejection_email": None,
    }


def _write(analysis_id, content):
    path = os.path.join(REPORT_DIR, f"analysis_{analysis_id}.json")
    with open(path, "w") as f:
        if isinstance(content, str):
            f.write(content)
        else:
            json.dump(content, f)
    return path


@pytest.fixture
def clean_reports():
    before = set(os.listdir(REPORT_DIR)) if os.path.exists(REPORT_DIR) else set()
    yield
    after = set(os.listdir(REPORT_DIR)) if os.path.exists(REPORT_DIR) else set()
    for name in after - before:
        try:
            os.remove(os.path.join(REPORT_DIR, name))
        except OSError:
            pass


# -------------------------------- Unit tests --------------------------------

def test_service_export_json_reads_stored_report(clean_reports):
    _write(7001, _valid_report(7001, overall=91))
    payload = report_export.export_json(7001)
    assert isinstance(payload, bytes)
    parsed = json.loads(payload)
    assert parsed["overall_score"] == 91
    assert parsed["analysis_id"] == 7001


def test_service_missing_raises_not_found(clean_reports):
    with pytest.raises(NotFoundError):
        report_export.export_json(999999)


def test_service_corrupted_unparseable(clean_reports):
    _write(7002, "this is not valid json {")
    with pytest.raises(CorruptedReportError):
        report_export.export_json(7002)


def test_service_corrupted_malformed(clean_reports):
    _write(7003, {"unexpected": "shape"})  # valid JSON, not a HiringReport
    with pytest.raises(CorruptedReportError):
        report_export.export_json(7003)


# ----------------------------- PDF generation -------------------------------

def test_service_export_pdf_is_valid_and_contains_content(clean_reports):
    _write(7004, _valid_report(7004))
    pdf = report_export.export_pdf(7004)
    assert isinstance(pdf, bytes)
    assert pdf[:4] == b"%PDF"

    # Reopen the generated PDF and confirm the report content rendered.
    doc = fitz.open(stream=pdf, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    assert "Hiring Analysis Report" in text
    assert "Highly Recommended" in text
    assert "Kubernetes" in text


# ------------------------------ Endpoint tests ------------------------------

def test_endpoint_export_json(client, clean_reports):
    _write(7005, _valid_report(7005, overall=88))
    response = client.get("/api/v1/analysis/7005/export/json")
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    assert 'attachment; filename="report_analysis_7005.json"' in response.headers["content-disposition"]
    assert response.headers["cache-control"] == "private, max-age=3600"
    assert response.headers.get("etag")
    assert response.json()["overall_score"] == 88


def test_endpoint_export_pdf(client, clean_reports):
    _write(7006, _valid_report(7006))
    response = client.get("/api/v1/analysis/7006/export/pdf")
    assert response.status_code == 200
    assert "application/pdf" in response.headers["content-type"]
    assert 'filename="report_analysis_7006.pdf"' in response.headers["content-disposition"]
    assert response.headers["cache-control"] == "private, max-age=3600"
    assert response.content[:4] == b"%PDF"


def test_endpoint_missing_report_404(client, clean_reports):
    for path in ("/api/v1/analysis/424242/export/json", "/api/v1/analysis/424242/export/pdf"):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"


def test_endpoint_corrupted_report_422(client, clean_reports):
    _write(7007, {"unexpected": "shape"})
    response = client.get("/api/v1/analysis/7007/export/json")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CORRUPTED_REPORT"


def test_endpoint_invalid_id_and_format(client, clean_reports):
    # Non-integer id -> FastAPI path validation 422
    assert client.get("/api/v1/analysis/abc/export/json").status_code == 422
    # Unsupported format -> no matching route -> 404
    _write(7008, _valid_report(7008))
    assert client.get("/api/v1/analysis/7008/export/xml").status_code == 404
