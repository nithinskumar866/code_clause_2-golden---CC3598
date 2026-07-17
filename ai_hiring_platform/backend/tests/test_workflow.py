import os
import json

import pytest
from sqlalchemy import create_engine, text

from app.core.constants import REPORT_DIR
from app.core.database import ensure_schema
from app.models.database import Resume, JobDescription, Analysis


def _seed_analysis(db):
    resume = Resume(filename="r.pdf", status="Indexed")
    jd = JobDescription(filename="j.docx", status="Indexed")
    db.add_all([resume, jd])
    db.commit()
    db.refresh(resume)
    db.refresh(jd)

    analysis = Analysis(resume_id=resume.id, jd_id=jd.id, status="Analysed")
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis.id


def _write_report_file(analysis_id):
    report = {
        "analysis_id": analysis_id,
        "resume_id": 1,
        "jd_id": 1,
        "retrieval_results": [],
        "overall_score": 82,
        "coverage_score": 80,
        "experience_score": 70,
        "project_score": 75,
        "confidence_score": 78,
        "quality_score": 85,
        "summary": "s",
        "requirements": [],
        "strengths": ["s"],
        "weaknesses": ["w"],
        "skill_relationships": [],
        "missing_skills": [],
        "learning_roadmap": [],
        "interview_questions": ["q"],
        "recruiter_recommendation": "Recommendation",
        "rejection_email": None,
    }
    path = os.path.join(REPORT_DIR, f"analysis_{analysis_id}.json")
    with open(path, "w") as f:
        json.dump(report, f)
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


# ------------------------------- GET status -------------------------------

def test_get_status_defaults_to_applied(client, db_session):
    aid = _seed_analysis(db_session)
    response = client.get(f"/api/v1/analysis/{aid}/status")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["analysis_id"] == aid
    assert data["workflow_status"] == "Applied"


def test_get_status_not_found(client, db_session):
    response = client.get("/api/v1/analysis/999999/status")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


# ------------------------------ PATCH status ------------------------------

def test_patch_status_updates_and_persists(client, db_session):
    aid = _seed_analysis(db_session)
    response = client.patch(f"/api/v1/analysis/{aid}/status", json={"workflow_status": "Technical"})
    assert response.status_code == 200
    assert response.json()["data"]["workflow_status"] == "Technical"

    # Persisted across a fresh read
    again = client.get(f"/api/v1/analysis/{aid}/status")
    assert again.json()["data"]["workflow_status"] == "Technical"


def test_patch_all_valid_statuses(client, db_session):
    aid = _seed_analysis(db_session)
    for status in ["Applied", "Screening", "Technical", "Manager", "HR", "Offer", "Joined", "Rejected"]:
        response = client.patch(f"/api/v1/analysis/{aid}/status", json={"workflow_status": status})
        assert response.status_code == 200
        assert response.json()["data"]["workflow_status"] == status


def test_patch_invalid_status_rejected(client, db_session):
    aid = _seed_analysis(db_session)
    response = client.patch(f"/api/v1/analysis/{aid}/status", json={"workflow_status": "Interviewing"})
    assert response.status_code == 422


def test_patch_status_not_found(client, db_session):
    response = client.patch("/api/v1/analysis/999999/status", json={"workflow_status": "Offer"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


# ------------------------- History preserves status ------------------------

def test_history_preserves_status(client, db_session, clean_reports):
    aid = _seed_analysis(db_session)
    _write_report_file(aid)  # makes it a completed, history-visible analysis
    client.patch(f"/api/v1/analysis/{aid}/status", json={"workflow_status": "Offer"})

    detail = client.get(f"/api/v1/analysis/history/{aid}")
    assert detail.status_code == 200
    assert detail.json()["data"]["workflow_status"] == "Offer"

    listing = client.get("/api/v1/analysis/history").json()["data"]
    item = next(i for i in listing if i["analysis_id"] == aid)
    assert item["workflow_status"] == "Offer"


# --------------------------- Additive migration ----------------------------

def test_ensure_schema_backfills_missing_column():
    """A legacy analyses table without workflow_status gets the column added."""
    legacy_engine = create_engine("sqlite://")
    with legacy_engine.connect() as conn:
        conn.execute(text("CREATE TABLE analyses (id INTEGER PRIMARY KEY, status VARCHAR)"))
        conn.execute(text("INSERT INTO analyses (id, status) VALUES (1, 'Analysed')"))
        conn.commit()

    ensure_schema(bind=legacy_engine)

    with legacy_engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(analyses)"))}
        assert "workflow_status" in cols
        # Existing row is backfilled with the default, not dropped.
        value = conn.execute(text("SELECT workflow_status FROM analyses WHERE id = 1")).scalar()
        assert value == "Applied"
    legacy_engine.dispose()
