import os
import json

import pytest

from app.core.constants import REPORT_DIR
from app.models.database import Resume, JobDescription, Analysis


def _write_report_file(analysis_id: int, resume_id: int, jd_id: int, overall: int = 82):
    """Persist a report JSON exactly as the Hiring Decision Agent does on disk."""
    report = {
        "analysis_id": analysis_id,
        "candidate_id": resume_id,
        "resume_id": resume_id,
        "jd_id": jd_id,
        "retrieval_results": [],
        "overall_score": overall,
        "coverage_score": 80,
        "experience_score": 70,
        "project_score": 75,
        "confidence_score": 78,
        "quality_score": 85,
        "summary": f"Executive summary for analysis {analysis_id}.",
        "requirements": [],
        "strengths": ["Strong Python backend experience"],
        "weaknesses": ["Limited container orchestration exposure"],
        "skill_relationships": [],
        "missing_skills": [],
        "learning_roadmap": [],
        "interview_questions": ["Describe a complex system you designed."],
        "recruiter_recommendation": "Highly Recommended - Proceed to Technical Screen",
        "rejection_email": None,
    }
    path = os.path.join(REPORT_DIR, f"analysis_{analysis_id}.json")
    with open(path, "w") as f:
        json.dump(report, f)
    return path


def _seed_completed_analysis(db, resume_name, jd_name, overall=82):
    """Recreate what upload + a completed evaluation leave persisted."""
    resume = Resume(filename=resume_name, status="Indexed")
    jd = JobDescription(filename=jd_name, status="Indexed")
    db.add(resume)
    db.add(jd)
    db.commit()
    db.refresh(resume)
    db.refresh(jd)

    analysis = Analysis(resume_id=resume.id, jd_id=jd.id, status="Analysed")
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    _write_report_file(analysis.id, resume.id, jd.id, overall=overall)
    return analysis.id


@pytest.fixture
def clean_reports():
    """Remove any report files created during a test (DB rows roll back; files don't)."""
    before = set(os.listdir(REPORT_DIR)) if os.path.exists(REPORT_DIR) else set()
    yield
    after = set(os.listdir(REPORT_DIR)) if os.path.exists(REPORT_DIR) else set()
    for name in after - before:
        try:
            os.remove(os.path.join(REPORT_DIR, name))
        except OSError:
            pass


def test_create_history(client, db_session, clean_reports):
    """A completed evaluation's persisted report appears in history."""
    analysis_id = _seed_completed_analysis(db_session, "alice.pdf", "backend_role.docx", overall=88)

    response = client.get("/api/v1/analysis/history")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True

    entry = next(i for i in body["data"] if i["analysis_id"] == analysis_id)
    assert entry["resume_filename"] == "alice.pdf"
    assert entry["jd_filename"] == "backend_role.docx"
    assert entry["overall_score"] == 88
    assert entry["recruiter_recommendation"].startswith("Highly Recommended")
    assert "summary" in entry and entry["summary"]
    assert entry["timestamp"] is not None


def test_fetch_history_newest_first(client, db_session, clean_reports):
    """History lists all analyses sorted newest first."""
    id1 = _seed_completed_analysis(db_session, "r1.pdf", "j1.docx")
    id2 = _seed_completed_analysis(db_session, "r2.pdf", "j2.docx")
    id3 = _seed_completed_analysis(db_session, "r3.pdf", "j3.docx")

    response = client.get("/api/v1/analysis/history")
    assert response.status_code == 200
    returned_ids = [i["analysis_id"] for i in response.json()["data"]]

    seeded_order = [i for i in returned_ids if i in {id1, id2, id3}]
    assert seeded_order == [id3, id2, id1]  # newest first


def test_fetch_single_report(client, db_session, clean_reports):
    """Fetching one entry returns its full stored hiring report."""
    analysis_id = _seed_completed_analysis(db_session, "solo.pdf", "role.docx", overall=91)

    response = client.get(f"/api/v1/analysis/history/{analysis_id}")
    assert response.status_code == 200
    data = response.json()["data"]

    assert data["analysis_id"] == analysis_id
    assert data["resume_filename"] == "solo.pdf"
    assert data["report"]["overall_score"] == 91
    assert data["report"]["recruiter_recommendation"].startswith("Highly Recommended")
    assert "requirements" in data["report"]

    # Unknown ID -> 404
    missing = client.get("/api/v1/analysis/history/999999")
    assert missing.status_code == 404


def test_delete_single(client, db_session, clean_reports):
    """Deleting one entry removes it (and its file) but leaves the rest."""
    keep_id = _seed_completed_analysis(db_session, "keep.pdf", "keep.docx")
    drop_id = _seed_completed_analysis(db_session, "drop.pdf", "drop.docx")
    drop_path = os.path.join(REPORT_DIR, f"analysis_{drop_id}.json")

    response = client.delete(f"/api/v1/analysis/history/{drop_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True

    assert not os.path.exists(drop_path)
    assert client.get(f"/api/v1/analysis/history/{drop_id}").status_code == 404
    assert client.get(f"/api/v1/analysis/history/{keep_id}").status_code == 200

    # Deleting again -> 404
    assert client.delete(f"/api/v1/analysis/history/{drop_id}").status_code == 404


def test_delete_all(client, db_session, clean_reports):
    """Clearing history removes every entry and its file."""
    id1 = _seed_completed_analysis(db_session, "a.pdf", "a.docx")
    id2 = _seed_completed_analysis(db_session, "b.pdf", "b.docx")

    response = client.delete("/api/v1/analysis/history")
    assert response.status_code == 200
    assert response.json()["success"] is True

    assert client.get("/api/v1/analysis/history").json()["data"] == []
    assert not os.path.exists(os.path.join(REPORT_DIR, f"analysis_{id1}.json"))
    assert not os.path.exists(os.path.join(REPORT_DIR, f"analysis_{id2}.json"))
