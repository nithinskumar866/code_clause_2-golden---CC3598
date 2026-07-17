import os
import json
from datetime import datetime, timedelta

import pytest

from app.core.constants import REPORT_DIR
from app.models.database import Resume, JobDescription, Analysis
from app.services import dashboard as dashboard_service


def _write_report_file(analysis_id, resume_id, jd_id, overall, coverage, experience, project, quality):
    """Persist a report JSON exactly as the Hiring Decision Agent does on disk."""
    report = {
        "analysis_id": analysis_id,
        "candidate_id": resume_id,
        "resume_id": resume_id,
        "jd_id": jd_id,
        "retrieval_results": [],
        "overall_score": overall,
        "coverage_score": coverage,
        "experience_score": experience,
        "project_score": project,
        "confidence_score": 78,
        "quality_score": quality,
        "summary": f"Summary for analysis {analysis_id}.",
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


def _seed(db, overall=80, coverage=70, experience=60, project=50, quality=90, created_at=None):
    """Recreate what upload + a completed evaluation leave persisted."""
    resume = Resume(filename="r.pdf", status="Indexed")
    jd = JobDescription(filename="j.docx", status="Indexed")
    db.add_all([resume, jd])
    db.commit()
    db.refresh(resume)
    db.refresh(jd)

    kwargs = dict(resume_id=resume.id, jd_id=jd.id, status="Analysed")
    if created_at is not None:
        kwargs["created_at"] = created_at
    analysis = Analysis(**kwargs)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    _write_report_file(analysis.id, resume.id, jd.id, overall, coverage, experience, project, quality)
    return analysis.id


@pytest.fixture
def clean_reports():
    """Remove report files created during a test (DB rows roll back; files don't)."""
    before = set(os.listdir(REPORT_DIR)) if os.path.exists(REPORT_DIR) else set()
    yield
    after = set(os.listdir(REPORT_DIR)) if os.path.exists(REPORT_DIR) else set()
    for name in after - before:
        try:
            os.remove(os.path.join(REPORT_DIR, name))
        except OSError:
            pass


# --------------------------- Unit tests (service) ---------------------------

def test_overview_counts_and_averages(db_session, clean_reports):
    # selected (>=80), borderline (60-79), rejected (<60)
    _seed(db_session, overall=90, coverage=80, experience=60, project=40, quality=100)
    _seed(db_session, overall=70, coverage=60, experience=40, project=20, quality=80)
    _seed(db_session, overall=30, coverage=10, experience=0, project=0, quality=60)

    overview = dashboard_service.get_overview(db_session)

    assert overview.total_analyses == 3
    assert overview.selected_count == 1
    assert overview.borderline_count == 1
    assert overview.rejected_count == 1
    assert overview.average_overall_score == round((90 + 70 + 30) / 3, 2)
    # "skill score" maps to coverage_score
    assert overview.average_skill_score == round((80 + 60 + 10) / 3, 2)
    assert overview.average_quality_score == round((100 + 80 + 60) / 3, 2)


def test_overview_empty_history_is_safe(db_session, clean_reports):
    overview = dashboard_service.get_overview(db_session)
    assert overview.total_analyses == 0
    assert overview.selected_count == 0
    assert overview.average_overall_score == 0.0  # no ZeroDivisionError


def test_trends_window_and_zero_fill(db_session, clean_reports):
    now = datetime.utcnow()
    _seed(db_session, created_at=now)                      # today
    _seed(db_session, created_at=now)                      # today
    _seed(db_session, created_at=now - timedelta(days=5))  # in window
    _seed(db_session, created_at=now - timedelta(days=40)) # outside 30-day window

    trends = dashboard_service.get_trends(db_session)

    assert trends.period_days == 30
    assert len(trends.trends) == 30                        # zero-filled
    assert trends.trends[-1].date == now.date().isoformat()
    assert trends.trends[-1].count == 2                    # two today
    assert sum(p.count for p in trends.trends) == 3        # 40-days-ago excluded


def test_distribution_buckets(db_session, clean_reports):
    _seed(db_session, overall=10)   # 0-20
    _seed(db_session, overall=30)   # 21-40
    _seed(db_session, overall=90)   # 81-100
    _seed(db_session, overall=85)   # 81-100

    dist = dashboard_service.get_distribution(db_session)

    assert dist.total_analyses == 4
    by_label = {r.label: r.count for r in dist.ranges}
    assert by_label == {"0-20": 1, "21-40": 1, "41-60": 0, "61-80": 0, "81-100": 2}


# ----------------------- Integration tests (endpoints) ----------------------

def test_overview_endpoint(client, db_session, clean_reports):
    _seed(db_session, overall=95)
    response = client.get("/api/v1/dashboard/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total_analyses"] == 1
    assert body["data"]["selected_count"] == 1


def test_trends_endpoint(client, db_session, clean_reports):
    _seed(db_session, created_at=datetime.utcnow())
    response = client.get("/api/v1/dashboard/trends")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["period_days"] == 30
    assert len(body["data"]["trends"]) == 30


def test_distribution_endpoint(client, db_session, clean_reports):
    _seed(db_session, overall=75)
    response = client.get("/api/v1/dashboard/distribution")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    ranges = {r["label"]: r["count"] for r in body["data"]["ranges"]}
    assert ranges["61-80"] == 1
