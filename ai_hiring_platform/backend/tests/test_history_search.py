import os
import json
from datetime import datetime, timedelta

import pytest

from app.core.constants import REPORT_DIR
from app.models.database import Resume, JobDescription, Analysis
from app.repositories import history_repository
from app.services import analysis as history_service


def _write_report(analysis_id, overall):
    report = {
        "analysis_id": analysis_id,
        "resume_id": 1,
        "jd_id": 1,
        "retrieval_results": [],
        "overall_score": overall,
        "coverage_score": 70,
        "experience_score": 60,
        "project_score": 55,
        "confidence_score": 65,
        "quality_score": 85,
        "summary": f"Summary {analysis_id}.",
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


def _seed(db, *, resume="resume.pdf", jd="job.docx", overall=80, created_at=None, write_report=True, set_column=True):
    r = Resume(filename=resume, status="Indexed")
    j = JobDescription(filename=jd, status="Indexed")
    db.add_all([r, j])
    db.commit()
    db.refresh(r)
    db.refresh(j)

    kwargs = dict(resume_id=r.id, jd_id=j.id, status="Analysed")
    if set_column:
        kwargs["overall_score"] = overall
    if created_at is not None:
        kwargs["created_at"] = created_at
    a = Analysis(**kwargs)
    db.add(a)
    db.commit()
    db.refresh(a)
    if write_report:
        _write_report(a.id, overall)
    return a.id


def _filters(**overrides):
    base = {
        "resume_filename": None, "jd_filename": None, "recommendation": None,
        "min_score": None, "max_score": None, "date_from": None, "date_to": None,
    }
    base.update(overrides)
    return base


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


# ----------------------------- Repository tests -----------------------------

def test_repository_filters_score_in_sql(db_session, clean_reports):
    _seed(db_session, overall=90)
    _seed(db_session, overall=40)
    rows, total = history_repository.search(db_session, filters=_filters(min_score=80), sort="newest")
    assert total == 1
    assert rows[0][0].overall_score == 90


def test_repository_filename_case_insensitive(db_session, clean_reports):
    _seed(db_session, resume="Python_Engineer.pdf", overall=88)
    _seed(db_session, resume="java_dev.pdf", overall=88)
    rows, total = history_repository.search(db_session, filters=_filters(resume_filename="python"), sort="newest")
    assert total == 1
    assert rows[0][1] == "Python_Engineer.pdf"


def test_repository_sort_and_pagination(db_session, clean_reports):
    for score in [60, 90, 75]:
        _seed(db_session, overall=score)
    rows, total = history_repository.search(db_session, filters=_filters(), sort="highest_score", limit=2, offset=0)
    assert total == 3
    scores = [r[0].overall_score for r in rows]
    assert scores == [90, 75]  # highest first, limited to 2


def test_repository_recommendation_buckets(db_session, clean_reports):
    _seed(db_session, overall=95)  # Selected
    _seed(db_session, overall=70)  # Borderline
    _seed(db_session, overall=30)  # Rejected
    _, sel = history_repository.search(db_session, filters=_filters(recommendation="Selected"), sort="newest")
    _, bor = history_repository.search(db_session, filters=_filters(recommendation="Borderline"), sort="newest")
    _, rej = history_repository.search(db_session, filters=_filters(recommendation="Rejected"), sort="newest")
    assert (sel, bor, rej) == (1, 1, 1)


# ------------------------------ Service tests -------------------------------

def test_service_pagination_metadata(db_session, clean_reports):
    for i in range(5):
        _seed(db_session, overall=70 + i)
    items, meta = history_service.search_history(db_session, sort="highest_score", page=1, page_size=2)
    assert len(items) == 2
    assert meta.total_count == 5
    assert meta.total_pages == 3
    assert meta.page == 1 and meta.page_size == 2
    assert items[0].overall_score >= items[1].overall_score


def test_service_unpaged_returns_all(db_session, clean_reports):
    for _ in range(3):
        _seed(db_session, overall=80)
    items, meta = history_service.search_history(db_session)  # no pagination
    assert len(items) == 3
    assert meta.total_count == 3 and meta.total_pages == 1


def test_sync_backfills_overall_score(db_session, clean_reports):
    aid = _seed(db_session, overall=77, write_report=True, set_column=False)  # column NULL, report present
    analysis = db_session.query(Analysis).filter(Analysis.id == aid).first()
    assert analysis.overall_score is None

    history_service.sync_history_scores(db_session)
    db_session.refresh(analysis)
    assert analysis.overall_score == 77


# ------------------------------ Endpoint tests ------------------------------

def test_endpoint_default_returns_all_with_meta(client, db_session, clean_reports):
    _seed(db_session, overall=90)
    _seed(db_session, overall=50)
    response = client.get("/api/v1/analysis/history")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert len(body["data"]) == 2
    assert body["meta"]["total_count"] == 2


def test_endpoint_combined_filters(client, db_session, clean_reports):
    now = datetime.utcnow()
    match = _seed(db_session, resume="python_dev.pdf", overall=90, created_at=now)
    _seed(db_session, resume="java_dev.pdf", overall=90, created_at=now)             # wrong filename
    _seed(db_session, resume="python_junior.pdf", overall=50, created_at=now)        # wrong score
    _seed(db_session, resume="python_old.pdf", overall=90, created_at=now - timedelta(days=60))  # too old

    date_from = (now.date() - timedelta(days=30)).isoformat()
    response = client.get(
        f"/api/v1/analysis/history"
        f"?resume_filename=python&recommendation=Selected&min_score=81&date_from={date_from}&sort=newest"
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert [d["analysis_id"] for d in data] == [match]


def test_endpoint_pagination(client, db_session, clean_reports):
    for i in range(5):
        _seed(db_session, overall=70 + i)
    response = client.get("/api/v1/analysis/history?page=1&page_size=2&sort=highest_score")
    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 2
    assert body["meta"] == {"total_count": 5, "page": 1, "page_size": 2, "total_pages": 3}


def test_endpoint_empty_results(client, db_session, clean_reports):
    _seed(db_session, resume="alpha.pdf", overall=90)
    response = client.get("/api/v1/analysis/history?resume_filename=zzz-no-match")
    assert response.status_code == 200
    body = response.json()
    assert body["data"] == []
    assert body["meta"]["total_count"] == 0


def test_endpoint_large_page_number(client, db_session, clean_reports):
    _seed(db_session, overall=80)
    _seed(db_session, overall=80)
    response = client.get("/api/v1/analysis/history?page=100&page_size=10")
    assert response.status_code == 200
    body = response.json()
    assert body["data"] == []
    assert body["meta"]["total_count"] == 2
    assert body["meta"]["page"] == 100


# ------------------------------ Validation tests ----------------------------

@pytest.mark.parametrize("query", [
    "recommendation=Maybe",       # invalid enum
    "min_score=90&max_score=10",  # min > max
    "min_score=200",              # out of range
    "page=0",                     # non-positive page
    "page_size=0",                # below limit
    "page_size=1000",             # above limit
    "date_from=not-a-date",       # invalid date
    "date_from=2025-02-02&date_to=2025-01-01",  # from after to
    "sort=sideways",              # invalid sort
])
def test_endpoint_validation_rejects_bad_input(client, db_session, query):
    response = client.get(f"/api/v1/analysis/history?{query}")
    assert response.status_code == 422
