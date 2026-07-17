from datetime import datetime, timedelta

from app.models.database import Resume, JobDescription, Analysis
from app.repositories import analytics_repository as repo
from app.services import analytics as analytics_service


def _seed(db, *, resume="r.pdf", jd="j.docx", overall=80, coverage=70, experience=60,
          project=50, quality=85, created_at=None):
    r = Resume(filename=resume, status="Indexed")
    j = JobDescription(filename=jd, status="Indexed")
    db.add_all([r, j])
    db.commit()
    db.refresh(r)
    db.refresh(j)

    kwargs = dict(
        resume_id=r.id, jd_id=j.id, status="Analysed",
        overall_score=overall, coverage_score=coverage, experience_score=experience,
        project_score=project, quality_score=quality,
    )
    if created_at is not None:
        kwargs["created_at"] = created_at
    a = Analysis(**kwargs)
    db.add(a)
    db.commit()
    db.refresh(a)
    return a.id


# ------------------------- Aggregation (repository) -------------------------

def test_repo_overall_statistics(db_session):
    _seed(db_session, overall=90, coverage=80)
    _seed(db_session, overall=50, coverage=40)
    stats = repo.overall_statistics(db_session)
    assert stats["total"] == 2
    assert round(float(stats["avg_overall"]), 2) == 70.0
    assert round(float(stats["avg_coverage"]), 2) == 60.0


def test_repo_recommendation_counts(db_session):
    _seed(db_session, overall=95)
    _seed(db_session, overall=70)
    _seed(db_session, overall=30)
    assert repo.recommendation_counts(db_session) == (1, 1, 1)


def test_repo_score_distribution(db_session):
    for score in [10, 30, 90, 85]:
        _seed(db_session, overall=score)
    assert repo.score_distribution(db_session) == [1, 1, 0, 0, 2]


def test_repo_top_resumes(db_session):
    _seed(db_session, resume="popular.pdf")
    _seed(db_session, resume="popular.pdf")
    _seed(db_session, resume="rare.pdf")
    top = repo.top_resumes(db_session, limit=5)
    assert top[0] == ("popular.pdf", 2)
    assert ("rare.pdf", 1) in top


def test_repo_daily_counts_windowed(db_session):
    now = datetime.utcnow()
    _seed(db_session, created_at=now)
    _seed(db_session, created_at=now)
    _seed(db_session, created_at=now - timedelta(days=5))
    _seed(db_session, created_at=now - timedelta(days=90))  # outside window
    daily = repo.daily_counts(db_session, days=30)
    assert sum(c for _, c in daily) == 3
    assert (now.date().isoformat(), 2) in daily


# ----------------------------- Unit (service) -------------------------------

def test_service_overall_statistics(db_session):
    _seed(db_session, overall=90, coverage=80, experience=70, project=60, quality=100)
    _seed(db_session, overall=70, coverage=60, experience=40, project=20, quality=80)
    _seed(db_session, overall=30, coverage=10, experience=0, project=0, quality=60)
    stats = analytics_service.get_overall_statistics(db_session)
    assert stats.total_analyses == 3
    assert (stats.selected, stats.borderline, stats.rejected) == (1, 1, 1)
    assert stats.average_overall_score == round((90 + 70 + 30) / 3, 2)
    assert stats.average_coverage_score == round((80 + 60 + 10) / 3, 2)


def test_service_recommendation_distribution_percentages(db_session):
    _seed(db_session, overall=95)
    _seed(db_session, overall=90)
    _seed(db_session, overall=40)
    dist = analytics_service.get_recommendation_distribution(db_session)
    by_label = {b.label: b for b in dist.distribution}
    assert by_label["Selected"].count == 2
    assert by_label["Selected"].percentage == round(2 / 3 * 100, 2)
    assert round(sum(b.percentage for b in dist.distribution)) == 100


def test_service_recent_activity_newest_first(db_session):
    now = datetime.utcnow()
    old = _seed(db_session, overall=30, created_at=now - timedelta(days=2))
    mid = _seed(db_session, overall=70, created_at=now - timedelta(days=1))
    new = _seed(db_session, overall=95, created_at=now)
    recent = analytics_service.get_recent_activity(db_session, limit=10)
    assert [r.analysis_id for r in recent] == [new, mid, old]
    assert recent[0].recommendation == "Selected"
    assert recent[2].recommendation == "Rejected"


def test_service_trends_shapes(db_session):
    _seed(db_session, created_at=datetime.utcnow())
    _seed(db_session, created_at=datetime.utcnow())
    trends = analytics_service.get_trends(db_session)
    assert sum(p.count for p in trends.daily) == 2
    assert len(trends.weekly) >= 1
    assert len(trends.monthly) >= 1


# ------------------------------ Endpoint tests ------------------------------

def test_endpoint_overview(client, db_session):
    _seed(db_session, overall=90)
    body = client.get("/api/v1/analytics/overview").json()
    assert body["success"] is True
    assert body["data"]["total_analyses"] == 1
    assert body["data"]["selected"] == 1


def test_endpoint_score_distribution(client, db_session):
    _seed(db_session, overall=75)
    ranges = client.get("/api/v1/analytics/score-distribution").json()["data"]["ranges"]
    assert {r["label"]: r["count"] for r in ranges}["61-80"] == 1


def test_endpoint_recommendation_distribution(client, db_session):
    _seed(db_session, overall=85)
    data = client.get("/api/v1/analytics/recommendation-distribution").json()["data"]
    assert data["total_analyses"] == 1
    assert {b["label"]: b["count"] for b in data["distribution"]}["Selected"] == 1


def test_endpoint_trends(client, db_session):
    _seed(db_session, created_at=datetime.utcnow())
    data = client.get("/api/v1/analytics/trends").json()["data"]
    assert "daily" in data and "weekly" in data and "monthly" in data


def test_endpoint_top_resumes_and_jobs(client, db_session):
    _seed(db_session, resume="star.pdf", jd="star.docx")
    _seed(db_session, resume="star.pdf", jd="star.docx")
    resumes = client.get("/api/v1/analytics/top-resumes?limit=3").json()["data"]
    jobs = client.get("/api/v1/analytics/top-jobs?limit=3").json()["data"]
    assert resumes[0] == {"name": "star.pdf", "count": 2}
    assert jobs[0] == {"name": "star.docx", "count": 2}


def test_endpoint_recent(client, db_session):
    _seed(db_session, overall=95)
    data = client.get("/api/v1/analytics/recent?limit=5").json()["data"]
    assert len(data) == 1
    assert data[0]["recommendation"] == "Selected"


def test_endpoint_limit_validation(client, db_session):
    assert client.get("/api/v1/analytics/top-resumes?limit=0").status_code == 422
    assert client.get("/api/v1/analytics/recent?limit=1000").status_code == 422


# ------------------------------- Empty database -----------------------------

def test_empty_overview(client, db_session):
    body = client.get("/api/v1/analytics/overview").json()["data"]
    assert body["total_analyses"] == 0
    assert body["selected"] == 0
    assert body["average_overall_score"] == 0.0  # no ZeroDivision / None


def test_empty_distribution_and_recommendation(client, db_session):
    dist = client.get("/api/v1/analytics/score-distribution").json()["data"]
    assert sum(r["count"] for r in dist["ranges"]) == 0
    rec = client.get("/api/v1/analytics/recommendation-distribution").json()["data"]
    assert rec["total_analyses"] == 0
    assert all(b["percentage"] == 0.0 for b in rec["distribution"])


def test_empty_trends_and_recent(client, db_session):
    trends = client.get("/api/v1/analytics/trends").json()["data"]
    assert trends["daily"] == [] and trends["weekly"] == [] and trends["monthly"] == []
    assert client.get("/api/v1/analytics/recent").json()["data"] == []


# ------------------------------- Large dataset ------------------------------

def test_large_dataset_aggregation(client, db_session):
    n = 120
    resumes = [Resume(filename=f"r{i}.pdf", status="Indexed") for i in range(n)]
    jds = [JobDescription(filename=f"j{i}.docx", status="Indexed") for i in range(n)]
    db_session.add_all(resumes + jds)
    db_session.commit()
    analyses = [
        Analysis(
            resume_id=resumes[i].id, jd_id=jds[i].id, status="Analysed",
            overall_score=i % 101, coverage_score=50, experience_score=50,
            project_score=50, quality_score=50,
        )
        for i in range(n)
    ]
    db_session.add_all(analyses)
    db_session.commit()

    overview = client.get("/api/v1/analytics/overview").json()["data"]
    assert overview["total_analyses"] == n
    assert overview["selected"] + overview["borderline"] + overview["rejected"] == n

    dist = client.get("/api/v1/analytics/score-distribution").json()["data"]
    assert sum(r["count"] for r in dist["ranges"]) == n


# --------------------------------- Regression -------------------------------

def test_dashboard_still_matches_analytics(client, db_session):
    """Existing dashboard endpoint stays consistent with analytics (no drift)."""
    _seed(db_session, overall=90, coverage=80, experience=70, project=60, quality=100)
    _seed(db_session, overall=40, coverage=30, experience=20, project=10, quality=50)

    dashboard = client.get("/api/v1/dashboard/overview").json()["data"]
    analytics = client.get("/api/v1/analytics/overview").json()["data"]

    assert dashboard["total_analyses"] == analytics["total_analyses"] == 2
    assert dashboard["selected_count"] == analytics["selected"]
    assert dashboard["average_skill_score"] == analytics["average_coverage_score"]
    assert dashboard["average_overall_score"] == analytics["average_overall_score"]
