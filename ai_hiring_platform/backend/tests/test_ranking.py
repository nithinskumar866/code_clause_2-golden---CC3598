"""Ranking service: leaderboard ordering, top candidate, and failure resilience.

The heavy pipeline (`run_evaluation`) is monkeypatched so these tests are fast
and deterministic — they verify the ranking/aggregation logic, not the RAG."""
import pytest
from app.models.database import Resume, JobDescription
from app.services import ranking as ranking_service


def _seed(db):
    jd = JobDescription(filename="backend_role.pdf", status="Uploaded")
    db.add(jd)
    resumes = [Resume(filename=f"cand_{i}.pdf", status="Uploaded") for i in range(3)]
    db.add_all(resumes)
    db.commit()
    for r in resumes:
        db.refresh(r)
    db.refresh(jd)
    return jd, resumes


def _report(overall, missing=None):
    return {
        "overall_score": overall, "coverage_score": overall, "experience_score": overall,
        "project_score": overall, "quality_score": overall, "confidence_score": overall,
        "recruiter_recommendation": "Proceed" if overall >= 60 else "Reject",
        "requirements": [
            {"status": "Matched"}, {"status": "Partial"}, {"status": "Missing"},
        ],
        "authenticity": {"credibility_score": overall, "keyword_stuffing_risk": "Low"},
        "candidate_profile": {"seniority_fit": "Meets"},
        "missing_skills": missing or ["Kubernetes"],
    }


def test_rank_orders_by_overall_score_desc(db_session, monkeypatch):
    jd, resumes = _seed(db_session)
    scores = {resumes[0].id: 55, resumes[1].id: 90, resumes[2].id: 72}

    def fake_run(db, resume_id, jd_id):
        return 1000 + resume_id, _report(scores[resume_id])

    monkeypatch.setattr(ranking_service, "run_evaluation", fake_run)

    result = ranking_service.rank_candidates(db_session, jd.id, [r.id for r in resumes])

    assert result.candidate_count == 3
    assert result.evaluated_count == 3
    assert [e.overall_score for e in result.entries] == [90, 72, 55]
    assert [e.rank for e in result.entries] == [1, 2, 3]
    assert result.top_candidate.overall_score == 90
    assert result.entries[0].matched_count == 1
    assert result.entries[0].missing_count == 1
    assert result.entries[0].credibility_score == 90


def test_rank_handles_failures_and_sinks_them(db_session, monkeypatch):
    jd, resumes = _seed(db_session)

    def fake_run(db, resume_id, jd_id):
        if resume_id == resumes[1].id:
            raise RuntimeError("boom")
        return 1, _report(80 if resume_id == resumes[0].id else 65)

    monkeypatch.setattr(ranking_service, "run_evaluation", fake_run)

    result = ranking_service.rank_candidates(db_session, jd.id, [r.id for r in resumes])

    assert result.evaluated_count == 2
    # Failure ranks last regardless of its (zero) score.
    assert result.entries[-1].error is not None
    assert result.entries[-1].resume_id == resumes[1].id
    assert result.top_candidate.overall_score == 80


def test_rank_unknown_jd_raises(db_session):
    from app.core.exceptions import NotFoundError
    with pytest.raises(NotFoundError):
        ranking_service.rank_candidates(db_session, 999999, [1])


def test_rank_dedupes_resume_ids(db_session, monkeypatch):
    jd, resumes = _seed(db_session)
    calls = []

    def fake_run(db, resume_id, jd_id):
        calls.append(resume_id)
        return 1, _report(70)

    monkeypatch.setattr(ranking_service, "run_evaluation", fake_run)
    rid = resumes[0].id
    result = ranking_service.rank_candidates(db_session, jd.id, [rid, rid, rid])
    assert result.candidate_count == 1
    assert calls == [rid]
