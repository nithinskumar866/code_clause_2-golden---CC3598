"""
Multi-candidate ranking service.

Evaluates several resumes against a single job description and produces a
leaderboard. It reuses the exact same LangGraph pipeline as the single
`/evaluate` endpoint (via `run_evaluation`) — no separate scoring path — so a
candidate's rank is their real, explainable overall score, not a shortcut. This
also keeps evaluation logic in one place (single source of truth) shared by
both the single-evaluate route and ranking.
"""
import os
from typing import Dict, Any, List, Tuple
from sqlalchemy.orm import Session

from app.core.constants import RESUME_UPLOAD_DIR, JOB_UPLOAD_DIR, STATUS_ANALYSED, STATUS_FAILED
from app.core.exceptions import NotFoundError
from app.core.logging import logger
from app.models.database import Resume, JobDescription, Analysis
from app.schemas.ranking import RankingEntry, RankingResponse
from app.workflows.hiring_workflow import execute_hiring_pipeline


def run_evaluation(db: Session, resume_id: int, jd_id: int) -> Tuple[int, Dict[str, Any]]:
    """
    Validate inputs, run the full LangGraph hiring pipeline for one resume × one
    JD, persist status, and return (analysis_id, final_report). Raises
    NotFoundError (→404) for a missing resume/JD/source file; re-raises pipeline
    failures after marking the analysis FAILED. Single source of truth used by
    both the single-evaluate route and the ranking service.
    """
    db_resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not db_resume:
        raise NotFoundError(f"Resume ID {resume_id} not found.")

    db_jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()
    if not db_jd:
        raise NotFoundError(f"Job Description ID {jd_id} not found.")

    resume_path = os.path.join(RESUME_UPLOAD_DIR, f"{db_resume.id}_{db_resume.filename}")
    jd_path = os.path.join(JOB_UPLOAD_DIR, f"{db_jd.id}_{db_jd.filename}")
    if not os.path.exists(resume_path):
        raise NotFoundError("Resume source document not found on disk.")
    if not os.path.exists(jd_path):
        raise NotFoundError("Job Description source document not found on disk.")

    db_analysis = db.query(Analysis).filter(
        Analysis.resume_id == resume_id, Analysis.jd_id == jd_id
    ).first()
    if not db_analysis:
        db_analysis = Analysis(resume_id=resume_id, jd_id=jd_id, status="Indexed")
        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)

    try:
        final_report = execute_hiring_pipeline(
            resume_id=resume_id, resume_path=resume_path,
            jd_id=jd_id, jd_path=jd_path, analysis_id=db_analysis.id,
        )
        db_analysis.status = STATUS_ANALYSED
        db_resume.status = "Indexed"
        db_jd.status = "Indexed"
        db.commit()
        db.refresh(db_analysis)
        return db_analysis.id, final_report
    except Exception:
        db_analysis.status = STATUS_FAILED
        db.commit()
        raise


def _entry_from_report(resume: Resume, analysis_id: int, report: Dict[str, Any]) -> RankingEntry:
    """Distil a full hiring report into a compact leaderboard row."""
    reqs = report.get("requirements", []) or []
    matched = sum(1 for r in reqs if r.get("status") == "Matched")
    partial = sum(1 for r in reqs if r.get("status") == "Partial")
    missing = sum(1 for r in reqs if r.get("status") == "Missing")

    auth = report.get("authenticity") or {}
    profile = report.get("candidate_profile") or {}

    return RankingEntry(
        rank=0,  # assigned after sorting
        analysis_id=analysis_id,
        resume_id=resume.id,
        resume_filename=resume.filename,
        overall_score=int(report.get("overall_score", 0) or 0),
        coverage_score=int(report.get("coverage_score", 0) or 0),
        experience_score=int(report.get("experience_score", 0) or 0),
        project_score=int(report.get("project_score", 0) or 0),
        quality_score=int(report.get("quality_score", 0) or 0),
        confidence_score=int(report.get("confidence_score", 0) or 0),
        recruiter_recommendation=report.get("recruiter_recommendation", "") or "",
        seniority_fit=profile.get("seniority_fit"),
        credibility_score=auth.get("credibility_score"),
        keyword_stuffing_risk=auth.get("keyword_stuffing_risk"),
        matched_count=matched,
        partial_count=partial,
        missing_count=missing,
        top_missing_skills=list(report.get("missing_skills", []) or [])[:5],
    )


def rank_candidates(db: Session, jd_id: int, resume_ids: List[int]) -> RankingResponse:
    """
    Evaluate every requested resume against one JD and return a leaderboard,
    best overall score first. A resume that fails to evaluate is still listed
    (with an error and a zero score) so the recruiter sees the full picture.
    """
    db_jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()
    if not db_jd:
        raise NotFoundError(f"Job Description ID {jd_id} not found.")

    # De-dupe while preserving order.
    seen: set = set()
    ordered_ids = [rid for rid in resume_ids if not (rid in seen or seen.add(rid))]

    entries: List[RankingEntry] = []
    for rid in ordered_ids:
        resume = db.query(Resume).filter(Resume.id == rid).first()
        if not resume:
            entries.append(RankingEntry(
                rank=0, resume_id=rid, resume_filename=f"(resume {rid} not found)",
                error=f"Resume ID {rid} not found.",
            ))
            continue
        try:
            analysis_id, report = run_evaluation(db, rid, jd_id)
            entries.append(_entry_from_report(resume, analysis_id, report))
        except Exception as e:  # keep the batch resilient — one bad resume must not sink the rest
            logger.error(f"Ranking: evaluation failed for resume {rid}: {e}", exc_info=True)
            entries.append(RankingEntry(
                rank=0, resume_id=rid, resume_filename=resume.filename, error=str(e),
            ))

    # Rank: successful candidates by descending overall score; failures sink to the bottom.
    entries.sort(key=lambda e: (e.error is not None, -e.overall_score))
    for i, e in enumerate(entries, start=1):
        e.rank = i

    successful = [e for e in entries if e.error is None]
    return RankingResponse(
        jd_id=jd_id,
        jd_filename=db_jd.filename,
        candidate_count=len(ordered_ids),
        evaluated_count=len(successful),
        top_candidate=successful[0] if successful else None,
        entries=entries,
    )
