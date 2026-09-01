"""
Prompt Lab endpoints — measure a system prompt, compare two, keep the history.

Thin by design (Golden Rule 7): grading lives in `prompt_rules_service`, running in
`prompt_lab_service`, and this module only translates HTTP to those and wraps the
standard `{success, message, data}` envelope.
"""
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import logger
from app.models.database import PromptRun, PromptSuite
from app.schemas.prompt_lab import (
    RuleInfo,
    RunDetail,
    RunRequest,
    RunResult,
    RunSummary,
    ScoreRequest,
    ScoreResult,
    Suite,
    SuiteCreate,
    SuiteUpdate,
)
from app.schemas.response import ApiResponse
from app.services.ai import prompt_lab_service as lab
from app.services.ai import prompt_rules_service as rules

router = APIRouter()


def _suite_out(row: PromptSuite) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description or "",
        "prompt": row.prompt,
        "cases": json.loads(row.cases or "[]"),
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def _run_out(row: PromptRun, with_detail: bool = False) -> dict:
    data = {
        "id": row.id,
        "suite_id": row.suite_id,
        "label": row.label,
        "prompt_hash": row.prompt_hash,
        "model": row.model or "",
        "total_cases": row.total_cases,
        "clean_cases": row.clean_cases,
        "passed": row.passed,
        "failed": row.failed,
        "score": row.score,
        "created_at": row.created_at.isoformat(),
    }
    if with_detail:
        data["results"] = json.loads(row.results or "{}")
    return data


@router.get("/rules", response_model=ApiResponse[List[RuleInfo]])
def rule_catalog():
    """Every compliance rule the lab can apply, with what it measures."""
    return ApiResponse[List[RuleInfo]](
        success=True,
        message="Compliance rules retrieved.",
        data=[RuleInfo(**r) for r in rules.RULE_CATALOG],
    )


@router.get("/starter", response_model=ApiResponse[dict])
def starter():
    """
    The person-query prompt as it runs today, with the turns that exercise each clause.

    Shipped from the backend rather than typed into the UI so that the prompt under test
    and the suite that grades it stay one artefact.
    """
    return ApiResponse[dict](
        success=True, message="Starter suite retrieved.", data=lab.starter_suite()
    )


@router.post("/run", response_model=ApiResponse[RunResult])
def run(payload: RunRequest, db: Session = Depends(get_db)):
    """
    Run every prompt variant over the same cases and grade both.

    One variant is a compliance check; two is the A/B. The cases are identical across
    variants by construction — a comparison whose inputs differ measures nothing.
    """
    cases = [c.model_dump() for c in payload.cases]
    result = lab.compare(
        variants=[v.model_dump() for v in payload.variants],
        cases=cases,
        temperature=payload.temperature,
        only=payload.rules,
    )

    # A run where nothing actually generated is not a datapoint. If every case errored —
    # the endpoint is down, the model id is wrong — its scores describe empty strings,
    # and storing them would put a phantom regression in the history.
    generated = any(
        case["answer"].strip()
        for variant in result["variants"]
        for case in variant["cases"]
    )

    if payload.persist and result["llm_available"] and generated:
        for variant in result["variants"]:
            db.add(PromptRun(
                suite_id=payload.suite_id,
                label=variant["label"],
                prompt_hash=variant["prompt_hash"],
                model=result["model"],
                total_cases=variant["total_cases"],
                clean_cases=variant["clean_cases"],
                passed=variant["passed"],
                failed=variant["failed"],
                score=variant["score"],
                results=json.dumps(variant),
            ))
        db.commit()

    if not result["llm_available"]:
        message = "No LLM is configured, so no answers could be generated."
    elif not generated:
        message = "The configured LLM returned nothing — see the per-case error. Nothing was recorded."
    else:
        message = "Prompt run complete."
    return ApiResponse[RunResult](success=True, message=message, data=RunResult(**result))


@router.post("/score", response_model=ApiResponse[ScoreResult])
def score(payload: ScoreRequest):
    """
    Grade an answer that already exists — a playground reply, or one pasted in from the
    live assistant. Deterministic and LLM-free, so it works with no key configured.
    """
    data = lab.score_case(
        prompt=payload.prompt,
        case=payload.case.model_dump(),
        answer=payload.answer,
        only=payload.rules,
    )
    return ApiResponse[ScoreResult](
        success=True, message="Answer graded.", data=ScoreResult(**data)
    )


@router.get("/suites", response_model=ApiResponse[List[Suite]])
def list_suites(db: Session = Depends(get_db)):
    rows = db.query(PromptSuite).order_by(PromptSuite.updated_at.desc()).all()
    return ApiResponse[List[Suite]](
        success=True, message=f"{len(rows)} suite(s) retrieved.",
        data=[Suite(**_suite_out(r)) for r in rows],
    )


@router.post("/suites", response_model=ApiResponse[Suite])
def create_suite(payload: SuiteCreate, db: Session = Depends(get_db)):
    row = PromptSuite(
        name=payload.name.strip() or "Untitled suite",
        description=payload.description,
        prompt=payload.prompt,
        cases=json.dumps([c.model_dump() for c in payload.cases]),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info(f"Prompt suite '{row.name}' saved (id={row.id}).")
    return ApiResponse[Suite](success=True, message="Suite saved.", data=Suite(**_suite_out(row)))


@router.put("/suites/{suite_id}", response_model=ApiResponse[Suite])
def update_suite(suite_id: int, payload: SuiteUpdate, db: Session = Depends(get_db)):
    row = db.query(PromptSuite).filter(PromptSuite.id == suite_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Suite {suite_id} not found.")
    if payload.name is not None:
        row.name = payload.name
    if payload.description is not None:
        row.description = payload.description
    if payload.prompt is not None:
        row.prompt = payload.prompt
    if payload.cases is not None:
        row.cases = json.dumps([c.model_dump() for c in payload.cases])
    db.commit()
    db.refresh(row)
    return ApiResponse[Suite](success=True, message="Suite updated.", data=Suite(**_suite_out(row)))


@router.delete("/suites/{suite_id}", response_model=ApiResponse[dict])
def delete_suite(suite_id: int, db: Session = Depends(get_db)):
    row = db.query(PromptSuite).filter(PromptSuite.id == suite_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Suite {suite_id} not found.")
    db.delete(row)
    db.commit()
    return ApiResponse[dict](success=True, message="Suite deleted.", data={"id": suite_id})


@router.get("/runs", response_model=ApiResponse[List[RunSummary]])
def list_runs(
    suite_id: Optional[int] = Query(None, description="Limit to one suite's history."),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Newest first — the regression history a prompt edit is judged against."""
    q = db.query(PromptRun)
    if suite_id is not None:
        q = q.filter(PromptRun.suite_id == suite_id)
    rows = q.order_by(PromptRun.created_at.desc(), PromptRun.id.desc()).limit(limit).all()
    return ApiResponse[List[RunSummary]](
        success=True, message=f"{len(rows)} run(s) retrieved.",
        data=[RunSummary(**_run_out(r)) for r in rows],
    )


@router.get("/runs/{run_id}", response_model=ApiResponse[RunDetail])
def get_run(run_id: int, db: Session = Depends(get_db)):
    """A past run reopened in full, rather than remembered as a number."""
    row = db.query(PromptRun).filter(PromptRun.id == run_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    return ApiResponse[RunDetail](
        success=True, message="Run retrieved.", data=RunDetail(**_run_out(row, with_detail=True))
    )
