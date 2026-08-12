"""
Ask every embedding model the same question, and measure how much they disagree.

WHY A DEDICATED SERVICE RATHER THAN "RUN THE CHAT THREE TIMES"
--------------------------------------------------------------
Running the chatbot once per model is the easy version and it produces a misleading
answer, for two reasons that have nothing to do with model quality:

  1. **Unequal populations.** If BGE has indexed 600 resumes and the GPU model 300,
     BGE wins by having more to choose from. That measures indexing coverage, not
     retrieval. `fair_mode` therefore confines every model to the intersection of what
     they have all indexed, and says how large that population was.

  2. **The LLM blurs the thing being measured.** What differs between these models is
     WHICH CHUNKS COME BACK. Layering a generative summary over that adds wording
     variance between runs and hides the retrieval underneath. So the deterministic
     retrieval answer is always produced, and the LLM answer is an explicit opt-in
     shown beside it rather than instead of it.

WHAT "ACCURACY" MEANS HERE
--------------------------
There is no labelled ground truth in a resume pool, so the honest measurement is
AGREEMENT, not correctness: whether the models pick the same top candidate, and how far
their result sets overlap. Unanimity means the choice of model is not currently buying
anything for this kind of question; divergence is where a human should look, and the
page puts the three answers side by side precisely so that look is cheap.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set

from sqlalchemy.orm import Session

from app.core.logging import logger
from app.services.ai import chat_service, embedding_engines, embedding_store

# Comparison runs are stateless: each model answers the question on its own, with no
# memory of previous turns. A shared conversation would let one model's answer steer
# another's, which is the opposite of a controlled comparison.
_SESSION_PREFIX = "model-comparison"


def _summarise_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    evidence = (candidate.get("evidence") or [{}])[0]
    return {
        "resume_id": candidate["resume_id"],
        "name": candidate.get("name"),
        "title": candidate.get("title"),
        "match_percentage": candidate.get("match_percentage", 0),
        "total_years": candidate.get("total_years"),
        "matched_skills": candidate.get("matched_skills") or [],
        "missing_skills": candidate.get("missing_skills") or [],
        "top_evidence": (evidence.get("text") or "")[:400] or None,
        "section": evidence.get("section"),
        "similarity": evidence.get("similarity"),
    }


def _run_one(
    db: Session,
    model: str,
    message: str,
    limit: int,
    use_llm: bool,
    restrict: Optional[Set[int]],
    coverage_by_model: Dict[str, int],
) -> Dict[str, Any]:
    """One model's column. Never raises — a failed model is reported, not fatal."""
    started = time.perf_counter()
    engine = embedding_engines.resolve_engine(model)
    if engine.name != model:
        # A remote model that cannot serve falls back to BGE, which would silently put
        # BGE's answer in another model's column. Report the outage instead.
        return {
            "model": model, "dimension": None, "available": False,
            "answer": f"{model} is not reachable right now, so it has no answer to give.",
            "answer_type": "empty", "candidates": [], "elapsed_ms": 0,
            "indexed_resumes": coverage_by_model.get(model, 0),
            "error": f"engine unavailable (resolved to '{engine.name}')",
        }

    try:
        # Deterministic first: this is the measurement.
        result = chat_service.answer(
            db=db, message=message, session_id=f"{_SESSION_PREFIX}-{model}",
            limit=limit, use_llm=False, embedding_engine=model,
            restrict_resume_ids=restrict,
        )
        llm_answer = None
        if use_llm:
            enriched = chat_service.answer(
                db=db, message=message, session_id=f"{_SESSION_PREFIX}-llm-{model}",
                limit=limit, use_llm=True, embedding_engine=model,
                restrict_resume_ids=restrict,
            )
            llm_answer = enriched.get("answer")

        return {
            "model": model,
            "dimension": engine.dimension,
            "available": True,
            "answer": result.get("answer", ""),
            "llm_answer": llm_answer,
            "answer_type": result.get("answer_type", "candidates"),
            "candidates": [_summarise_candidate(c) for c in result.get("candidates", [])],
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "indexed_resumes": coverage_by_model.get(model, 0),
            "error": None,
        }
    except Exception as e:
        logger.error(f"Model comparison failed for '{model}': {e}", exc_info=True)
        return {
            "model": model, "dimension": engine.dimension, "available": True,
            "answer": "", "answer_type": "empty", "candidates": [],
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "indexed_resumes": coverage_by_model.get(model, 0), "error": str(e),
        }


def _agreement(columns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Top-1 unanimity and pairwise result overlap."""
    answered = [c for c in columns if c.get("candidates")]
    top1 = {
        c["model"]: (c["candidates"][0]["name"] if c.get("candidates") else None)
        for c in columns
    }
    sets = {c["model"]: {x["resume_id"] for x in c["candidates"]} for c in answered}

    overlap: Dict[str, float] = {}
    for a, b in combinations(sorted(sets), 2):
        union = sets[a] | sets[b]
        overlap[f"{a}|{b}"] = round(len(sets[a] & sets[b]) / len(union), 3) if union else 0.0

    unanimous = (
        len(answered) > 1
        and len({c["candidates"][0]["resume_id"] for c in answered}) == 1
    )
    common = set.intersection(*sets.values()) if sets else set()
    return {
        "top1_unanimous": unanimous,
        "top1_by_model": top1,
        "overlap": overlap,
        "common_resume_ids": sorted(common),
    }


def compare(
    db: Session,
    message: str,
    models: Sequence[str],
    limit: int = 5,
    use_llm: bool = False,
    fair_mode: bool = True,
) -> Dict[str, Any]:
    """
    Run one question through several models and report the results side by side.

    Models run concurrently: a search is one query embedding plus a dot product, so
    they do not contend the way indexing does, and the GPU model is network-bound
    anyway. Each gets its own conversation so no model can see another's answer.
    """
    selected = [m for m in models if m]
    if not selected:
        raise ValueError("Select at least one model to compare.")

    report = embedding_store.coverage(db)
    coverage_by_model = {m["model"]: m["indexed_resumes"] for m in report["models"]}

    warnings: List[str] = []
    restrict: Optional[Set[int]] = None
    if fair_mode:
        restrict = embedding_store.comparable_resume_ids(selected)
        if not restrict:
            warnings.append(
                "These models have no resumes in common yet, so there is nothing to "
                "compare fairly. Index them over the same pool first."
            )
        else:
            behind = [m for m in selected if coverage_by_model.get(m, 0) > len(restrict)]
            if behind:
                warnings.append(
                    f"Comparing over the {len(restrict)} resumes all selected models "
                    f"have indexed. {', '.join(behind)} has indexed more, and the extra "
                    f"resumes are excluded so the comparison stays like-for-like."
                )
    else:
        spread = {m: coverage_by_model.get(m, 0) for m in selected}
        if len(set(spread.values())) > 1:
            warnings.append(
                f"Fair mode is off and these models have indexed different numbers of "
                f"resumes ({spread}). Differences below may reflect coverage rather "
                f"than retrieval quality."
            )

    compared_over = len(restrict) if restrict is not None else max(
        [coverage_by_model.get(m, 0) for m in selected] or [0]
    )

    with ThreadPoolExecutor(max_workers=min(len(selected), 3)) as pool:
        columns = list(pool.map(
            lambda m: _run_one(db, m, message, limit, use_llm, restrict, coverage_by_model),
            selected,
        ))

    logger.info(
        f"Model comparison [{message[:60]!r}] over {compared_over} resumes: "
        + ", ".join(f"{c['model']}={c['elapsed_ms']}ms/{len(c['candidates'])}" for c in columns)
    )
    return {
        "question": message,
        "models": columns,
        "agreement": _agreement(columns),
        "compared_over_resumes": compared_over,
        "fair_mode": fair_mode,
        "warnings": warnings,
    }
