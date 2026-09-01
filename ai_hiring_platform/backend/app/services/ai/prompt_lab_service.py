"""
Prompt Lab — run a system prompt against a fixed set of recruiter turns and measure
how often it actually obeys itself.

WHY IT EXISTS
A prompt that "does not work properly" is a claim nobody can act on: which rule does it
break, on which question, and does the rewrite fix that one without breaking two others?
This service turns that into a measurement. The same cases, the same context, two
prompts side by side, and a deterministic verdict per rule
(`prompt_rules_service`) — so a prompt change is an experiment with a result rather than
a hunch.

THE DIVISION OF LABOUR IS THE USUAL ONE (CLAUDE.md §4)
The LLM only GENERATES the answer under test. Everything that decides whether that
answer is acceptable is algorithmic and reproducible. Nothing here reads a resume, ranks
a candidate or joins the hiring workflow — it is a test harness, not a third agent.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Sequence

from app.core.logging import logger
from app.services.ai import llm_service
from app.services.ai import prompt_rules_service as rules
from app.services.ai.prompt_rules_service import Expectations, Turn


def prompt_fingerprint(prompt: str) -> str:
    """Short, stable id for a prompt revision — what a regression run is keyed on."""
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def _compose_user_message(turn: Turn) -> str:
    """
    Everything the assistant would have been handed at answer time: the retrieved
    context, the thread so far, then the new question. Assembled here so both prompt
    variants see byte-identical input — a comparison where the inputs differ measures
    nothing.
    """
    parts: List[str] = []
    if turn.context.strip():
        parts.append(f"RETRIEVED CONTEXT:\n{turn.context.strip()}")
    if turn.history:
        thread = "\n".join(
            f"{(m.get('role') or 'user').upper()}: {(m.get('content') or '').strip()}"
            for m in turn.history
            if (m.get("content") or "").strip()
        )
        if thread:
            parts.append(f"CONVERSATION SO FAR:\n{thread}")
    parts.append(f"USER QUESTION: {turn.question.strip()}")
    return "\n\n".join(parts)


def generate(prompt: str, turn: Turn, temperature: float = 0.2) -> Dict[str, Any]:
    """
    One answer from the configured LLM under the prompt being tested.

    Returns `{answer, error, latency_ms}`. A missing LLM is reported, never faked: an
    invented answer would make a prompt look compliant when it was never exercised.
    """
    llm = llm_service.get_llm()
    if llm is None:
        return {"answer": "", "error": "No LLM is configured, so no answer could be generated.", "latency_ms": 0}

    message = _compose_user_message(turn)
    started = time.perf_counter()
    try:
        complete = getattr(llm, "complete", None)
        varnames = getattr(getattr(complete, "__code__", None), "co_varnames", ())
        if "system" in varnames and "temperature" in varnames:
            response = llm.complete(message, system=prompt, temperature=temperature)
        elif "system" in varnames:
            response = llm.complete(message, system=prompt)
        else:
            response = llm.complete(f"{prompt}\n\n{message}")
        text = (getattr(response, "text", "") or "").strip()
        return {"answer": text, "error": None, "latency_ms": int((time.perf_counter() - started) * 1000)}
    except Exception as e:  # a failing endpoint is a result, not a crash
        logger.error(f"Prompt Lab generation failed: {e}", exc_info=True)
        return {"answer": "", "error": str(e), "latency_ms": int((time.perf_counter() - started) * 1000)}


# --------------------------------------------------------------------------------------
# Running a suite
# --------------------------------------------------------------------------------------


def _as_turn(case: Dict[str, Any]) -> Turn:
    return Turn(
        question=(case.get("question") or "").strip(),
        context=case.get("context") or "",
        history=list(case.get("history") or []),
    )


def score_case(
    prompt: str,
    case: Dict[str, Any],
    answer: str,
    only: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Grade an answer that already exists — the playground path, and the unit-test path."""
    return rules.evaluate(
        answer=answer,
        turn=_as_turn(case),
        expectations=Expectations.from_dict(case.get("expectations")),
        prompt=prompt,
        only=only,
    )


def run_variant(
    prompt: str,
    cases: List[Dict[str, Any]],
    label: str = "A",
    temperature: float = 0.2,
    only: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Every case through one prompt, scored. The unit of an A/B side."""
    case_results: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        turn = _as_turn(case)
        gen = generate(prompt, turn, temperature=temperature)
        verdict = rules.evaluate(
            answer=gen["answer"],
            turn=turn,
            expectations=Expectations.from_dict(case.get("expectations")),
            prompt=prompt,
            only=only,
        )
        case_results.append({
            "case_id": case.get("id") or f"case-{i + 1}",
            "name": case.get("name") or case.get("question") or f"Case {i + 1}",
            "question": turn.question,
            "answer": gen["answer"],
            "error": gen["error"],
            "latency_ms": gen["latency_ms"],
            **verdict,
        })

    return {"label": label, "prompt_hash": prompt_fingerprint(prompt), **_aggregate(case_results)}


def _aggregate(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Suite-level numbers.

    The headline is the RULE pass rate, not the share of clean cases: a prompt that
    breaks one rule everywhere and a prompt that fails one case completely are different
    problems, and only the per-rule breakdown tells them apart. `by_rule` is what points
    at the clause to rewrite.
    """
    passed = sum(c["passed"] for c in case_results)
    failed = sum(c["failed"] for c in case_results)
    applicable = passed + failed

    by_rule: Dict[str, Dict[str, Any]] = {}
    for case in case_results:
        for r in case["rules"]:
            slot = by_rule.setdefault(
                r["rule"], {"rule": r["rule"], "title": r["title"], "passed": 0, "failed": 0, "not_applicable": 0}
            )
            if r["status"] == rules.PASS:
                slot["passed"] += 1
            elif r["status"] == rules.FAIL:
                slot["failed"] += 1
            else:
                slot["not_applicable"] += 1
    for slot in by_rule.values():
        denom = slot["passed"] + slot["failed"]
        slot["score"] = round(100.0 * slot["passed"] / denom, 1) if denom else None

    clean = sum(1 for c in case_results if c["failed"] == 0 and not c["error"])
    return {
        "cases": case_results,
        "total_cases": len(case_results),
        "clean_cases": clean,
        "passed": passed,
        "failed": failed,
        "score": round(100.0 * passed / applicable, 1) if applicable else None,
        "by_rule": sorted(by_rule.values(), key=lambda s: (s["score"] is None, s["score"] or 0)),
        "errors": [c["error"] for c in case_results if c["error"]],
    }


def compare(
    variants: List[Dict[str, str]],
    cases: List[Dict[str, Any]],
    temperature: float = 0.2,
    only: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Run every prompt variant over the same cases and diff them.

    `deltas` is the point of the screen: which rules the rewrite fixed, which it broke,
    and which it left alone. A prompt edit that raises the headline while quietly
    regressing the grounding rules is the failure mode this catches.
    """
    results = [
        run_variant(v.get("prompt") or "", cases, label=v.get("label") or f"V{i + 1}",
                    temperature=temperature, only=only)
        for i, v in enumerate(variants)
    ]

    deltas: List[Dict[str, Any]] = []
    if len(results) >= 2:
        base, other = results[0], results[1]
        base_rules = {r["rule"]: r for r in base["by_rule"]}
        for r in other["by_rule"]:
            b = base_rules.get(r["rule"])
            if not b:
                continue
            deltas.append({
                "rule": r["rule"],
                "title": r["title"],
                "baseline_score": b["score"],
                "variant_score": r["score"],
                "delta": None if (b["score"] is None or r["score"] is None) else round(r["score"] - b["score"], 1),
            })
        deltas.sort(key=lambda d: (d["delta"] is None, d["delta"] if d["delta"] is not None else 0))

    return {
        "variants": results,
        "deltas": deltas,
        "llm_available": llm_service.get_llm() is not None,
        "model": _model_label(),
    }


def _model_label() -> str:
    """Which model produced these answers — a run without it is not reproducible."""
    llm = llm_service.get_llm()
    if llm is None:
        return "none (deterministic fallback — no answers generated)"
    model = getattr(llm, "model", "") or llm_service.resolve_model(llm_service.resolve_provider())
    return str(model or llm.__class__.__name__)


# --------------------------------------------------------------------------------------
# The starter suite — the prompt currently in the field, and the turns it gets wrong
# --------------------------------------------------------------------------------------

STARTER_PROMPT = """When answering a person/employee query:
- Use ONLY the name/role data found in PEOPLE DIRECTORY or retrieved context.
- Keep the answer to 1-2 short lines about who the person is at the company.
- Use chat history for follow-ups (for example: "he", "his") to continue with the same person.
- If user asks about years/experience:
    1) Provide exact years only when explicitly present in retrieved context.
    2) If exact years are not present, respond briefly that exact years are not verified in current data and avoid long fallback text.
- Include LinkedIn URL only when relevant:
    1) first answer about that person in the current thread, OR
    2) user explicitly asks for link/profile/more details, OR
    3) exact years are unavailable and the link helps verification.
- For person skill questions:
    1) If Skills=... is present in retrieved profile context, list those skills first.
    2) Then include LinkedInSkills URL (if available) for full details.
- If LinkedInSkills URL is missing but LinkedIn profile URL exists, derive and include: <LinkedInProfileUrl>/details/skills/
- Do NOT repeat the same LinkedIn URL in every follow-up answer.
- Do NOT use markdown link format like [url](url); output plain URL only.
- Do NOT provide job list suggestions unless the user explicitly asks to search/list/find jobs."""

_DIRECTORY = (
    "PEOPLE DIRECTORY:\n"
    "Name=Priya Raman | Role=Senior Data Engineer | Team=Platform | "
    "LinkedIn=https://www.linkedin.com/in/priya-raman | "
    "Skills=Apache Spark, Airflow, dbt, Snowflake, Python"
)
_DIRECTORY_WITH_YEARS = _DIRECTORY + " | Experience=9 years"

STARTER_CASES: List[Dict[str, Any]] = [
    {
        "id": "intro",
        "name": "First mention of a person",
        "question": "Who is Priya Raman?",
        "context": _DIRECTORY,
        "history": [],
        "expectations": {"max_lines": 2, "max_chars": 320, "link_allowed": True},
    },
    {
        "id": "years-unknown",
        "name": "Years the context does not verify",
        "question": "How many years of experience does she have?",
        "context": _DIRECTORY,
        "history": [
            {"role": "user", "content": "Who is Priya Raman?"},
            {"role": "assistant", "content":
                "Priya Raman is a Senior Data Engineer on the Platform team. "
                "https://www.linkedin.com/in/priya-raman"},
        ],
        "expectations": {"max_lines": 2, "max_chars": 260, "years_known": False, "link_allowed": True},
    },
    {
        "id": "years-known",
        "name": "Years the context does verify",
        "question": "How many years has she worked in data engineering?",
        "context": _DIRECTORY_WITH_YEARS,
        "history": [
            {"role": "user", "content": "Who is Priya Raman?"},
            {"role": "assistant", "content":
                "Priya Raman is a Senior Data Engineer on the Platform team. "
                "https://www.linkedin.com/in/priya-raman"},
        ],
        "expectations": {
            "max_lines": 2, "max_chars": 260, "years_known": True, "expected_years": "9",
        },
    },
    {
        "id": "skills",
        "name": "Skills question",
        "question": "What are her skills?",
        "context": _DIRECTORY,
        "history": [
            {"role": "user", "content": "Who is Priya Raman?"},
            {"role": "assistant", "content":
                "Priya Raman is a Senior Data Engineer on the Platform team. "
                "https://www.linkedin.com/in/priya-raman"},
        ],
        "expectations": {
            "max_lines": 2, "max_chars": 400, "skills_requested": True,
            "link_allowed": True, "link_required": True,
        },
    },
    {
        "id": "pronoun-followup",
        "name": "Pronoun follow-up, link already given",
        "question": "Which team is he on again?",
        "context": _DIRECTORY,
        "history": [
            {"role": "user", "content": "Who is Priya Raman?"},
            {"role": "assistant", "content":
                "Priya Raman is a Senior Data Engineer on the Platform team. "
                "https://www.linkedin.com/in/priya-raman"},
        ],
        "expectations": {"max_lines": 2, "max_chars": 200},
    },
    {
        "id": "no-job-list",
        "name": "Must not volunteer a job list",
        "question": "Is Priya a good person to ask about Airflow?",
        "context": _DIRECTORY,
        "history": [],
        "expectations": {"max_lines": 2, "max_chars": 320, "jobs_requested": False, "link_allowed": True},
    },
    {
        "id": "explicit-link",
        "name": "Link explicitly requested",
        "question": "Can you send me her profile link?",
        "context": _DIRECTORY,
        "history": [
            {"role": "user", "content": "Who is Priya Raman?"},
            {"role": "assistant", "content": "Priya Raman is a Senior Data Engineer on the Platform team."},
        ],
        "expectations": {"max_lines": 2, "max_chars": 200, "link_required": True, "link_allowed": True},
    },
    {
        "id": "unknown-person",
        "name": "Person not in the directory",
        "question": "What does Marcus Feld do here?",
        "context": _DIRECTORY,
        "history": [],
        "expectations": {"max_lines": 2, "max_chars": 240},
    },
]


def starter_suite() -> Dict[str, Any]:
    """The prompt as it runs today plus the turns that exercise each of its clauses."""
    return {
        "name": "Person / employee query prompt",
        "prompt": STARTER_PROMPT,
        "cases": [json.loads(json.dumps(c)) for c in STARTER_CASES],
    }
