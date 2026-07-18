"""
AI Recruiter — interview simulation.

Given a completed analysis, the AI plays a senior technical interviewer: for each
interview question it produces the strongest answer the candidate's *own resume*
can support, cites the evidence, rates its confidence, flags what's missing, and
gives a recruiter's verdict. This lets a recruiter see whether the resume can
genuinely justify each skill instead of inferring everything by hand.

Evidence comes from RAG (the stored evidence package); the reasoning is done by
the configured LLM (Gemini recommended). With no LLM key it falls back to a
deterministic pass built from the same evidence, so the feature always works.
"""
import json
import os
from typing import Any, Dict, List

from app.core.constants import REPORT_DIR
from app.core.exceptions import NotFoundError
from app.core.logging import logger
from app.schemas.interview import InterviewQA, InterviewSimulation
from app.services.ai import llm_service
from app.services.ai.evaluation_service import _format_evidence_for_prompt, _extract_json


def _load_report(analysis_id: int) -> Dict[str, Any]:
    path = os.path.join(REPORT_DIR, f"analysis_{analysis_id}.json")
    if not os.path.exists(path):
        raise NotFoundError(f"No stored analysis {analysis_id} to interview.")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise NotFoundError(f"Analysis {analysis_id} report could not be read: {e}")


# --- LLM path ---------------------------------------------------------------

_INTERVIEW_SCHEMA = """[
  {"question": "", "ideal_answer": "", "evidence": "", "confidence": 0,
   "missing_information": "", "follow_up_questions": [], "recruiter_evaluation": ""}
]"""


def _build_interview_prompt(evidence_block: str, questions: List[str]) -> str:
    q_list = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return (
        "You are a senior technical interviewer assessing ONE candidate. You are given "
        "the evidence retrieved from their resume and a list of interview questions.\n\n"
        "For EACH question, answer it the way THIS candidate could — using ONLY their "
        "resume evidence. Never invent experience that is not in the evidence; if the "
        "resume cannot support a strong answer, say so honestly and lower the confidence.\n\n"
        f"RESUME EVIDENCE:\n{evidence_block}\n\n"
        f"QUESTIONS:\n{q_list}\n\n"
        "Return ONLY a valid JSON array (no markdown, no prose) with one object per "
        f"question, in order, EXACTLY this shape:\n{_INTERVIEW_SCHEMA}\n\n"
        "Field rules:\n"
        "- ideal_answer: the strongest answer the resume supports, in the candidate's "
        "voice, grounded in the evidence. Plain, specific, no fluff.\n"
        "- evidence: quote the exact resume text that backs the answer ('' if none).\n"
        "- confidence (0-100): how well the resume supports a convincing answer.\n"
        "- missing_information: what the resume does NOT show that a strong answer would "
        "need ('' if nothing missing).\n"
        "- follow_up_questions: 1-2 sharper probes ONLY when confidence is below 60, else [].\n"
        "- recruiter_evaluation: a recruiter's short, honest verdict on how convincing "
        "this would be and whether to probe further."
    )


def _simulate_with_llm(llm, evidence_data: Dict[str, Any], questions: List[str]) -> List[InterviewQA]:
    prompt = _build_interview_prompt(_format_evidence_for_prompt(evidence_data), questions)
    response = llm.complete(prompt)
    raw = _extract_json(response.text)
    if isinstance(raw, dict):  # tolerate a single object or a wrapped list
        raw = raw.get("items") or raw.get("questions") or [raw]
    items: List[InterviewQA] = []
    for i, obj in enumerate(raw):
        items.append(InterviewQA(
            question=str(obj.get("question") or (questions[i] if i < len(questions) else "")),
            ideal_answer=str(obj.get("ideal_answer", "")).strip(),
            evidence=str(obj.get("evidence", "")).strip(),
            confidence=max(0, min(100, int(obj.get("confidence", 0) or 0))),
            missing_information=str(obj.get("missing_information", "")).strip(),
            follow_up_questions=[str(f) for f in (obj.get("follow_up_questions") or [])][:3],
            recruiter_evaluation=str(obj.get("recruiter_evaluation", "")).strip(),
        ))
    return items


# --- Deterministic fallback -------------------------------------------------

def _match_requirement(question: str, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find the requirement this question is about (its skill name appears in the text)."""
    q = question.lower()
    best = None
    for r in requirements:
        name = str(r.get("requirement", "")).lower()
        if name and name in q:
            if best is None or len(name) > len(str(best.get("requirement", ""))):
                best = r
    return best or {}


def _simulate_deterministic(report: Dict[str, Any], questions: List[str]) -> List[InterviewQA]:
    requirements = report.get("requirements", []) or []
    items: List[InterviewQA] = []
    for q in questions:
        req = _match_requirement(q, requirements)
        name = req.get("requirement", "this area")
        status = req.get("status", "Missing")
        evidence = req.get("matched_evidence", "") or ""
        conf = int(req.get("confidence", 0) or 0)

        if status == "Matched" and evidence:
            ideal = (f"The candidate can point to concrete work with {name}: \"{evidence}\". "
                     f"They should describe the problem, what they built, and the outcome.")
            missing = "Specific metrics or scale of impact are not stated in the resume."
            verdict = "The resume backs this well — expect a solid, evidence-based answer."
        elif status == "Partial":
            ideal = (f"The candidate lists {name}"
                     + (f" and can reference \"{evidence}\"" if evidence else "")
                     + ", but the resume shows little hands-on depth, so the answer may stay high-level.")
            missing = f"No project outcome or measurable result for {name} is shown."
            verdict = "Likely a shallow answer — probe for a concrete, hands-on example."
        else:
            ideal = (f"The resume shows no direct evidence of {name}, so the candidate would "
                     f"likely struggle to answer this from real experience.")
            missing = f"There is no {name} experience in the resume to draw on."
            verdict = "Treat as a real gap — the candidate probably cannot substantiate this."

        follow_ups = []
        if conf < 60:
            follow_ups = [f"Can you give one specific example where you used {name}, and what the measurable result was?"]

        items.append(InterviewQA(
            question=q, ideal_answer=ideal, evidence=evidence, confidence=conf,
            missing_information=missing, follow_up_questions=follow_ups, recruiter_evaluation=verdict,
        ))
    return items


# --- Public entrypoint ------------------------------------------------------

def simulate_interview(analysis_id: int) -> InterviewSimulation:
    """Run the AI-recruiter interview over a stored analysis. LLM if configured, else
    a deterministic pass over the same evidence."""
    report = _load_report(analysis_id)
    questions = [q for q in (report.get("interview_questions") or []) if str(q).strip()]
    if not questions:
        questions = ["Walk me through the project you are most proud of and your specific role in it."]

    evidence_data = {"retrieval_results": report.get("retrieval_results", [])}

    llm = llm_service.get_llm()
    if llm:
        try:
            items = _simulate_with_llm(llm, evidence_data, questions)
            if items:
                logger.info(f"AI Recruiter: generated {len(items)} answers via LLM for analysis {analysis_id}.")
                return InterviewSimulation(analysis_id=analysis_id, generated_by="llm", items=items)
        except Exception as e:
            logger.error(f"AI Recruiter LLM path failed for {analysis_id}: {e}. Falling back.", exc_info=True)

    items = _simulate_deterministic(report, questions)
    return InterviewSimulation(analysis_id=analysis_id, generated_by="deterministic", items=items)
