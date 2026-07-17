"""
Authenticity service — deterministic keyword-stuffing / over-claim detection.

Answers a question the recruiter vision demands ("is the candidate likely
exaggerating or overstating experience?") without any LLM involvement. It is a
pure algorithm over the evidence already gathered by the Candidate Intelligence
Agent: a skill only counts as *demonstrated* when it appears in a narrative
section (Experience / Projects); a skill that surfaces only in listing sections
(Skills / Summary) is *over-claimed* — the signature of keyword stuffing.

Because it reads only retrieved-evidence metadata (section + chunk text), the
result is reproducible and traceable — Requirement -> Evidence -> Verdict — and
never inspects the raw document. This also replaces the previously hardcoded
``quality_score = 85`` with a real, defensible resume-quality signal.
"""
from dataclasses import dataclass
from typing import Any, Dict, List

from app.core.config import settings
from app.schemas.analysis import AuthenticityAssessment

# Sections where a skill is actually *shown* being used, as opposed to merely
# listed. Mirrors the section taxonomy produced by resume_structuring_service.
NARRATIVE_SECTIONS = {"Experience", "Projects"}


@dataclass
class AuthenticityResult:
    """Bundles the report-facing assessment with the derived quality sub-score."""
    assessment: AuthenticityAssessment
    quality_score: int


def _density(chunk: str) -> int:
    """Detail-density of an evidence chunk (0-100), using the same length bands
    as retrieval_service so 'quality' aligns with how evidence is otherwise ranked."""
    length = len(chunk or "")
    if length > 150:
        return 100
    if length > 70:
        return 70
    if length > 30:
        return 30
    return 0


def _best_narrative_density(matches: List[Dict[str, Any]]) -> int:
    """Density of the strongest Experience/Projects chunk for a demonstrated skill."""
    narrative = [m for m in matches if m.get("section") in NARRATIVE_SECTIONS]
    if not narrative:
        return 0
    best = max(narrative, key=lambda m: m.get("confidence", m.get("score", 0.0) * 100))
    return _density(best.get("chunk", ""))


def _is_demonstrated(matches: List[Dict[str, Any]]) -> bool:
    """A claim is demonstrated when at least one supporting chunk is narrative."""
    return any(m.get("section") in NARRATIVE_SECTIONS for m in matches)


def _build_explanation(claimed: int, demonstrated: int, over_claimed: List[str], risk: str) -> str:
    if claimed == 0:
        return (
            "No claimed skill had retrievable evidence, so credibility could not be "
            "established from demonstrated work."
        )
    if not over_claimed:
        return (
            f"All {claimed} claimed skills are corroborated by concrete Experience or "
            f"Project evidence — no signs of keyword stuffing (risk: {risk})."
        )
    listed = ", ".join(over_claimed)
    return (
        f"{len(over_claimed)} of {claimed} claimed skills ({listed}) appear only in "
        f"listing sections with no supporting Experience/Project evidence, indicating "
        f"possible keyword stuffing or overstated proficiency (risk: {risk})."
    )


def assess_authenticity(evidence_data: Dict[str, Any]) -> AuthenticityResult:
    """
    Assess how credibly a candidate's *claimed* skills (those with any retrieved
    evidence) are backed by narrative demonstration, and derive the resume
    quality sub-score. Runs identically on the LLM and deterministic paths.
    """
    results = evidence_data.get("retrieval_results", [])

    # A skill is "claimed" if the candidate's resume produced any evidence for it.
    # Absent skills (no matches) are Missing, not over-claimed — exclude them.
    claimed = [it for it in results if it.get("matches")]
    claimed_count = len(claimed)

    demonstrated = [it for it in claimed if _is_demonstrated(it["matches"])]
    demonstrated_count = len(demonstrated)
    over_claimed_skills = [
        it["requirement"] for it in claimed if not _is_demonstrated(it["matches"])
    ]

    corroboration_ratio = round(demonstrated_count / claimed_count, 2) if claimed_count else 0.0
    over_claimed_fraction = (len(over_claimed_skills) / claimed_count) if claimed_count else 0.0

    if claimed_count == 0:
        risk = "Low"
    elif over_claimed_fraction >= settings.STUFFING_HIGH_FRACTION:
        risk = "High"
    elif over_claimed_fraction >= settings.STUFFING_MEDIUM_FRACTION:
        risk = "Medium"
    else:
        risk = "Low"

    credibility_score = int(round(corroboration_ratio * 100))

    # Resume quality blends "are claims substantiated" with "how detailed is the
    # demonstrating evidence", so quality is distinct from raw credibility.
    if demonstrated_count:
        avg_depth = sum(_best_narrative_density(it["matches"]) for it in demonstrated) / demonstrated_count
    else:
        avg_depth = 0.0
    quality_score = int(round(
        credibility_score * settings.QUALITY_WEIGHT_CORROBORATION
        + avg_depth * settings.QUALITY_WEIGHT_DEPTH
    ))
    quality_score = min(max(quality_score, 0), 100)

    assessment = AuthenticityAssessment(
        credibility_score=credibility_score,
        keyword_stuffing_risk=risk,
        over_claimed_skills=over_claimed_skills,
        corroboration_ratio=corroboration_ratio,
        explanation=_build_explanation(claimed_count, demonstrated_count, over_claimed_skills, risk),
    )
    return AuthenticityResult(assessment=assessment, quality_score=quality_score)
