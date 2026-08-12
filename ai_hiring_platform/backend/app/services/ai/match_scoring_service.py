"""
Match Score — the candidate-versus-job number.

A deterministic 0-100 score built from nine weighted parameters. The AI's job is
upstream of this: it extracts structure from the documents (skills, sections,
titles, evidence). Once that structure exists, the score is arithmetic, so the same
candidate against the same job always produces the same number — on any machine, in
any run, for anyone implementing this specification.

    Match Score = Skill×20% + Experience×15% + Technology×14% + Designation×14%
                + Industry×9% + Education×8% + Location×8% + Availability×7%
                + Freshness×5%

Two rules govern the whole design:

**Silence is neutral, not failure.** When a JD does not state a location, an
industry or a joining window, that parameter scores mid-range. A requirement nobody
wrote down must never be able to sink a candidate — otherwise a terse JD would rank
everyone as a poor fit, and the score would measure JD verbosity rather than
candidate quality. Failing a *stated* requirement is different, and still scores low.

**Claimed and practised are different parameters.** Skill Match reads the whole
resume, including the skills list. Technology Match reads only Experience and
Projects. A candidate who lists Kubernetes but never used it scores on the first and
not the second, which is precisely the distinction a recruiter is trying to make.

Every parameter reports the reasoning that produced it, so a score is never a bare
number: `MatchParameter.basis` is what the UI shows when someone asks "why 78.4?".
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.logging import logger
from app.services.ai import text_matching as tm
from app.services.ai.candidate_facets_service import CandidateFacets
from app.services.ai.job_profile_extractor import JobProfile

# A parameter's outcome: its 0-100 score, the sentence explaining it, and whether
# it was decided by evidence or by the absence of any requirement to test.
Outcome = Tuple[float, str, bool]


@dataclass
class MatchParameter:
    key: str
    label: str
    weight: float
    score: float
    contribution: float
    basis: str
    neutral: bool

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "weight": self.weight,
            "score": self.score,
            "contribution": self.contribution,
            "basis": self.basis,
            "neutral": self.neutral,
        }


@dataclass
class MatchScore:
    score: float
    band: str
    parameters: List[MatchParameter]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "band": self.band,
            "parameters": [p.to_dict() for p in self.parameters],
        }


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _round1(value: float) -> float:
    return round(value + 0.0, 1)


def band_for(score: float) -> str:
    """The recruiter-facing interpretation of a Match Score."""
    for label, minimum in settings.MATCH_SCORE_BANDS:
        if score >= minimum:
            return label
    return settings.MATCH_SCORE_BANDS[-1][0]


# ---------------------------------------------------------------------------
# Skill credit
# ---------------------------------------------------------------------------

def _semantic_equivalent(requirement: str, pool: Sequence[str]) -> bool:
    """
    Whether the candidate holds a skill that *is* this requirement under another
    name — the JS ↔ JavaScript case that no amount of string comparison resolves.

    Delegated to `skill_semantics_service`, which decides equivalence by embedding
    similarity rather than a lookup table, so it generalises to names nobody has
    catalogued. Imported lazily and wrapped: the model is heavy, and a scorer that
    cannot run without it would break the "algorithms always work" guarantee.
    """
    try:
        from app.services.ai import skill_semantics_service

        _, _, relation = skill_semantics_service.find_related_skill(requirement, list(pool))
        return relation == "equivalent"
    except Exception as e:  # pragma: no cover - defensive; lexical layer still stands
        logger.debug(f"Semantic equivalence unavailable for '{requirement}': {e}")
        return False


def _skill_credit(requirement: str, pool: Sequence[str], semantic: bool) -> Tuple[float, str]:
    """
    How much credit one required skill earns against a candidate's skill pool.

    Three tiers, cheapest first — the expensive semantic check only runs when the
    string layers have already failed:

      1.0  the same skill, once spelling conventions are normalised away
           (`ReactJS`, `React.js` and `react` are one skill)
      1.0  a semantically equivalent skill under a different name
      0.6  a near-miss that is a typo rather than a different skill
      0.0  absent

    Returns (credit, matched_name).
    """
    req_norm = tm.normalise(requirement)
    if not req_norm:
        return 0.0, ""

    normalised_pool = [(tm.normalise(s), s) for s in pool]

    for norm, original in normalised_pool:
        if norm and norm == req_norm:
            return 1.0, original

    # A multi-word requirement ("spring boot") is held if the resume writes it as
    # one token or with different punctuation.
    req_joined = tm.normalise("".join(tm.tokens(requirement)))
    for norm, original in normalised_pool:
        if norm and req_joined and norm == req_joined:
            return 1.0, original

    if semantic and _semantic_equivalent(requirement, [p for _, p in normalised_pool]):
        return 1.0, requirement

    for norm, original in normalised_pool:
        if tm.is_misspelling(req_norm, norm):
            return settings.MATCH_PARTIAL_SKILL_CREDIT, original

    return 0.0, ""


def _score_skill_pool(
    required: Sequence[str], pool: Sequence[str], semantic: bool
) -> Tuple[float, int, int, List[str]]:
    """(percentage, full matches, partial matches, unmatched requirements)."""
    total = 0.0
    full = partial = 0
    missing: List[str] = []
    for req in required:
        credit, _ = _skill_credit(req, pool, semantic)
        total += credit
        if credit >= 1.0:
            full += 1
        elif credit > 0.0:
            partial += 1
        else:
            missing.append(req)
    percentage = (total / len(required)) * 100.0 if required else 0.0
    return percentage, full, partial, missing


# ---------------------------------------------------------------------------
# The nine parameters
# ---------------------------------------------------------------------------

def _skill_match(job: JobProfile, facets: CandidateFacets, semantic: bool) -> Outcome:
    required = job.required_skills
    if not required:
        return settings.MATCH_NEUTRAL_SKILL, "The job description lists no specific skills to match.", True

    pool = facets.all_skills()
    if not pool:
        return 0.0, f"No skills could be read from the resume to match against {len(required)} requirements.", False

    pct, full, partial, missing = _score_skill_pool(required, pool, semantic)
    detail = f"Matched {full} of {len(required)} required skills"
    if partial:
        detail += f", plus {partial} matched through a spelling variant"
    if missing:
        detail += f". Not found: {', '.join(missing[:4])}"
        if len(missing) > 4:
            detail += f" and {len(missing) - 4} more"
    return _clamp(pct), detail + ".", False


def _technology_match(job: JobProfile, facets: CandidateFacets, semantic: bool) -> Outcome:
    required = job.required_skills
    if not required:
        return settings.MATCH_NEUTRAL_SKILL, "The job description names no technologies to look for.", True

    practised = facets.technologies
    if not practised:
        return 0.0, "No Experience or Projects content showed any of the required technologies in use.", False

    pct, full, partial, _ = _score_skill_pool(required, practised, semantic)
    detail = (
        f"{full} of {len(required)} required technologies appear in real work "
        f"(Experience or Projects), not just in a skills list"
    )
    if partial:
        detail += f"; {partial} more matched through a spelling variant"
    return _clamp(pct), detail + ".", False


def _designation_match(job: JobProfile, facets: CandidateFacets) -> Outcome:
    if not job.title:
        return settings.MATCH_NEUTRAL_DESIGNATION, "The job description does not state a job title.", True
    if not facets.title:
        return settings.MATCH_NEUTRAL_DESIGNATION, "No current role title could be read from the resume.", True

    similarity = tm.phrase_similarity(job.title, facets.title)
    score = _clamp(similarity * 100.0)
    return score, (
        f"Candidate's role '{facets.title}' against the target '{job.title}' "
        f"— {round(similarity * 100)}% title overlap."
    ), False


def _experience_match(job: JobProfile, facets: CandidateFacets) -> Outcome:
    if job.min_years is None and job.max_years is None:
        return settings.MATCH_NEUTRAL_EXPERIENCE, "The job description states no experience requirement.", True
    if facets.total_years is None:
        return settings.MATCH_NEUTRAL_EXPERIENCE, "Years of experience could not be determined from the resume.", True

    years = facets.total_years
    low = job.min_years if job.min_years is not None else 0.0
    high = job.max_years if job.max_years is not None else float("inf")
    requirement = (
        f"{low:g}-{high:g} years" if job.max_years is not None else f"{low:g}+ years"
    )

    if low <= years <= high:
        return 100.0, f"{years:g} years of experience sits inside the required {requirement}.", False

    gap = (low - years) if years < low else (years - high)
    score = _clamp(100.0 - settings.MATCH_EXPERIENCE_PENALTY_PER_YEAR * gap)
    direction = "short of" if years < low else "beyond"
    return score, (
        f"{years:g} years against a requirement of {requirement} — {gap:g} "
        f"{'year' if abs(gap - 1) < 1e-9 else 'years'} {direction} the range."
    ), False


def _industry_match(job: JobProfile, facets: CandidateFacets) -> Outcome:
    if not job.industry:
        return settings.MATCH_NEUTRAL_INDUSTRY, "The job description does not name an industry or domain.", True
    if not facets.experience_text:
        return settings.MATCH_NEUTRAL_INDUSTRY, "The resume has no work history to compare the industry against.", True

    similarity = tm.phrase_similarity(job.industry, facets.experience_text)
    score = _clamp(similarity * 100.0)
    verdict = "clear overlap" if similarity >= 0.6 else "little overlap"
    return score, (
        f"Work history against the '{job.industry}' domain — {verdict} "
        f"({round(similarity * 100)}%)."
    ), False


def _education_match(job: JobProfile, facets: CandidateFacets) -> Outcome:
    if not job.education:
        return settings.MATCH_NEUTRAL_EDUCATION, "The job description states no education requirement.", True

    candidate = facets.education or facets.education_text
    if not candidate:
        return settings.MATCH_NEUTRAL_EDUCATION, "No education details could be read from the resume.", True

    similarity = tm.phrase_similarity(job.education, candidate)
    score = _clamp(similarity * 100.0)
    shown = facets.education or "the resume's education section"
    return score, (
        f"'{shown}' against the required '{job.education}' — {round(similarity * 100)}% overlap."
    ), False


def _location_match(job: JobProfile, facets: CandidateFacets) -> Outcome:
    if job.is_remote:
        return 100.0, "The role is remote, so the candidate's location does not constrain the match.", True
    if not job.location:
        return settings.MATCH_NEUTRAL_LOCATION, "The job description does not state a work location.", True
    if not facets.location:
        return settings.MATCH_NEUTRAL_LOCATION, "No location could be read from the resume.", True

    similarity = tm.phrase_similarity(job.location, facets.location)
    if similarity >= 0.85:
        return 100.0, f"Based in {facets.location}, matching the job's {job.location}.", False
    if similarity >= 0.4:
        # Partial credit shades from the different-city floor up to a full match, so
        # "Chennai, Tamil Nadu" against "Chennai" reads as near-identical while a
        # loose regional overlap reads as merely close.
        floor = settings.MATCH_LOCATION_DIFFERENT
        score = _clamp(floor + (100.0 - floor) * similarity)
        return score, f"Based in {facets.location}, near the job's {job.location}.", False

    return settings.MATCH_LOCATION_DIFFERENT, (
        f"Based in {facets.location}, a different location from the job's {job.location} "
        f"— reachable only by relocation."
    ), False


def _availability_match(job: JobProfile, facets: CandidateFacets) -> Outcome:
    if job.availability_weeks is None:
        return settings.MATCH_NEUTRAL_AVAILABILITY, "The job description states no joining deadline.", True
    if facets.availability_weeks is None:
        return settings.MATCH_MISSING_AVAILABILITY, (
            "The resume does not say when the candidate could start, and the job asks for a date."
        ), False

    delay = facets.availability_weeks - job.availability_weeks
    if delay <= 0:
        when = "immediately" if facets.availability_weeks <= 0 else f"in {facets.availability_weeks:g} weeks"
        return 100.0, f"Available {when}, within the job's joining window.", False

    score = _clamp(100.0 - settings.MATCH_AVAILABILITY_PENALTY_PER_WEEK * delay)
    return score, (
        f"Available in {facets.availability_weeks:g} weeks against a window of "
        f"{job.availability_weeks:g} — {delay:g} weeks late."
    ), False


def _freshness_match(facets: CandidateFacets, as_of: datetime) -> Outcome:
    if facets.uploaded_at is None:
        return settings.MATCH_NEUTRAL_FRESHNESS, "The resume has no recorded upload date.", True

    uploaded = facets.uploaded_at
    # Compare like with like: a naive timestamp from SQLite against an aware `as_of`
    # raises rather than returning a wrong age, so normalise before subtracting.
    if uploaded.tzinfo is not None and as_of.tzinfo is None:
        uploaded = uploaded.replace(tzinfo=None)
    elif uploaded.tzinfo is None and as_of.tzinfo is not None:
        uploaded = uploaded.replace(tzinfo=timezone.utc)

    age_days = max(0.0, (as_of - uploaded).total_seconds() / 86400.0)
    floor = settings.MATCH_FRESHNESS_FLOOR
    span = settings.MATCH_FRESHNESS_DAYS
    score = _clamp(floor + (100.0 - floor) * max(0.0, 1.0 - age_days / span))

    if age_days < 1:
        detail = "Added today."
    elif age_days < 30:
        detail = f"Added {int(age_days)} days ago."
    else:
        detail = f"Added about {int(age_days / 30)} months ago."
    return score, detail + " Recency breaks ties between otherwise equal candidates.", False


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_SPEC: Iterable[tuple] = (
    ("skill", "Skill Match", "MATCH_WEIGHT_SKILL"),
    ("experience", "Experience Match", "MATCH_WEIGHT_EXPERIENCE"),
    ("technology", "Technology Match", "MATCH_WEIGHT_TECHNOLOGY"),
    ("designation", "Designation Match", "MATCH_WEIGHT_DESIGNATION"),
    ("industry", "Industry Match", "MATCH_WEIGHT_INDUSTRY"),
    ("education", "Education Match", "MATCH_WEIGHT_EDUCATION"),
    ("location", "Location Match", "MATCH_WEIGHT_LOCATION"),
    ("availability", "Availability Match", "MATCH_WEIGHT_AVAILABILITY"),
    ("freshness", "Resume Freshness", "MATCH_WEIGHT_FRESHNESS"),
)


def compute_match_score(
    job: JobProfile,
    facets: CandidateFacets,
    as_of: Optional[datetime] = None,
    semantic: bool = True,
) -> MatchScore:
    """
    Score one candidate against one job across all nine parameters.

    `as_of` fixes the clock for the freshness parameter — passing it explicitly is
    what makes a run reproducible, and what lets a test assert an exact number.
    `semantic=False` disables the embedding-backed equivalence check, leaving pure
    string matching; the score stays defined and deterministic without a model.
    """
    now = as_of or datetime.utcnow()

    outcomes = {
        "skill": _skill_match(job, facets, semantic),
        "experience": _experience_match(job, facets),
        "technology": _technology_match(job, facets, semantic),
        "designation": _designation_match(job, facets),
        "industry": _industry_match(job, facets),
        "education": _education_match(job, facets),
        "location": _location_match(job, facets),
        "availability": _availability_match(job, facets),
        "freshness": _freshness_match(facets, now),
    }

    parameters: List[MatchParameter] = []
    total = 0.0
    for key, label, weight_attr in _SPEC:
        score, basis, neutral = outcomes[key]
        weight = getattr(settings, weight_attr)
        contribution = score * weight
        total += contribution
        parameters.append(
            MatchParameter(
                key=key,
                label=label,
                weight=weight,
                score=_round1(_clamp(score)),
                contribution=_round1(contribution),
                basis=basis,
                neutral=neutral,
            )
        )

    final = _round1(_clamp(total))
    return MatchScore(score=final, band=band_for(final), parameters=parameters)
