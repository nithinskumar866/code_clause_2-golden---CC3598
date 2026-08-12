"""
Structured job profile — the JD side of the Match Score.

The requirement extractor answers "which skills does this role need". Six of the
nine Match Score parameters need something it never produced: the job's *title*,
*location*, *industry*, *education bar*, *experience range* and *joining window*.
This module derives those, deterministically, from the same parsed JD text.

Two passes, in this order:

  1. labelled fields — `Location: Chennai`, `Experience: 3-5 years`. Almost every
     real JD writes at least some of these, and a label is unambiguous evidence.
  2. free-text patterns — `we are hiring a Senior Backend Engineer`, `5+ years`,
     `immediate joiners preferred`. Weaker, so it only fills what pass 1 left blank.

Anything neither pass finds stays `None`, and `match_scoring_service` scores that
parameter neutrally. That asymmetry is deliberate and load-bearing: a JD that never
mentions relocation must not be allowed to punish a candidate for living somewhere.
Guessing would be worse than abstaining, because a wrong guess is indistinguishable
from a real requirement once it reaches the score.

Fully deterministic — no LLM, no embeddings, no clock. The same JD always yields
the same profile.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.logging import logger
from app.services.ai import text_matching as tm

# --- Field labels -----------------------------------------------------------
# A JD writes its structured facts as "Label: value", in a heading, or in a bullet.
# One pattern per facet; the alternatives are wording variants of the same label,
# not a knowledge base about jobs.


def _label(*words: str) -> re.Pattern:
    """`Label:` / `Label -` / `Label –` at the start of a line, case-insensitive."""
    alternatives = "|".join(words)
    return re.compile(rf"^\s*(?:{alternatives})\s*[:\-–]\s*(.+)$", re.I | re.M)


_TITLE_LABEL = _label("job title", "position", "role", "designation", "job role", "title")
_LOCATION_LABEL = _label("location", "work location", "job location", "place", "based in", "base location")
_INDUSTRY_LABEL = _label("industry", "domain", "sector", "vertical", "business unit")
_EDUCATION_LABEL = _label("education", "qualification", "qualifications", "educational qualification",
                          "academic qualification", "degree")
_EXPERIENCE_LABEL = _label("experience", "work experience", "years of experience", "exp")
_AVAILABILITY_LABEL = _label("availability", "notice period", "joining", "start date",
                             "joining date", "date of joining")

# --- Free-text patterns -----------------------------------------------------

# "3-5 years", "3 to 5 years" — an explicit band.
_YEARS_RANGE = re.compile(r"(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I)
# "5+ years", "minimum 5 years", "at least 5 years" — an open-ended floor.
_YEARS_MIN = re.compile(
    r"(?:minimum|min|at least|atleast|over|more than)?\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I
)

# Remote / hybrid / onsite are locations in their own right, and the spec scores a
# remote role as a full match regardless of where the candidate lives.
_REMOTE = re.compile(r"\b(?:fully\s+)?remote\b|\bwork\s+from\s+home\b|\bwfh\b|\banywhere\b", re.I)
_HYBRID = re.compile(r"\bhybrid\b", re.I)

# "immediate joiner", "notice period of 30 days", "join within 2 weeks".
_IMMEDIATE = re.compile(r"\bimmediate(?:ly)?\b|\bASAP\b|\bright away\b", re.I)
_JOIN_WINDOW = re.compile(
    r"(?:within|in|of|upto|up to|maximum|max)?\s*(\d{1,3})\s*(day|days|week|weeks|month|months)", re.I
)

# A phrase that names an industry in prose: "in the fintech domain", "healthcare sector".
_INDUSTRY_PHRASE = re.compile(
    r"\b([A-Za-z][A-Za-z&/ ]{2,30}?)\s+(?:industry|domain|sector|vertical)\b", re.I
)

# Role-title grammar, shared in spirit with profile_service._ROLE: the nouns job
# titles are built from, not a list of jobs.
_ROLE_NOUN = re.compile(
    r"\b(engineer|developer|programmer|architect|analyst|manager|designer|scientist|"
    r"administrator|consultant|specialist|lead|director|intern|devops|sre|tester|"
    r"researcher|scrum master|product owner|technician|executive|associate|officer)\b",
    re.I,
)

_HIRING_PHRASE = re.compile(
    r"(?:hiring|looking for|seeking|searching for|recruiting|position of|role of|"
    r"opening for|we need)\s+(?:an?\s+|the\s+)?([A-Za-z][A-Za-z0-9+#./ -]{2,60})", re.I
)

_NOISE_TAIL = re.compile(r"\s*(?:who|that|with|to|for|at|in|having|and)\b.*$", re.I)


def _weeks_from(quantity: int, unit: str) -> float:
    """Normalise a joining window to weeks, so day/week/month notices compare."""
    u = unit.lower().rstrip("s")
    if u == "day":
        return quantity / 7.0
    if u == "month":
        return quantity * 4.345
    return float(quantity)


def _first(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text or "")
    return m.group(1).strip() if m else None


def _clean(value: Optional[str], limit: int = 80) -> Optional[str]:
    """Trim a captured field to one tidy phrase."""
    if not value:
        return None
    v = " ".join(value.split())
    v = re.split(r"\s*[|•·]\s*", v)[0]
    v = v.rstrip(".,;:")
    return v[:limit].strip() or None


@dataclass
class JobProfile:
    """What the JD asks for, beyond its list of skills. `None` means 'not stated'."""

    title: Optional[str] = None
    location: Optional[str] = None
    is_remote: bool = False
    industry: Optional[str] = None
    education: Optional[str] = None
    min_years: Optional[float] = None
    max_years: Optional[float] = None
    # Weeks the employer is willing to wait. 0.0 means an immediate joiner is wanted.
    availability_weeks: Optional[float] = None
    # Skills the JD requires, supplied by the requirement extractor.
    required_skills: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "location": self.location,
            "is_remote": self.is_remote,
            "industry": self.industry,
            "education": self.education,
            "min_years": self.min_years,
            "max_years": self.max_years,
            "availability_weeks": self.availability_weeks,
            "required_skills": list(self.required_skills),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "JobProfile":
        """Rebuild a profile persisted in an evidence report."""
        d = data or {}
        return cls(
            title=d.get("title"),
            location=d.get("location"),
            is_remote=bool(d.get("is_remote")),
            industry=d.get("industry"),
            education=d.get("education"),
            min_years=d.get("min_years"),
            max_years=d.get("max_years"),
            availability_weeks=d.get("availability_weeks"),
            required_skills=list(d.get("required_skills") or []),
        )


def _extract_title(text: str) -> Optional[str]:
    """The designation the JD is hiring for."""
    labelled = _clean(_first(_TITLE_LABEL, text))
    if labelled:
        return labelled

    # "We are hiring a Senior Backend Engineer to join..." — keep the title, drop the
    # subordinate clause that follows it.
    for m in _HIRING_PHRASE.finditer(text or ""):
        candidate = _clean(_NOISE_TAIL.sub("", m.group(1)))
        if candidate and _ROLE_NOUN.search(candidate):
            return candidate

    # Otherwise the first short line near the top that reads like a title — a JD
    # almost always opens with the role name.
    for line in [ln.strip() for ln in (text or "").split("\n") if ln.strip()][:8]:
        if _ROLE_NOUN.search(line) and 1 < len(line.split()) <= 7 and ":" not in line:
            return _clean(line)
    return None


def _extract_location(text: str) -> tuple:
    """(location, is_remote). A remote role needs no place to be a full match."""
    body = text or ""
    labelled = _clean(_first(_LOCATION_LABEL, body))
    if labelled:
        remote = bool(_REMOTE.search(labelled))
        if remote and not re.sub(_REMOTE, "", labelled).strip(" ,-"):
            return None, True
        return labelled, remote or bool(_REMOTE.search(body))
    if _REMOTE.search(body):
        return None, True
    if _HYBRID.search(body):
        # Hybrid still implies an office, but the JD did not name it here.
        return None, False
    return None, False


def _extract_industry(text: str) -> Optional[str]:
    labelled = _clean(_first(_INDUSTRY_LABEL, text))
    if labelled:
        return labelled
    m = _INDUSTRY_PHRASE.search(text or "")
    if m:
        phrase = _clean(m.group(1))
        # "the industry" / "our domain" carry no information.
        if phrase and len(phrase.split()) <= 4 and phrase.lower() not in {"the", "our", "this", "your"}:
            return phrase
    return None


def _extract_education(text: str) -> Optional[str]:
    labelled = _clean(_first(_EDUCATION_LABEL, text))
    if labelled:
        return labelled
    for line in (text or "").split("\n"):
        if tm.mentions_a_degree(line) and len(line.split()) <= 20:
            return _clean(line)
    return None


def _extract_years(text: str) -> tuple:
    """(min_years, max_years). A band wins over a bare floor."""
    scope = _first(_EXPERIENCE_LABEL, text) or text or ""
    m = _YEARS_RANGE.search(scope) or _YEARS_RANGE.search(text or "")
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    m = _YEARS_MIN.search(scope) or _YEARS_MIN.search(text or "")
    if m:
        return (float(m.group(1)), None)
    return (None, None)


def _extract_availability(text: str) -> Optional[float]:
    """How long the employer will wait, in weeks. 0.0 = immediate."""
    scope = _first(_AVAILABILITY_LABEL, text)
    if scope:
        if _IMMEDIATE.search(scope):
            return 0.0
        m = _JOIN_WINDOW.search(scope)
        if m:
            return _weeks_from(int(m.group(1)), m.group(2))

    body = text or ""
    # Only read a window from prose when it sits next to joining language, so
    # "5 years experience" and "30 days paid leave" are never mistaken for a notice.
    for m in re.finditer(r"[^.\n]*\b(?:notice|joining|join|onboard|start)\b[^.\n]*", body, re.I):
        sentence = m.group(0)
        if _IMMEDIATE.search(sentence):
            return 0.0
        w = _JOIN_WINDOW.search(sentence)
        if w:
            return _weeks_from(int(w.group(1)), w.group(2))
    return None


def extract_job_profile(jd_text: str, required_skills: Optional[List[str]] = None) -> JobProfile:
    """
    Build the structured job profile. Never raises: a JD that defeats every pattern
    yields an all-`None` profile, which scores neutrally rather than failing the run.
    """
    try:
        location, is_remote = _extract_location(jd_text)
        min_years, max_years = _extract_years(jd_text)
        return JobProfile(
            title=_extract_title(jd_text),
            location=location,
            is_remote=is_remote,
            industry=_extract_industry(jd_text),
            education=_extract_education(jd_text),
            min_years=min_years,
            max_years=max_years,
            availability_weeks=_extract_availability(jd_text),
            required_skills=list(required_skills or []),
        )
    except Exception as e:
        logger.error(f"Job profile extraction failed: {e}", exc_info=True)
        return JobProfile(required_skills=list(required_skills or []))
