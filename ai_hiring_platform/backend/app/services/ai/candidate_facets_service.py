"""
Candidate facets — the resume side of the Match Score.

`profile_service` answers "who is this and how senior are they". The Match Score
needs more: which skills are *claimed* versus which are *practised*, where the
person lives, what they studied, which industry they have worked in, and how soon
they could start.

The distinction this module exists to draw is between the Skills section and the
rest of the resume. A skills list is a claim; a line in Experience or Projects is
evidence. Match Score treats those as two different parameters — Skill Match reads
the claim, Technology Match reads the evidence — and that separation is only
possible because the resume is parsed section by section first.

Deterministic throughout. Section detection is reused from
`resume_structuring_service`, so the sections scored here are exactly the sections
retrieval indexed; deriving them a second way would let the score and the evidence
trail disagree about what the resume says.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from app.core.logging import logger
from app.services.ai import text_matching as tm
from app.services.ai.resume_structuring_service import parse_sections
from app.services.ai.retrieval_service import _KNOWN_TECH

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./-]*")

# Sections whose content is a demonstration rather than a claim.
NARRATIVE_SECTIONS = ("Experience", "Projects")

# --- Location ---------------------------------------------------------------
# A resume states its address in one of two recognisable ways: behind a label, or
# as a line carrying a postal code. Nothing else is trusted, because a bare city
# name is indistinguishable from a client name, a university town, or — the bug
# this guard exists for — a technology sitting in a pipe-separated skills line.
_ADDRESS_LABEL = re.compile(
    r"\b(?:address|location|based (?:in|at)|residing (?:in|at)|city|hometown|current location)\b\s*[:\-–]\s*",
    re.I,
)
_POSTAL = re.compile(r"\b\d{5,6}\b")
_POSTAL_TAIL = re.compile(r"[\s,-]+\d{5,6}\s*$")
_CONTACT_NOISE = re.compile(r"[\w.+-]+@[\w.-]+|\+?\d[\d\s()-]{7,}|https?://\S+", re.I)

# --- Availability -----------------------------------------------------------
_IMMEDIATE = re.compile(r"\bimmediate(?:ly)?\s*(?:joiner|joining|available)?\b|\bavailable\s+now\b|\bASAP\b", re.I)
_NOTICE = re.compile(
    r"\b(?:notice\s*period|notice|availability|available|can\s+join|joining)\b[^.\n]{0,40}?"
    r"(\d{1,3})\s*(day|days|week|weeks|month|months)",
    re.I,
)
_SERVING = re.compile(r"\bserving\s+notice\b", re.I)



def technical_terms(text: str) -> Set[str]:
    """
    Technical vocabulary in a piece of text.

    Two layers: taxonomy terms matched whole-word (so `PyTorch` is never found
    inside another word), plus any word whose *shape* is technical — an acronym,
    internal capitals, a digit, or tech punctuation. The second layer is what lets
    a skill nobody has catalogued still be recognised, keeping Golden Rule 4 intact.

    This is the single definition of "a technical token" for the platform; both the
    facet extractor and the evidence-side skill recovery read it.
    """
    lowered = (text or "").lower()
    found = {t for t in _KNOWN_TECH if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", lowered)}
    for m in _WORD_RE.finditer(text or ""):
        core = m.group(0).strip("+#/.")
        if len(core) < 2:
            continue
        techy = (
            any(ch.isdigit() for ch in core)
            or any(ch in "+#/." for ch in core)
            or core.isupper()                 # acronym: SQL, AWS, API
            or core[1:] != core[1:].lower()   # internal caps: PyTorch, FastAPI
        )
        if techy:
            found.add(core.lower())
    return found


# Inside a Skills section, a list separator marks the boundary between one claimed
# skill and the next. Everything before a colon on such a line is a group heading
# ("Languages:", "Cloud:") rather than a skill.
_SKILL_SEPARATOR = re.compile(r"\s*[,;|/•·]\s*|\s{3,}")
_GROUP_HEADING = re.compile(r"^[^:]{0,30}:\s*")
# Filler that appears in skills lists without naming a skill.
_SKILL_NOISE = {
    "and", "etc", "others", "other", "various", "including", "such as", "skills",
    "technologies", "tools", "expertise", "proficient", "familiar", "knowledge",
    "experience", "basic", "advanced", "intermediate", "strong", "good", "working",
}


def declared_skills(skills_text: str) -> Set[str]:
    """
    Skills claimed in the Skills section, taken from the section's *structure*
    rather than from how the words look.

    `technical_terms` recognises a skill by its shape — an acronym, internal
    capitals, a digit. That is the right rule for prose, where a technology has to
    announce itself among ordinary English. It is the wrong rule for a skills list,
    where every comma-separated item is a skill by construction. It also has a
    specific failure that matters here: a misspelling like `Kubernets` has none of
    the technical shapes, so it was never extracted, and the Match Score's
    partial-credit-for-typos rule could never fire on the one place typos live.

    Reading the list as a list fixes both.
    """
    found: Set[str] = set()
    for raw in (skills_text or "").split("\n"):
        line = _GROUP_HEADING.sub("", raw.strip())
        for part in _SKILL_SEPARATOR.split(line):
            item = part.strip().strip(".-–—•").strip()
            low = item.lower()
            if not item or low in _SKILL_NOISE:
                continue
            # A skill name is short. Anything longer is a sentence that happened to
            # land in this section.
            if not (1 <= len(item.split()) <= 3) or len(item) > 30:
                continue
            if not re.search(r"[A-Za-z]", item):
                continue
            found.add(low)
    return found


def _section_text(sections: Dict[str, List[Dict[str, Any]]], *names: str) -> str:
    """Flatten one or more parsed sections back into plain text."""
    out: List[str] = []
    for name in names:
        for item in sections.get(name, []) or []:
            line = (item.get("text") or "").strip()
            if line:
                out.append(line)
    return "\n".join(out)


def _weeks_from(quantity: int, unit: str) -> float:
    u = unit.lower().rstrip("s")
    if u == "day":
        return quantity / 7.0
    if u == "month":
        return quantity * 4.345
    return float(quantity)


def _extract_location(text: str, skill_terms: Set[str]) -> Optional[str]:
    """
    The candidate's stated place of residence, or None.

    `skill_terms` are the technologies this resume claims; a candidate location is
    never allowed to be one of them. Without that guard a line reading
    `Languages: Java, Python, SQL | Chennai 600017` donates "Java" as a city, and
    every location-filtered search then scores on residence in a programming
    language.
    """
    for raw in (text or "").split("\n"):
        line = _CONTACT_NOISE.sub(" ", raw).strip()
        if not line:
            continue

        region = ""
        if (m := _ADDRESS_LABEL.search(line)):
            region = line[m.end():]
        elif _POSTAL.search(line):
            # Narrow to the pipe-segment that actually carries the postal code, then
            # to the last few comma-parts of it. The rest of the line is not an address.
            segments = [s for s in re.split(r"\s*[|·•]\s*", line) if _POSTAL.search(s)]
            if segments:
                region = ", ".join(re.split(r"\s*,\s*", segments[-1])[-3:])

        region = _POSTAL_TAIL.sub("", " ".join(region.split())).strip(" ,-|")
        if not region or len(region) > 60:
            continue
        # Reject anything that is really a technology list wearing an address label.
        parts = [p.strip() for p in re.split(r"\s*,\s*", region) if p.strip()]
        if not parts or any(p.lower() in skill_terms for p in parts):
            continue
        if not re.search(r"[A-Za-z]{3}", region):
            continue
        return region
    return None


def _extract_availability(text: str) -> Optional[float]:
    """Weeks before the candidate could start, or None when the resume is silent."""
    body = text or ""
    if _IMMEDIATE.search(body):
        return 0.0
    m = _NOTICE.search(body)
    if m:
        return _weeks_from(int(m.group(1)), m.group(2))
    if _SERVING.search(body):
        # "Serving notice" without a number: they are leaving, but we cannot say when.
        return None
    return None


def _extract_education(sections: Dict[str, List[Dict[str, Any]]], text: str) -> Optional[str]:
    """The candidate's highest-signal qualification line."""
    scope = _section_text(sections, "Education") or (text or "")
    for line in scope.split("\n"):
        if tm.mentions_a_degree(line) and len(line.split()) <= 25:
            cleaned = " ".join(line.split()).strip(" .,;:")
            if cleaned:
                return cleaned[:120]
    return None


@dataclass
class CandidateFacets:
    """What the resume shows, split by how strongly it shows it."""

    title: Optional[str] = None
    location: Optional[str] = None
    education: Optional[str] = None
    total_years: Optional[float] = None
    availability_weeks: Optional[float] = None
    uploaded_at: Optional[datetime] = None

    # Claimed in the Skills section — an assertion.
    primary_skills: List[str] = field(default_factory=list)
    # Mentioned anywhere else — supporting, weaker.
    secondary_skills: List[str] = field(default_factory=list)
    # Present in Experience or Projects — the technology stack actually practised.
    technologies: List[str] = field(default_factory=list)

    # Free text the industry and education comparisons read against.
    experience_text: str = ""
    education_text: str = ""

    def all_skills(self) -> List[str]:
        seen: Dict[str, None] = {}
        for s in self.primary_skills + self.technologies + self.secondary_skills:
            seen.setdefault(s, None)
        return list(seen)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "location": self.location,
            "education": self.education,
            "total_years": self.total_years,
            "availability_weeks": self.availability_weeks,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "primary_skills": list(self.primary_skills),
            "secondary_skills": list(self.secondary_skills),
            "technologies": list(self.technologies),
            "experience_text": self.experience_text,
            "education_text": self.education_text,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "CandidateFacets":
        """Rebuild facets persisted in an evidence report."""
        d = data or {}
        uploaded = d.get("uploaded_at")
        if isinstance(uploaded, str):
            try:
                uploaded = datetime.fromisoformat(uploaded)
            except ValueError:
                uploaded = None
        return cls(
            title=d.get("title"),
            location=d.get("location"),
            education=d.get("education"),
            total_years=d.get("total_years"),
            availability_weeks=d.get("availability_weeks"),
            uploaded_at=uploaded if isinstance(uploaded, datetime) else None,
            primary_skills=list(d.get("primary_skills") or []),
            secondary_skills=list(d.get("secondary_skills") or []),
            technologies=list(d.get("technologies") or []),
            experience_text=d.get("experience_text") or "",
            education_text=d.get("education_text") or "",
        )


def extract_candidate_facets(
    resume_text: str,
    title: Optional[str] = None,
    total_years: Optional[float] = None,
    uploaded_at: Optional[datetime] = None,
) -> CandidateFacets:
    """
    Derive the candidate facets the Match Score needs.

    `title` and `total_years` are passed in rather than recomputed: `profile_service`
    already derived them upstream, and two independent derivations of the same fact
    would eventually disagree with each other in front of a recruiter.

    Never raises — a resume that defeats every pattern yields empty facets, and each
    affected parameter then scores neutrally instead of failing the analysis.
    """
    try:
        sections = parse_sections(resume_text or "")

        skills_text = _section_text(sections, "Skills")
        narrative_text = _section_text(sections, *NARRATIVE_SECTIONS)
        education_text = _section_text(sections, "Education")
        other_text = _section_text(sections, "Summary", "Certifications")

        # The Skills section is read both ways: by shape (catches technologies named
        # mid-sentence) and as a list (catches everything the author declared,
        # including names too plain or too misspelt to look technical).
        primary = technical_terms(skills_text) | declared_skills(skills_text)
        technologies = technical_terms(narrative_text)
        secondary = technical_terms(other_text) - primary - technologies

        all_terms = primary | technologies | secondary
        return CandidateFacets(
            title=title,
            location=_extract_location(resume_text, all_terms),
            education=_extract_education(sections, resume_text),
            total_years=total_years,
            availability_weeks=_extract_availability(resume_text),
            uploaded_at=uploaded_at,
            primary_skills=sorted(primary),
            secondary_skills=sorted(secondary),
            technologies=sorted(technologies),
            experience_text=narrative_text,
            education_text=education_text or (resume_text or "")[:2000],
        )
    except Exception as e:
        logger.error(f"Candidate facet extraction failed: {e}", exc_info=True)
        return CandidateFacets(title=title, total_years=total_years, uploaded_at=uploaded_at)
