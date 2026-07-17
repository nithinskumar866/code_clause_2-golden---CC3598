"""
Candidate profile service — deterministic identity + seniority extraction.

Answers "who is this candidate and are they senior enough for THIS role", which
pure skill-fit ignores. Fully deterministic (regex/heuristics over the parsed resume
text + the JD's stated experience) — no hardcoded names/titles, no LLM. This is
resume *understanding*, so it lives on the evidence side (Agent 1), consistent with
the deterministic-parsing the pipeline already does; the LLM never sees the raw doc.
"""
import re
from typing import List, Optional
from app.schemas.analysis import CandidateProfile
from app.core.logging import logger

# Role/title vocabulary (grammar of job titles, not a fixed candidate list).
_ROLE = (r"(engineer|developer|programmer|architect|analyst|manager|designer|scientist|"
         r"administrator|consultant|specialist|lead|director|intern|devops|sre|"
         r"researcher|scrum master|product owner|technician)")

# Seniority bands by estimated years of experience.
_SENIORITY_BANDS = [(8.0, "Lead"), (5.0, "Senior"), (2.0, "Mid"), (0.0, "Junior")]

_YEARS_PHRASE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", re.I)
_FOUR_DIGIT_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _clean_lines(text: str, limit: int = 8) -> List[str]:
    """First meaningful lines (skip page markers / blanks)."""
    out: List[str] = []
    for ln in (text or "").split("\n"):
        s = ln.strip()
        if not s or re.match(r"^---\s*PAGE", s, re.I):
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _extract_name(lines: List[str]) -> Optional[str]:
    """Heuristic: a short, all-alphabetic line at the very top that is not a role title
    or contact detail — how virtually every resume opens."""
    for s in lines[:3]:
        if "@" in s or re.search(r"\d", s) or "http" in s.lower():
            continue
        words = s.split()
        if 1 <= len(words) <= 4 and all(re.match(r"^[A-Za-z.'’-]+$", w) for w in words):
            if not re.search(_ROLE, s, re.I):
                return s
    return None


def _extract_title(lines: List[str]) -> Optional[str]:
    """First short line near the top that reads like a job title."""
    for s in lines[:6]:
        if re.search(_ROLE, s, re.I) and len(s.split()) <= 6:
            return s.strip()
    return None


def _extract_total_years(text: str) -> Optional[float]:
    """Prefer an explicit 'N years' claim; else infer a span from 4-digit years."""
    phrases = [float(m.group(1)) for m in _YEARS_PHRASE.finditer(text or "")]
    if phrases:
        return max(phrases)
    years = [int(y) for y in _FOUR_DIGIT_YEAR.findall(text or "")]
    if years:
        span = max(years) - min(years)
        if 0 < span <= 50:
            return float(span)
    return None


def _seniority_level(total_years: Optional[float]) -> Optional[str]:
    if total_years is None:
        return None
    for threshold, label in _SENIORITY_BANDS:
        if total_years >= threshold:
            return label
    return "Junior"


def _required_years(jd_text: str) -> Optional[float]:
    phrases = [float(m.group(1)) for m in _YEARS_PHRASE.finditer(jd_text or "")]
    return max(phrases) if phrases else None


def _seniority_fit(total: Optional[float], required: Optional[float]) -> str:
    if total is None or required is None:
        return "Unknown"
    if total >= required + 2:
        return "Exceeds"
    if total >= required:
        return "Meets"
    return "Below"


def _explanation(p_name, total, required, level, fit) -> str:
    if total is None:
        return "Could not determine years of experience from the resume."
    yrs = f"~{total:g} year(s) of experience ({level})"
    if fit == "Unknown":
        return f"{yrs}; the JD does not state a required experience level."
    return (f"{yrs} vs the JD's {required:g}+ year requirement — "
            f"candidate {fit.lower()} the stated seniority bar.")


def extract_profile(resume_text: str, jd_text: str = "") -> CandidateProfile:
    """
    Build a deterministic candidate profile from the resume text and the JD's stated
    experience requirement. Never raises — returns a best-effort profile.
    """
    try:
        lines = _clean_lines(resume_text)
        name = _extract_name(lines)
        title = _extract_title(lines)
        total = _extract_total_years(resume_text)
        required = _required_years(jd_text)
        level = _seniority_level(total)
        fit = _seniority_fit(total, required)
        return CandidateProfile(
            name=name, title=title, total_years=total, seniority_level=level,
            required_years=required, seniority_fit=fit,
            explanation=_explanation(name, total, required, level, fit),
        )
    except Exception as e:
        logger.error(f"Profile extraction failed: {e}", exc_info=True)
        return CandidateProfile(explanation="Profile could not be derived.")
