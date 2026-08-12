"""
Shared builders for chatbot tests.

Query understanding now resolves every word against the POOL (`chat_lexicon`) rather
than against a handed-in dictionary, because only the corpus can say whether "now" is
a technology or an ordinary word. Tests therefore need a small but real corpus rather
than a stub vocabulary — this module builds one from a compact description.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.services.ai.chat_lexicon import CorpusLexicon, build_lexicon


def make_profiles(
    names: Sequence[str],
    skills: Optional[Dict[str, Sequence[str]]] = None,
    locations: Optional[Dict[str, str]] = None,
    years: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """One profile record per name, in the shape the corpus manifest stores."""
    skills = skills or {}
    locations = locations or {}
    years = years or {}
    return [
        {
            "resume_id": i + 1,
            "name": name,
            "filename": f"{name.replace(' ', '_')}.pdf",
            "skills": list(skills.get(name, [])),
            "location": locations.get(name),
            "title": None,
            "total_years": years.get(name),
            "seniority_level": None,
            "email": None,
            "phone": None,
        }
        for i, name in enumerate(names)
    ]


def make_chunks(profiles: Sequence[Dict[str, Any]], prose: str = "") -> List[Dict[str, Any]]:
    """
    Indexed passages for those profiles.

    Skills are written capitalised, as a real CV writes a technology, so the lexicon's
    "is this an ordinary English word?" measurement sees realistic evidence. `prose`
    injects extra lowercase body text when a test needs a word to READ as English.
    """
    chunks: List[Dict[str, Any]] = []
    for profile in profiles:
        rid = profile["resume_id"]
        location = profile.get("location") or ""
        header = f"{profile['name']}\n{location} 600017" if location else profile["name"]
        chunks.append({
            "resume_id": rid, "section": "Summary", "page": 1,
            "filename": profile["filename"], "text": f"{header}\n{prose}",
        })
        if profile.get("skills"):
            chunks.append({
                "resume_id": rid, "section": "Skills", "page": 1,
                "filename": profile["filename"],
                "text": "Technical Skills: " + ", ".join(profile["skills"]),
            })
        chunks.append({
            "resume_id": rid, "section": "Experience", "page": 1,
            "filename": profile["filename"],
            "text": (
                f"Worked extensively with {', '.join(profile.get('skills') or ['software'])} "
                f"delivering production systems. {prose}"
            ),
        })
    return chunks


def make_lexicon(
    names: Sequence[str] = (),
    skills: Optional[Dict[str, Sequence[str]]] = None,
    locations: Optional[Dict[str, str]] = None,
    years: Optional[Dict[str, float]] = None,
    prose: str = "",
) -> CorpusLexicon:
    """A query-ready lexicon over a small synthetic pool."""
    profiles = make_profiles(names, skills, locations, years)
    return build_lexicon(make_chunks(profiles, prose), profiles)


def lexicon_with_skills(names: Sequence[str], vocabulary: Iterable[str]) -> CorpusLexicon:
    """Every named candidate carries the same skill list — the common test shape."""
    vocabulary = list(vocabulary)
    return make_lexicon(names, {name: vocabulary for name in names})
