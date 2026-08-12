"""
The recruiter-facing VIEW of one embedding model's index.

Vectors and chunks live in `embedding_store`, which is shared by ranking, analysis and
the chatbot. This module turns one model's slice of that store into the object the
chatbot searches: the FAISS index, the chunk table in row order, the deterministic
per-candidate profiles, a precomputed BM25 table, and the corpus lexicon.

WHAT MOVED, AND WHY
-------------------
This module used to build and persist its own pool-wide index per engine, while
`vector_store_service` built a separate index per resume for the evaluation pipeline.
Every resume was therefore embedded twice with BGE; Analysis and Ranking were locked to
BGE because that index hardcoded 384 dimensions; and a resume uploaded while the remote
model was unreachable entered one index and not the other, with nothing to report the
gap. Adding a third model to that shape would have meant six embedding passes per
resume and three ways to silently diverge.

So parsing, chunking and vector storage moved to `embedding_store`, which holds ONE
chunk set that every model indexes. What stays here is everything downstream of the
vectors — the profile extraction, the sparse table, and `Corpus` itself.

Still true, and still the reason this layer exists: the chatbot's question is "which of
my 600 candidates matches this need", and answering it over per-resume indexes would
cost 600 index loads and 600 BM25 tokenizations per question. One flat index plus a
precomputed BM25 table answers it with one embedding, one vector search and one sparse
scan, at a cost the recruiter cannot feel as the pool grows.

Golden Rules honoured: no hardcoded skill lists (the taxonomy + morphology rules from
the existing retrieval stack are reused) and fully deterministic.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.constants import RESUME_UPLOAD_DIR, VECTOR_STORE_DIR
from app.core.logging import logger
from app.models.database import Resume
from app.services.ai import (
    document_loader,
    embedding_engines,
    keyword_retrieval,
    profile_service,
    resume_structuring_service,
    vector_store_service,
)

# Guards rebuilds/sync against concurrent requests hitting the same process.
_LOCK = threading.RLock()

# --- Contact extraction (deterministic, no LLM) -----------------------------
# HR explicitly needs to be able to reach the candidate, so contact details are
# extracted and surfaced. Patterns describe the *grammar* of an email/phone, not a
# list of known candidates.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,5}[\s-]?\d{3,4}[\s-]?\d{0,4}")
_LOCATION_RE = re.compile(
    r"^(?:[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)*)\s*,\s*(?:[A-Z]{2,}|[A-Z][a-z]+)$"
)


def _extract_contact(text: str) -> Dict[str, Optional[str]]:
    """Pull email / phone / location from the resume header region."""
    head = "\n".join((text or "").split("\n")[:12])
    email = _EMAIL_RE.search(head)

    phone = None
    for m in _PHONE_RE.finditer(head):
        raw = m.group(0).strip(" -")
        digits = re.sub(r"\D", "", raw)
        # A real phone number has 10-13 digits; this rejects years, zip codes,
        # percentages and "31%"-style metrics that live in the same region.
        if 10 <= len(digits) <= 13:
            phone = raw
            break

    location = None
    for line in head.split("\n"):
        s = line.strip()
        # Contact lines are commonly "email | phone | City, ST".
        for part in re.split(r"\s*[|·•]\s*", s):
            part = part.strip()
            if _LOCATION_RE.match(part) and "@" not in part:
                location = part
                break
        if location:
            break

    return {
        "email": email.group(0) if email else None,
        "phone": phone,
        "location": location,
    }


# Sections whose content describes capability. The header/contact block, Summary and
# Education are excluded when harvesting unknown tokens: that text is dense with phone
# digits, email fragments, degree abbreviations and university names, none of which are
# skills (a resume from "SRM" was previously credited with the skill "SRM").
_SKILL_BEARING_SECTIONS = ("Skills", "Experience", "Projects", "Certifications")


# Ordinary English that appears in resumes IN CAPS — section headers, connectives and
# CV boilerplate. The morphology rule reads any all-caps run as an acronym, so without
# this a resume contributed "WITHIN", "ABOUT" and "ACHIEVEMENTS" to the skill
# vocabulary, and an unrecognised name could then fuzzy-match one of them.
_ENGLISH_NOT_SKILLS = {
    # CV structure
    "resume", "curriculum", "vitae", "profile", "summary", "objective", "about",
    "experience", "education", "skills", "projects", "certifications", "certificate",
    "achievements", "accomplishments", "accolades", "awards", "references", "declaration",
    "personal", "details", "contact", "address", "phone", "email", "linkedin", "github",
    "additional", "additionally", "academic", "academia", "professional", "employment",
    "history", "background", "responsibilities", "duties", "role", "roles", "company",
    "organization", "organisation", "client", "clients", "project", "team", "teams",
    "current", "present", "years", "year", "month", "months", "date", "dates",
    # connectives / prose
    "within", "without", "about", "above", "below", "under", "over", "between", "among",
    "through", "throughout", "during", "before", "after", "since", "until", "while",
    "with", "using", "used", "including", "include", "included", "such", "also", "other",
    "others", "various", "multiple", "several", "many", "more", "most", "less", "least",
    "strong", "excellent", "good", "very", "highly", "well", "able", "ability", "worked",
    "working", "work", "handling", "handled", "managed", "managing", "developed",
    "development", "designed", "design", "created", "implemented", "involved", "part",
    "based", "level", "levels", "type", "types", "system", "systems", "process",
    "processes", "support", "supported", "provide", "provided", "ensure", "ensured",
    "knowledge", "understanding", "expertise", "proficient", "familiar", "hands",
    "language", "languages", "tools", "tool", "technologies", "technology", "environment",
    "domain", "industry", "business", "customer", "customers", "user", "users",
    "requirement", "requirements", "solution", "solutions", "application", "applications",
    "college", "university", "institute", "school", "degree", "bachelor", "master",
    "confidential", "notice", "period", "location", "salary", "gender", "nationality",
    "male", "female", "married", "single", "father", "mother", "name", "surname",
}


def _looks_like_an_identifier(tok: str) -> bool:
    """
    Reject reference codes and measurements: `0AFZZ`, `8U04`, `234J`, `50CGPA`, `24X7`.

    The discriminator is simple and holds across the corpus: those tokens LEAD with a
    digit, whereas technology names that contain digits lead with a letter — k8s, s3,
    ec2, log4j, oauth2, python3, ipv6. Anything the taxonomy already knows never
    reaches this check, so a digit-leading real product name is still safe.
    """
    return bool(tok) and tok[0].isdigit() and any(ch.isalpha() for ch in tok)


# Longest run of capitals still readable as an acronym rather than a shouted word.
# AWS, SQL, API, HTML, REST, JSON, CI/CD all fit; CHENNAI, DEVELOPER, SCIENCE do not.
_MAX_ACRONYM_LEN = 5


def _is_plausible_skill_token(tok: str) -> bool:
    """
    Reject the noise that morphology alone lets through.

    A resume header is dense with tokens that *look* technical to a regex — `9876543004`,
    `2017`, `+91`, `david.pillai` — and every one of them previously surfaced as a
    "skill" on the candidate's profile. A real technology name contains letters, is not
    a date or phone fragment, and is not part of an address.
    """
    if len(tok) < 2 or len(tok) > 30:
        return False
    if not any(ch.isalpha() for ch in tok):
        return False                                   # 17, 2017, 91
    if sum(ch.isdigit() for ch in tok) >= 4:
        return False                                   # phone/id fragments (keeps k8s, s3, ec2)
    if "@" in tok or tok.lower().endswith((".com", ".in", ".org", ".net")):
        return False                                   # email / URL fragments
    if re.fullmatch(r"(?:19|20)\d{2}", tok):
        return False                                   # calendar years
    if tok.lower() in _ENGLISH_NOT_SKILLS:
        return False                                   # WITHIN, ABOUT, ACHIEVEMENTS…
    if _looks_like_an_identifier(tok):
        return False                                   # 0AFZZ, 8U04, 234J
    return True


def _skill_region(text: str) -> str:
    """
    Only the capability-bearing sections, using the same section parser the chunking
    pipeline uses — so "which part of a resume is about skills" is defined in exactly
    one place rather than re-guessed here.
    """
    try:
        sections = resume_structuring_service.parse_sections(text or "")
    except Exception:
        return text or ""
    parts = [
        entry["text"]
        for name in _SKILL_BEARING_SECTIONS
        for entry in sections.get(name, [])
    ]
    return "\n".join(parts) if parts else (text or "")


def _extract_skills(text: str) -> List[str]:
    """
    Deterministic skill surface for a resume, reusing the retrieval stack's notion of
    a "technical token" — known taxonomy terms plus morphologically-technical tokens
    (CamelCase / acronyms / symbol-bearing). No hardcoded mapping.

    Known taxonomy terms are matched over the whole document (a skill named in the
    summary is still a real skill); unknown tokens are harvested only from the
    capability-bearing sections, where a technical-looking token is much more likely to
    be an actual technology than a phone number.
    """
    from app.core.constants import TECH_TAXONOMY
    from app.services.ai.retrieval_service import _MORPH_TECH

    known = {s.lower(): s for skills in TECH_TAXONOMY.values() for s in skills}
    lowered = (text or "").lower()
    found: Dict[str, str] = {}

    for term_lc, canonical in known.items():
        if re.search(rf"(?<!\w){re.escape(term_lc)}(?!\w)", lowered):
            found[term_lc] = canonical

    region = _skill_region(text)
    for m in _MORPH_TECH.finditer(region):
        # The shared CamelCase rule matches a prefix of a longer word ("DataCorp" ->
        # "DataC"), so a match immediately followed by a letter is a truncation, not a
        # technology. Dropping those keeps company names out of the skill list.
        if m.end() < len(region) and region[m.end()].isalpha():
            continue
        tok = m.group(0).strip("+#/.")
        if tok.lower() in found or not _is_plausible_skill_token(tok):
            continue
        # An acronym is SHORT; a shouted word is long. Resumes capitalise headings,
        # cities and job titles, and the bare "run of capitals" rule read every one of
        # them as a technology — CHENNAI, DEVELOPER, SCIENCE, ACCENTURE and ABILITIES
        # all became searchable skills, so "react developer in chennai" was understood
        # as three skill requirements and matched everybody at 88% depth.
        #
        # Genuinely long acronyms (JAVASCRIPT, POSTGRESQL) are taxonomy terms and were
        # already matched over the whole document above, so nothing real is lost.
        if tok.isupper() and len(tok) > _MAX_ACRONYM_LEN and tok.lower() not in known:
            continue
        found[tok.lower()] = tok

    return sorted(found.values(), key=str.lower)


# --- Precomputed BM25 -------------------------------------------------------
class _Bm25Stats:
    """
    BM25 with the corpus-wide statistics computed **once** at load time.

    `keyword_retrieval.bm25_scores` re-tokenizes the entire corpus on every call —
    fine for one resume's ~15 chunks, wasteful for the pool's ~1500. Term frequencies,
    document lengths and idf are cached here; a query then costs one pass over the
    postings of its own terms only.
    """

    __slots__ = ("tf", "doc_len", "avgdl", "idf", "postings", "n")

    def __init__(self, docs: List[str]) -> None:
        self.tf: List[Counter] = [Counter(keyword_retrieval.tokenize(d)) for d in docs]
        self.doc_len: List[int] = [sum(c.values()) for c in self.tf]
        self.n = len(docs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

        df: Counter = Counter()
        self.postings: Dict[str, List[int]] = {}
        for i, counts in enumerate(self.tf):
            for term in counts:
                df[term] += 1
                self.postings.setdefault(term, []).append(i)

        import math

        self.idf: Dict[str, float] = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def scores_for(self, query: str, k1: float = 1.5, b: float = 0.75) -> Dict[int, float]:
        """Sparse score map {chunk_row -> bm25}; absent rows scored 0."""
        if not self.n or not self.avgdl:
            return {}
        out: Dict[int, float] = {}
        for term in set(keyword_retrieval.tokenize(query)):
            rows = self.postings.get(term)
            if not rows:
                continue
            idf = self.idf[term]
            for row in rows:
                f = self.tf[row][term]
                denom = f + k1 * (1 - b + b * self.doc_len[row] / self.avgdl)
                out[row] = out.get(row, 0.0) + idf * (f * (k1 + 1)) / denom
        return out


# --- The corpus itself ------------------------------------------------------
class Corpus:
    """Loaded, query-ready view of the whole talent pool."""

    def __init__(
        self,
        index: Optional[faiss.Index],
        chunks: List[Dict[str, Any]],
        manifest: Dict[str, Any],
        engine: Optional[Any] = None,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.manifest = manifest
        # The engine that BUILT this index. Queries must use the same one — a query
        # embedded by a different model would search a space it does not belong to.
        self.engine = engine or embedding_engines.resolve_engine("bge")
        self.revision: str = manifest.get("revision", "")
        self._bm25: Optional[_Bm25Stats] = None
        self._vocab: Optional[Dict[str, str]] = None
        self._lexicon: Optional[Any] = None

    @property
    def engine_name(self) -> str:
        return getattr(self.engine, "name", "bge")

    @property
    def bm25(self) -> _Bm25Stats:
        """Built on first sparse query, then cached for the corpus' lifetime."""
        if self._bm25 is None:
            logger.info(f"Precomputing BM25 statistics over {len(self.chunks)} corpus chunks...")
            self._bm25 = _Bm25Stats([c["text"] for c in self.chunks])
        return self._bm25

    @property
    def resumes(self) -> Dict[int, Dict[str, Any]]:
        """{resume_id -> profile record} for the indexed pool."""
        return {int(k): v for k, v in self.manifest.get("resumes", {}).items()}

    @property
    def lexicon(self) -> Any:
        """
        What this pool's own text says its words mean — skills, places and people.

        Built lazily from the chunk table and profile records that are already in
        memory, so it costs one pass over the corpus text on first use and nothing
        afterwards. Nothing is re-embedded and no extra file is written, which is why
        an existing index picks this up without a rebuild.

        See `chat_lexicon` for why a corpus-relative answer is the only correct one:
        "now" is a technology in no pool and ordinary English in every pool, and only
        the corpus can say which.
        """
        if self._lexicon is None:
            from app.services.ai.chat_lexicon import build_lexicon

            self._lexicon = build_lexicon(self.chunks, list(self.resumes.values()))
        return self._lexicon

    @property
    def skill_vocabulary(self) -> Dict[str, str]:
        """
        Terms that may act as a hard search requirement, {lowercase -> canonical}.

        Delegates to the lexicon, which admits a harvested term only when the corpus
        writes it like a technology rather than like an ordinary word.
        """
        if self._vocab is None:
            self._vocab = self.lexicon.skill_terms
            logger.info(f"Corpus skill vocabulary: {len(self._vocab)} terms.")
        return self._vocab

    def is_empty(self) -> bool:
        return self.index is None or not self.chunks

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Dense search over every resume at once -> [(chunk_row, cosine)]."""
        if self.is_empty():
            return []
        vec = self.engine.embed_query(query)   # same engine that built the index
        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return []
        sims, ids = self.index.search(vec, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0]


# One cached corpus per engine — they are different indexes, not alternatives.
_CACHE: Dict[str, Corpus] = {}


# Experience entries are conventionally written "Role | Company" / "Role at Company",
# so when a resume has no header title the current role is recoverable from the first
# experience line. Grammar of an experience heading, not a list of known titles.
_EXPERIENCE_TITLE_RE = re.compile(
    r"^\s*([A-Z][A-Za-z/&.\- ]{2,45}?)\s*(?:\||\bat\b|,)\s*[A-Z]", re.M
)


def _fallback_title(text: str) -> Optional[str]:
    """Current role taken from the first experience heading."""
    body = re.split(r"(?im)^\s*(?:professional\s+)?experience\s*$", text or "", maxsplit=1)
    region = body[1] if len(body) > 1 else (text or "")
    if (m := _EXPERIENCE_TITLE_RE.search(region)):
        title = m.group(1).strip(" -|,")
        if 2 <= len(title.split()) <= 6 or len(title) > 4:
            return title
    return None


# Words that mean the "name" we picked up is a heading or an institution, not a person.
_NOT_A_PERSON = re.compile(
    r"^(?:resume|cv|curriculum\s+vitae|profile|summary|professional\s+summary|objective|"
    r"experience|work\s+experience|professional\s+experience|education|skills|"
    r"technical\s+skills|projects?|certifications?|achievements?|declaration|"
    r"personal\s+details?|contact|about(?:\s+me)?|confidential|references?)\s*:?\s*$",
    re.I,
)
_INSTITUTION = re.compile(
    r"\b(?:college|university|institute|school|academy|polytechnic|ltd|limited|pvt|"
    r"private|inc|corporation|corp|technologies|solutions|services|systems|consultancy)\b",
    re.I,
)
# Upload pipelines prefix files with a UUID/hash: "00ca930e-1d49-...-Poravi_Resume.docx".
_FILE_ID_PREFIX = re.compile(r"^[0-9a-f]{6,}(?:-[0-9a-f]{4,}){0,4}-", re.I)
_FILE_NOISE = re.compile(
    r"\b(?:resume|resume\d*|cv|updated|final|new|latest|copy|doc|docx|pdf|profile|"
    r"enhanced|naukri|monster|indeed)\b|\(\d+\)|_+|\d{4,}",
    re.I,
)
# Role/technology words that follow a name in an upload filename, e.g.
# "Ramani priya Java Full Stack Developer" or "Shubham Kalegore Jr. Software Developer".
# Everything from the first such word onward describes the JOB, not the person.
_FILE_ROLE_MARKER = re.compile(
    r"^(?:jr|sr|junior|senior|lead|principal|staff|associate|trainee|intern|"
    r"java|python|dotnet|net|php|node|react|angular|full|stack|frontend|front|backend|"
    r"back|end|software|web|mobile|android|ios|data|cloud|devops|qa|test|automation|"
    r"business|system|systems|network|security|support|technical|developer|engineer|"
    r"analyst|architect|consultant|manager|administrator|specialist|designer|scientist|"
    r"recruiter|bench|sales|years?|yrs?|exp|experience|j2ee|sap|aws|azure)$",
    re.I,
)


def _name_from_filename(filename: str) -> Optional[str]:
    """
    Best-effort person name from an upload filename.

    Real pipelines name files `<uuid>-Firstname Lastname_Resume (1).docx`; using the
    raw stem put UUIDs on candidate cards. Strip the id prefix and the CV boilerplate,
    then keep it only if what remains reads like a name.
    """
    stem = os.path.splitext(filename or "")[0]
    stem = _FILE_ID_PREFIX.sub("", stem)
    # Separators become spaces BEFORE noise words are stripped: `_` is a word
    # character, so "Poravi_Resume" is a single token and `\bresume\b` cannot see it.
    stem = re.sub(r"[-_.]+", " ", stem)
    stem = _FILE_NOISE.sub(" ", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" -_.")

    # Keep the leading run of name-like words; stop at the first role/technology word,
    # so "Ramani priya Java Full Stack Developer" yields "Ramani Priya".
    words: List[str] = []
    for raw in stem.split():
        word = raw.strip(" -_.,")
        if not word:
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", word):
            break
        if _FILE_ROLE_MARKER.match(word.rstrip(".")):
            break
        words.append(word)
        if len(words) == 4:
            break

    if not words:
        return None
    return " ".join(w.capitalize() if (w.isupper() or w.islower()) else w for w in words)


def _clean_person_name(candidate: Optional[str]) -> Optional[str]:
    """Reject headings, institutions and id-like strings picked up as a person's name."""
    name = (candidate or "").strip(" :|-")
    if not name or len(name) > 60:
        return None
    if _NOT_A_PERSON.match(name) or _INSTITUTION.search(name):
        return None
    if any(ch.isdigit() for ch in name) or "@" in name:
        return None
    if len(name.split()) > 4:
        return None
    return name


def _profile_for(resume_id: int, path: str, text: str, filename: str) -> Dict[str, Any]:
    """Deterministic per-candidate record powering cheap structured pre-filters."""
    prof = profile_service.extract_profile(text, jd_text="")
    contact = _extract_contact(text)
    title = prof.title or _fallback_title(text)
    return {
        "resume_id": resume_id,
        "filename": filename,
        "fingerprint": vector_store_service.compute_fingerprint(path),
        # Prefer a validated name from the document; fall back to the filename with
        # its upload-id prefix stripped; only then admit we don't know.
        "name": (
            _clean_person_name(prof.name)
            or _name_from_filename(filename)
            or f"Candidate #{resume_id}"
        ),
        "title": title,
        "total_years": prof.total_years,
        "seniority_level": prof.seniority_level,
        "email": contact["email"],
        "phone": contact["phone"],
        "location": contact["location"],
        "skills": _extract_skills(text),
    }


# --- Load / sync ------------------------------------------------------------
# Everything below is a thin adapter over `embedding_store`, which owns the vectors.
#
# This module used to build and persist its OWN pool-wide index per engine, alongside
# `vector_store_service`'s per-resume indexes — so every resume was embedded twice with
# BGE, and Analysis and Ranking were locked to BGE because their index hardcoded 384
# dimensions. The chunk table, the manifest and the incremental sync all moved to the
# shared store; what remains here is the recruiter-facing *view* of it: profiles, the
# BM25 table, the corpus lexicon, and the `Corpus` object the chatbot searches.


def get_corpus(
    db: Optional[Session] = None, auto_sync: bool = False, engine_name: Optional[str] = None
) -> Corpus:
    """
    The query-ready view of one model's index.

    Deliberately does NOT embed. Indexing is explicit and per model (see
    `embedding_store.index_models_async`) because an implicit step is what let the two
    models drift apart in the first place: it succeeded for the local model, failed for
    the remote one, and reported nothing. A chat question against a model that is behind
    returns what that model actually holds, and the coverage endpoint says so plainly.

    `auto_sync` only refreshes the model-independent DOCUMENT layer (parse + chunk), and
    is off by default so no screen can trigger a long job as a side effect.
    """
    from app.services.ai import embedding_store

    if db is not None and auto_sync:
        embedding_store.sync_documents(db)

    model_index = embedding_store.load_index(engine_name)
    profiles = embedding_store.read_profiles()
    manifest = {
        "revision": f"{len(profiles)}:{len(model_index.chunks)}",
        "engine": model_index.model,
        "dimension": model_index.dimension,
        "resumes": {str(rid): profile for rid, profile in profiles.items()},
    }
    return Corpus(model_index.index, model_index.chunks, manifest, model_index.engine)


def sync_corpus(db: Session, force: bool = False, engine_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Refresh documents and (re)index ONE model, synchronously.

    Kept for the existing `/chat/corpus/sync` endpoint and for scripted use. The UI now
    drives `embedding_store.index_models_async`, which can index several models and
    reports progress instead of holding a request open for minutes.
    """
    from app.services.ai import embedding_store

    documents = embedding_store.sync_documents(db, force=force)
    engine = embedding_engines.resolve_engine(engine_name)
    result = embedding_store.index_model(engine.name)
    invalidate_cache(engine.name)
    return {
        "added": documents.added,
        "updated": documents.updated,
        "removed": documents.removed,
        "unchanged": documents.unchanged,
        "total_resumes": documents.total_resumes,
        "total_chunks": documents.total_chunks,
        "engine": engine.name,
        "embedded": result.get("embedded", 0),
        "reused": result.get("reused", 0),
    }


def invalidate_cache(engine_name: Optional[str] = None) -> None:
    """Drop the in-process cache (next access reloads from disk)."""
    from app.services.ai import embedding_store

    with _LOCK:
        embedding_store.invalidate(engine_name)


def corpus_status(db: Optional[Session] = None, engine_name: Optional[str] = None) -> Dict[str, Any]:
    """Coverage of one model's index vs the document set. Never rebuilds."""
    from app.services.ai import embedding_store

    engine = embedding_engines.resolve_engine(engine_name)
    name = engine.name
    report = embedding_store.coverage(db)
    mine = next((m for m in report["models"] if m["model"] == name), None) or {}
    requested = (engine_name or settings.EMBEDDING_ENGINE or "bge").lower()
    return {
        "indexed_resumes": mine.get("indexed_resumes", 0),
        "indexed_chunks": mine.get("indexed_chunks", 0),
        "database_resumes": report.get("database_resumes"),
        "in_sync": bool(mine.get("in_sync")) and report.get("documents_in_sync", True),
        "cached": name in embedding_store._CACHE,
        "engine": name,
        "engine_requested": requested,
        # True when the requested engine could not serve and the local one stood in.
        "engine_fell_back": requested not in ("bge", "local", "cpu", "") and name == "bge",
        "dimension": engine.dimension,
        "store": embedding_store.store_dir(),
    }


def corpus_dir(engine_name: str) -> str:
    """
    Legacy path helper.

    Each engine no longer owns a directory — vectors live in one shared store keyed by
    filename — but callers and tests still ask "where does this engine's data live",
    and the answer must stay distinct per engine.
    """
    from app.services.ai import embedding_store

    return os.path.join(embedding_store.store_dir(), f"corpus-{engine_name}")
