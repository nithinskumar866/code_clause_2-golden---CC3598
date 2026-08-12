"""
Pool-wide candidate retrieval, fact lookup and scoring for the recruiter chatbot.

Query understanding lives in `chat_query_understanding`; this module consumes a typed
`QueryIntent` and does the work, in passes that get progressively more expensive so
the costly ones only ever see a small set:

  1. UNDERSTAND (free)     Already done — intent, skills, places, years, people.
  2. FACET SETS (µs)       Each constraint becomes a set of resume ids over the small
                           profile records. Kept SEPARATE rather than intersected up
                           front, which is what makes the funnel explainable.
  3. RETRIEVE (one search) ONE dense search per skill across the WHOLE pool (single
                           flat FAISS index), fused with precomputed BM25 via RRF.
  4. SCORE (cheap)         Deterministic, decomposed match % — every number shown to a
                           recruiter is explainable and reproducible.

WHY THE FACET SETS STAY SEPARATE
--------------------------------
"Best candidate in T. Nagar who knows Java" is a conjunction, and the useful answer
when it returns nothing is not "no results" — it is *which half failed*:

    3 candidates live in T. Nagar. None of them evidence Java.
    12 candidates evidence Java, none in T. Nagar.

That sentence is only available if each constraint's result is measured on its own, so
`facet_sets` computes them independently and `relaxation_options` reads off the
counterfactual for dropping each one. The assistant can then ask a real question back
instead of quietly discarding a constraint and answering something else — which is how
a question about T. Nagar and Java came back with a candidate from Manjunath Nagar who
knows Ansible.

SCORING — WHY COVERAGE IS NOT BINARY
------------------------------------
Finding the word "Java" somewhere in a resume is not evidence that someone can build
with it. A skill listed under *Skills* is a claim; the same skill described in
*Experience*, demonstrated in *Projects* and backed by a *Certification* is proof. So
each requested skill earns a **depth** score from where its evidence actually sits, and
coverage is the mean depth rather than a found/not-found flag.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.services.ai import text_matching as tm
from app.services.ai.chat_query_understanding import QueryIntent, QueryKind
from app.services.ai.embedding_store import ModelIndex as Corpus

# How much a mention in each section proves. Experience is professional application;
# a Skills list is only a claim. Mirrors the JD pipeline's section importance and the
# authenticity service's "corroboration" idea.
_SECTION_PROOF: Dict[str, float] = {
    "Experience": 1.00,
    "Projects": 0.90,
    "Certifications": 0.85,
    "Summary": 0.45,
    "Education": 0.40,
    "Skills": 0.35,
}
_STRONG_SECTIONS = {"Experience", "Projects", "Certifications"}
_DEPTH_WEIGHT_BEST = 0.75
_DEPTH_WEIGHT_BREADTH = 0.25
_BREADTH_DENOMINATOR = 4  # four distinct sections is "fully corroborated"

# Chat search is exploratory — a recruiter would rather see a 65% match than an empty
# screen — so the floor sits a little below the engine's hard evidence floor.
_CHAT_FLOOR_RELAXATION = 0.10

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.-]*")


def _chat_min_similarity(corpus: Corpus) -> float:
    """
    The evidence floor for THIS corpus's embedding model.

    Thresholds are a property of the model, not of the platform: BGE-large and
    nomic-embed-text place the same semantic relationship at different cosine values,
    so a single shared constant silently over-filters whichever model it was not
    calibrated on.
    """
    base = getattr(corpus.engine, "min_similarity", settings.RETRIEVAL_MIN_SIMILARITY)
    return max(float(base) - _CHAT_FLOOR_RELAXATION, 0.30)


def section_proof(section: str) -> float:
    return _SECTION_PROOF.get(section, 0.45)


# --- Facet sets -------------------------------------------------------------
class FacetSets:
    """
    Each constraint's result, measured independently over the pool.

    Holding these apart is what lets the assistant say *why* a search came back empty,
    and offer the recruiter a specific choice instead of a shrug.
    """

    def __init__(self, corpus: Corpus, intent: QueryIntent, restrict_ids: Optional[Set[int]] = None) -> None:
        pool = set(corpus.resumes)
        self.pool = pool
        self.restricted = set(restrict_ids) & pool if restrict_ids is not None else pool

        # Years. Inferred from resume text, so allow tolerance; candidates whose years
        # could not be determined are kept — absence of evidence is not evidence of
        # absence — and simply score lower on experience fit.
        if intent.min_years is None and intent.max_years is None:
            self.by_years = set(pool)
        else:
            tolerance = 0.5
            low = (intent.min_years - tolerance) if intent.min_years is not None else None
            high = (intent.max_years + tolerance) if intent.max_years is not None else None
            self.by_years = {
                rid for rid, p in corpus.resumes.items()
                if p.get("total_years") is None
                or ((low is None or p["total_years"] >= low)
                    and (high is None or p["total_years"] <= high))
            }

        # Place. The pool is its own gazetteer (see chat_lexicon), so this is an exact
        # set lookup rather than a fuzzy text match.
        if intent.places:
            self.by_place = corpus.lexicon.resumes_at(intent.places) & pool
        else:
            self.by_place = set(pool)

        # People named in the question.
        if intent.people:
            self.by_person = {p.resume_id for p in intent.people} & pool
        else:
            self.by_person = set(pool)

        # Skills are filled in after retrieval — they need the index.
        self.by_skill: Set[int] = set(pool)
        self.by_skill_all: Set[int] = set(pool)

    @property
    def structural(self) -> Set[int]:
        """Everything that can be decided without touching the vector index."""
        return self.restricted & self.by_years & self.by_place & self.by_person

    @property
    def final(self) -> Set[int]:
        return self.structural & self.by_skill

    def counts(self, intent: QueryIntent) -> Dict[str, Any]:
        out: Dict[str, Any] = {"pool_size": len(self.pool)}
        if self.restricted != self.pool:
            out["restricted_to_previous"] = len(self.restricted)
        if intent.min_years is not None or intent.max_years is not None:
            out["meeting_experience_bar"] = len(self.by_years & self.restricted)
        if intent.places:
            out["in_requested_location"] = len(self.by_place & self.restricted)
        if intent.skills:
            out["with_any_requested_skill"] = len(self.by_skill & self.restricted)
            out["with_every_requested_skill"] = len(self.by_skill_all & self.restricted)
        out["after_prefilter"] = len(self.structural)
        return out


def relaxation_options(facets: FacetSets, intent: QueryIntent) -> List[Dict[str, Any]]:
    """
    What dropping each single constraint would yield.

    Only constraints that actually unblock the search are offered, and each carries the
    real count, so the recruiter is choosing between measured outcomes rather than
    being asked to guess. This is the mechanism behind "nobody in T. Nagar knows Java —
    do you want the T. Nagar people, or the Java people?".
    """
    if not intent.has_facets:
        return []

    options: List[Dict[str, Any]] = []

    def offer(drop: str, label: str, ids: Set[int], query: str) -> None:
        if ids:
            options.append({"drop": drop, "label": label, "count": len(ids), "query": query})

    remaining_skills = " ".join(intent.skills)
    place_label = ", ".join(intent.place_labels)

    if intent.places:
        # Keep the skill, lose the place.
        ids = facets.restricted & facets.by_years & facets.by_person & facets.by_skill
        offer(
            "location",
            f"Search anywhere, not just {place_label}",
            ids - facets.by_place if facets.by_place != facets.pool else ids,
            f"who knows {remaining_skills}".strip() or intent.raw,
        )
    if intent.skills:
        # Keep the place, lose the skill.
        ids = facets.restricted & facets.by_years & facets.by_person & facets.by_place
        offer(
            "skills",
            f"Show everyone in {place_label}" if intent.places
            else f"Drop the {remaining_skills} requirement",
            ids - facets.by_skill if facets.by_skill != facets.pool else ids,
            f"candidates in {place_label}" if intent.places else "most experienced candidates",
        )
    if intent.min_years is not None or intent.max_years is not None:
        ids = facets.restricted & facets.by_place & facets.by_person & facets.by_skill
        offer(
            "experience",
            f"Drop the {experience_bar_label(intent)} bar",
            ids - facets.by_years if facets.by_years != facets.pool else ids,
            f"who knows {remaining_skills}".strip() or intent.raw,
        )

    options.sort(key=lambda o: o["count"], reverse=True)
    return options[:3]


# --- Hybrid retrieval -------------------------------------------------------
def _rrf_rank(dense_rank: Dict[int, int], sparse_rank: Dict[int, int], rows: List[int]) -> Dict[int, float]:
    """Reciprocal Rank Fusion over chunk rows — same formula as retrieval_service."""
    k = settings.HYBRID_RRF_K
    wd, ws = settings.HYBRID_WEIGHT_DENSE, settings.HYBRID_WEIGHT_SPARSE
    big = len(rows) + k
    return {
        row: wd / (k + dense_rank.get(row, big)) + ws / (k + sparse_rank.get(row, big))
        for row in rows
    }


def _search_skill(
    corpus: Corpus, skill: str, allowed: Set[int], pool_size: int, floor: float
) -> Dict[int, List[Dict[str, Any]]]:
    """
    One hybrid search for one skill across the entire pool.

    Returns {resume_id -> [evidence chunks]}. Up to four chunks per candidate are kept
    so a skill appearing in several sections can be *seen* to be corroborated — that
    breadth is what the depth score reads.
    """
    depth = min(max(pool_size * 6, 60), 800)
    hits = corpus.search(skill, top_k=depth)
    if not hits:
        return {}

    kept = [(row, cos) for row, cos in hits
            if cos >= floor and corpus.chunks[row]["resume_id"] in allowed]
    if not kept:
        return {}

    rows = [r for r, _ in kept]
    cos_by_row = dict(kept)

    if settings.HYBRID_RETRIEVAL_ENABLED:
        bm25 = corpus.bm25.scores_for(skill)
        dense_order = sorted(rows, key=lambda r: cos_by_row[r], reverse=True)
        sparse_order = sorted(rows, key=lambda r: bm25.get(r, 0.0), reverse=True)
        fused = _rrf_rank(
            {r: i for i, r in enumerate(dense_order)},
            {r: i for i, r in enumerate(sparse_order)},
            rows,
        )
        rows.sort(key=lambda r: fused[r], reverse=True)
    else:
        rows.sort(key=lambda r: cos_by_row[r], reverse=True)

    by_resume: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        chunk = corpus.chunks[row]
        bucket = by_resume.setdefault(chunk["resume_id"], [])
        if len(bucket) >= 4:
            continue
        # A literal mention is much stronger proof than a semantic neighbour, and the
        # recruiter can see it in the quote — so record it for the depth calculation.
        literal = skill.lower() in chunk["text"].lower()
        bucket.append({
            "skill": skill,
            "text": chunk["text"],
            "section": chunk["section"],
            "page": chunk["page"],
            "filename": chunk["filename"],
            "similarity": round(cos_by_row[row], 3),
            "literal": literal,
        })
    return by_resume


# --- Scoring ----------------------------------------------------------------
def _skill_depth(evidence: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
    """
    How well one skill is substantiated, 0..1, plus the sections that proved it.

    A literal mention in Experience scores ~1.0; the same skill only listed under
    Skills scores ~0.33. Breadth across sections lifts the score toward 1.0.
    """
    if not evidence:
        return 0.0, []

    literal_sections = sorted({e["section"] for e in evidence if e.get("literal")})
    sections = literal_sections or sorted({e["section"] for e in evidence})

    best = max(section_proof(s) for s in sections)
    if not literal_sections:
        best *= 0.7  # semantic-only support is weaker evidence

    breadth = min(len(sections), _BREADTH_DENOMINATOR) / _BREADTH_DENOMINATOR
    depth = _DEPTH_WEIGHT_BEST * best + _DEPTH_WEIGHT_BREADTH * breadth
    return min(depth, 1.0), sections


def _confirms(chunk: str, skill: str) -> bool:
    """
    Whether a retrieved passage actually EVIDENCES the skill, rather than merely
    sitting near it in embedding space.

    Dense retrieval answers "what is this passage about", and for a soft requirement
    like `Communication` or `HR` the answer is "close enough" for almost every resume
    in the pool — everyone writes about working with teams. So a search for
    communication skills for an HR role returned a Mulesoft architect and a business
    analyst, each at 100%, on passages that never mention either word.

    Retrieval stays semantic; the CONFIRMATION is lexical. The skill's own words must
    appear in the passage, allowing for the spelling variants the pool actually uses
    ("Reat js" for React). Multi-word skills confirm on any of their content words, so
    "Spring Boot" is still evidenced by a line that says only "Spring".

    For role/designation terms (e.g., "recruiter", "developer", "manager"), semantic
    similarity is checked as a fallback: "HR Executive" semantically matches "recruiter"
    because both are recruitment-domain roles, working across ANY domain generically.
    """
    if not chunk or not skill:
        return False
    words = [w for w in tm.tokens(skill) if len(w) > 1]
    if not words:
        return True                       # nothing lexical to check — keep the evidence

    present = {tm.normalise(t) for t in tm.tokens(chunk)}
    present.discard("")
    for word in words:
        target = tm.normalise(word)
        if target and (target in present
                       or any(tm.is_misspelling(target, p) for p in present)):
            return True

    # Fallback: check for semantic role/designation similarity.
    # This allows "HR Executive" to match "recruiter" or "Developer" to match "Engineer".
    if _is_role_term(skill) and _has_role_semantic_match(chunk, skill):
        return True

    return False


def _is_role_term(text: str) -> bool:
    """Whether this text describes a job role/designation rather than a skill."""
    # Role vocabulary: engineer, developer, manager, designer, architect, consultant, etc.
    # Excludes technical skill names: Python, React, Kubernetes, AWS, etc.
    role_keywords = {
        "engineer", "developer", "programmer", "architect", "analyst", "manager",
        "designer", "scientist", "administrator", "consultant", "specialist", "lead",
        "director", "intern", "devops", "sre", "researcher", "owner", "technician",
        "recruiter", "hr", "humanresource", "coordinator", "officer", "executive",
    }
    # Normalize: remove spaces and punctuation for comparison
    normalized = tm.normalise(text).lower()
    tokens = tm.tokens(text.lower())
    # Check both full normalized text and individual tokens
    return any(kw in normalized for kw in role_keywords) or any(kw in role_keywords for kw in tokens)


def _has_role_semantic_match(chunk: str, requested_role: str) -> bool:
    """
    Extract any role mentioned in the chunk and compare it semantically to the
    requested role. Returns True if they are semantically similar (>0.58 embedding cosine).

    This allows "HR Executive" to match "recruiter", "Senior Developer" to match "developer",
    "Engineering Manager" to match "manager", across ANY domain.

    Uses embedding-based semantic similarity since roles like "HR" and "recruiter"
    share no lexical tokens but are semantically equivalent.
    """
    try:
        # Import here to avoid circular imports
        from app.services.ai.embedding_service import get_embedding_model
        import numpy as np

        # Extract sequences that look like job titles (contain role keywords)
        role_tokens = {
            "engineer", "developer", "programmer", "architect", "analyst", "manager",
            "designer", "scientist", "administrator", "consultant", "specialist", "lead",
            "director", "intern", "devops", "sre", "researcher", "owner", "technician",
            "recruiter", "hr", "coordinator", "officer", "executive", "talent", "hiring",
        }

        # Modifiers that typically precede a role (seniority, domain, etc.)
        role_modifiers = {
            "senior", "junior", "lead", "principal", "staff", "backend", "frontend",
            "full", "full-stack", "fullstack", "senior", "mid", "software", "solutions",
            "technical", "engineering", "product", "systems", "infrastructure", "devops",
            "platform", "data", "machine", "learning", "cloud", "security", "network",
        }

        # Split chunk into words and find sequences containing role keywords
        words = chunk.split()
        found_roles = []

        for i, word in enumerate(words):
            normalized_word = tm.normalise(word).lower()
            if normalized_word in role_tokens:
                # Build minimal phrase: include the role and the word immediately before if it's a modifier
                # e.g., "Senior Developer", "Backend Engineer"
                start = i
                if i > 0:
                    prev_word = tm.normalise(words[i-1]).lower()
                    if prev_word in role_modifiers or len(prev_word) > 1:
                        # Include previous word if it's a known modifier or looks like an adjective
                        start = i - 1

                phrase = " ".join(words[start:i+1])
                # Clean up punctuation from the phrase (e.g., "Engineer." -> "Engineer")
                phrase = re.sub(r'[.,;:!?]+$', '', phrase).strip()
                if phrase:
                    found_roles.append(phrase)

        if not found_roles:
            return False

        # Get embedding model and compute semantic similarity
        model = get_embedding_model()
        requested_embedding = np.asarray(model.get_text_embedding(requested_role.strip()), dtype=float)
        requested_embedding /= (np.linalg.norm(requested_embedding) + 1e-9)

        similarity_threshold = 0.60
        for found_role in found_roles:
            found_embedding = np.asarray(model.get_text_embedding(found_role.strip()), dtype=float)
            found_embedding /= (np.linalg.norm(found_embedding) + 1e-9)
            # Cosine similarity of L2-normalized vectors
            similarity = float(np.dot(requested_embedding, found_embedding))
            if similarity >= similarity_threshold:
                return True

        return False
    except Exception as e:
        logger.debug(f"Role semantic match failed (falling back to False): {e}")
        return False


def _misspelled_matches(
    requested: Sequence[str], matched: Sequence[str], profile: Dict[str, Any]
) -> List[str]:
    """
    Requested skills the candidate holds but SPELLS WRONG.

    Retrieval cannot find these: BM25 needs the literal token, and a one-letter slip
    is not reliably close in embedding space either. So a real React developer whose
    resume says "Reat js" was scored as having no React at all — while the JD-side
    Match Score, reading the same specification, gave him partial credit.

    Checked against the candidate's own declared skill list rather than the whole
    document, because that is where a claimed skill lives and where a typo in it is
    unambiguous. Whole-document matching would let a near-miss inside ordinary prose
    manufacture a skill nobody claimed.
    """
    outstanding = [s for s in requested if s not in matched]
    if not outstanding:
        return []
    declared = [tm.normalise(s) for s in (profile.get("skills") or [])]
    if not declared:
        return []

    found: List[str] = []
    for skill in outstanding:
        target = tm.normalise(skill)
        if target and any(tm.is_misspelling(target, d) for d in declared):
            found.append(skill)
    return found


def score_request(
    *,
    requested: Sequence[str],
    matched: Sequence[str],
    demonstrated: Sequence[str],
    place_ok: bool,
    experience_fit: float,
    intent: QueryIntent,
    fallback: float,
    misspelled: Sequence[str] = (),
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Score a candidate against a recruiter's *question*, using the Match Score
    parameters and weights.

    The platform must not carry two scoring algorithms. The chatbot used to run its
    own — `coverage×0.5 + evidence×0.3 + experience×0.2` — which meant the same
    candidate got one number on the Analysis screen and a different one in chat, and
    neither could be reconciled against the agreed specification.

    A question is not a Job Description, so only the parameters the recruiter actually
    stated are scored, and their Match Score weights are **renormalised** across them.
    Ask for React in Chennai and the score is Skill 20 + Technology 14 + Location 8,
    rescaled to 100 — so a perfect match still reads 100 rather than being dragged to
    66 by six neutral dimensions nobody asked about.

    Per-parameter marking follows the team's convention:

      * Skill       — the fraction of requested skills the resume evidences at all.
      * Technology  — the fraction proven in Experience or Projects rather than merely
                      listed. This is the Match Score's claimed-versus-practised split,
                      and it is what stops a skills-list keyword from outranking real work.
      * Location    — binary. In the requested place or not; there is no half a city.
      * Experience  — how well total years sit inside the requested bar.

    Returns (score 0-100, per-parameter breakdown).
    """
    components: List[Dict[str, Any]] = []

    if requested:
        total = len(requested)
        # A typo is a spelling mistake, not a missing skill. Shabeer's resume writes
        # "Reat js"; refusing him any credit for React because of one dropped letter
        # is the exact case the specification calls "minor spelling mistakes receive
        # partial credit".
        partial = settings.MATCH_PARTIAL_SKILL_CREDIT * len(misspelled)
        basis = f"{len(matched)} of {total} requested skills evidenced"
        if misspelled:
            basis += f"; {', '.join(misspelled)} matched through a spelling variant"
        components.append({
            "key": "skill", "label": "Skill Match",
            "weight": settings.MATCH_WEIGHT_SKILL,
            "score": (len(matched) + partial) / total * 100.0,
            "basis": basis,
        })
        components.append({
            "key": "technology", "label": "Technology Match",
            "weight": settings.MATCH_WEIGHT_TECHNOLOGY,
            "score": len(demonstrated) / total * 100.0,
            "basis": f"{len(demonstrated)} of {total} proven in real work, not just listed",
        })

    if intent.places:
        where = ", ".join(intent.place_labels) or "the requested location"
        components.append({
            "key": "location", "label": "Location Match",
            "weight": settings.MATCH_WEIGHT_LOCATION,
            "score": 100.0 if place_ok else 0.0,
            "basis": f"{'in' if place_ok else 'not in'} {where}",
        })

    if intent.min_years is not None or intent.max_years is not None:
        components.append({
            "key": "experience", "label": "Experience Match",
            "weight": settings.MATCH_WEIGHT_EXPERIENCE,
            "score": max(0.0, min(1.0, experience_fit)) * 100.0,
            "basis": f"against {experience_bar_label(intent)}",
        })

    weight_sum = sum(c["weight"] for c in components)
    if not weight_sum:
        # Nothing was asked for — an overview question. There is no requirement to
        # score against, so inventing a percentage would imply a judgement.
        return max(0.0, min(1.0, fallback)), []

    for c in components:
        c["score"] = round(max(0.0, min(100.0, c["score"])), 1)
        # The share of the FINAL score this parameter carries, after renormalising.
        c["weight"] = round(c["weight"] / weight_sum, 4)
        c["contribution"] = round(c["score"] * c["weight"], 1)

    return sum(c["contribution"] for c in components) / 100.0, components


def experience_bar_label(intent: QueryIntent) -> str:
    """How to name the experience constraint in recruiter-facing copy."""
    if intent.min_years is not None and intent.max_years is not None:
        return f"{intent.min_years:g}-{intent.max_years:g} years"
    if intent.max_years is not None:
        return f"under {intent.max_years:g} years"
    if intent.min_years is not None:
        return f"{intent.min_years:g}+ years"
    return "experience"


def _experience_fit(
    total_years: Optional[float],
    min_years: Optional[float],
    max_years: Optional[float] = None,
) -> Tuple[float, str]:
    """
    0..1 fit plus a human-readable verdict.

    Handles both directions. Treating every bar as a floor is what let "less than 5
    years" return an 11-year lead and call it a 100% experience fit — the more
    experience someone had, the better they scored against a request for *less*.
    """
    if min_years is None and max_years is None:
        if total_years is None:
            return 1.0, "no experience requirement was stated"
        return 1.0, f"{total_years:g} years of total experience"
    if total_years is None:
        return 0.5, "years of experience could not be determined from the resume"

    bar = experience_bar_label_from(min_years, max_years)

    if max_years is not None and total_years > max_years:
        excess = total_years - max_years
        return (
            max(0.0, 1.0 - excess / max(max_years, 1.0)),
            f"{total_years:g} yrs against {bar} — {excess:g} yrs over the limit",
        )
    if min_years is not None and total_years < min_years:
        shortfall = min_years - total_years
        return (
            max(0.0, 1.0 - shortfall / max(min_years, 1.0)),
            f"{total_years:g} yrs against {bar} — {shortfall:g} yrs short",
        )

    # Inside the requested band.
    if max_years is not None:
        return 1.0, f"{total_years:g} yrs, within {bar}"
    surplus = total_years - (min_years or 0.0)
    note = f"{total_years:g} yrs against {bar}"
    note += f" — {surplus:g} yrs clear of the bar" if surplus else " — exactly on the bar"
    return min(1.0, 0.9 + min(surplus, 5) * 0.02), note


def experience_bar_label_from(
    min_years: Optional[float], max_years: Optional[float]
) -> str:
    if min_years is not None and max_years is not None:
        return f"the {min_years:g}-{max_years:g} years you asked for"
    if max_years is not None:
        return f"the under-{max_years:g} years you asked for"
    return f"the {min_years:g} you asked for"


def _location_note(
    corpus: Optional[Corpus], profile: Dict[str, Any], intent: QueryIntent
) -> Tuple[bool, str]:
    """Whether this candidate satisfies the requested place, and how to say it."""
    if not intent.places or corpus is None:
        return True, ""
    rid = profile["resume_id"]
    hit = [
        corpus.lexicon.place_display.get(key, key)
        for key in intent.places
        if rid in corpus.lexicon.places.get(key, set())
    ]
    if hit:
        return True, f"based in {', '.join(hit)}, as you asked"
    stated = profile.get("location")
    return False, (
        f"listed in {stated}, not {', '.join(intent.place_labels)}" if stated
        else f"no address in the resume places them in {', '.join(intent.place_labels)}"
    )


def _score_candidate(
    profile: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    intent: QueryIntent,
    corpus: Optional[Corpus] = None,
) -> Dict[str, Any]:
    """
    Deterministic match percentage, decomposed so it can be defended.

      coverage   — mean DEPTH of the requested skills (not found/not-found)  .50
      strength   — how strong the retrieved evidence is                      .30
      experience — years against the stated bar                              .20

    With no experience requirement, that weight folds into coverage so the score still
    spans the full range.
    """
    requested = intent.skills or []
    by_skill: Dict[str, List[Dict[str, Any]]] = {}
    for e in evidence:
        if not _confirms(e.get("text") or "", e["skill"]):
            continue
        by_skill.setdefault(e["skill"], []).append(e)

    depths: Dict[str, float] = {}
    sections_by_skill: Dict[str, List[str]] = {}
    for skill in requested:
        depth, sections = _skill_depth(by_skill.get(skill, []))
        depths[skill] = depth
        sections_by_skill[skill] = sections

    matched = [s for s in requested if depths.get(s, 0.0) > 0]
    missing = [s for s in requested if depths.get(s, 0.0) == 0]

    if requested:
        coverage = sum(depths.values()) / len(requested)
    else:
        coverage = 1.0 if evidence else 0.0

    if evidence:
        best_per_skill = [
            max(e["similarity"] * section_proof(e["section"]) for e in by_skill[s]) * 1.25
            for s in matched if by_skill.get(s)
        ]
        strength = min(sum(best_per_skill) / len(best_per_skill), 1.0) if best_per_skill else 0.0
    else:
        strength = 0.0

    exp_fit, exp_note = _experience_fit(profile.get("total_years"), intent.min_years, intent.max_years)
    place_ok, place_note = _location_note(corpus, profile, intent)

    demonstrated = sorted(s for s in matched if set(sections_by_skill[s]) & _STRONG_SECTIONS)
    listed_only = sorted(s for s in matched if s not in demonstrated)

    misspelled = _misspelled_matches(requested, matched, profile)

    score, breakdown = score_request(
        requested=requested,
        matched=matched,
        demonstrated=demonstrated,
        place_ok=place_ok,
        experience_fit=exp_fit,
        intent=intent,
        fallback=coverage * 0.70 + strength * 0.30,
        misspelled=misspelled,
    )

    return {
        "resume_id": profile["resume_id"],
        "name": profile.get("name"),
        "title": profile.get("title"),
        "filename": profile.get("filename"),
        "email": profile.get("email"),
        "phone": profile.get("phone"),
        "location": profile.get("location"),
        "total_years": profile.get("total_years"),
        "seniority_level": profile.get("seniority_level"),
        "all_skills": (profile.get("skills") or [])[:20],
        "match_percentage": int(round(min(max(score, 0.0), 1.0) * 100)),
        # The Match Score parameters this question actually stated, with their
        # renormalised weights — the same decomposition the Analysis screen shows.
        "match_parameters": breakdown,
        "breakdown": {
            "skill_coverage": int(round(coverage * 100)),
            "evidence_strength": int(round(strength * 100)),
            "experience_fit": int(round(exp_fit * 100)),
        },
        "skill_depth": {s: int(round(d * 100)) for s, d in depths.items()},
        "skill_sections": sections_by_skill,
        "matched_skills": matched,
        "missing_skills": missing,
        "demonstrated_skills": demonstrated,
        "listed_only_skills": listed_only,
        "experience_note": exp_note,
        "location_match": place_ok,
        "location_note": place_note,
        "evidence": sorted(
            evidence, key=lambda e: (e.get("literal", False), e["similarity"]), reverse=True
        )[:6],
    }


# --- Per-candidate retrieval (fact questions) -------------------------------
def _resume_rows(corpus: Corpus, resume_id: int) -> List[int]:
    """This candidate's rows — precomputed by the store, so this is a dict lookup."""
    return list(corpus.rows_by_resume.get(resume_id, []))


def search_within_resume(
    corpus: Corpus, resume_id: int, probe: str, limit: int = 4
) -> List[Dict[str, Any]]:
    """
    Rank ONE candidate's own passages against an arbitrary question.

    This is what makes "which college did he attend", "what is his notice period" or
    any other question the document happens to answer actually answerable. The pool-wide
    index is the wrong instrument here — it ranks across everybody, so a passage that is
    the best answer *for this person* can be buried under three hundred other resumes.
    Reconstructing this resume's vectors from the flat index and scoring them directly
    costs one query embedding and a dot product over ~15 rows.

    Dense similarity is blended with literal word overlap, because a factual question
    is often answered by a line that shares its vocabulary ("Notice period: 30 days")
    rather than by a semantic paraphrase.
    """
    rows = _resume_rows(corpus, resume_id)
    if not rows or not (probe or "").strip():
        return []

    scores: Dict[int, float] = {}
    try:
        # The shared store scores a single candidate's rows EXHAUSTIVELY rather than
        # filtering a global top-N, so a passage that is the best answer for this
        # person cannot be buried under three hundred other resumes.
        scores = {
            row: score
            for row, score in corpus.search(probe, top_k=len(rows), resume_ids={resume_id})
        }
    except Exception as e:  # a missing/incompatible index must not break a fact lookup
        logger.warning(f"In-resume dense scoring unavailable ({e}); using lexical overlap only.")

    terms = {w for w in _WORD.findall((probe or "").lower()) if len(w) > 2}
    ranked: List[Tuple[float, int]] = []
    for row in rows:
        text = (corpus.chunks[row].get("text") or "").lower()
        overlap = sum(1 for t in terms if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text))
        lexical = overlap / len(terms) if terms else 0.0
        ranked.append((0.6 * scores.get(row, 0.0) + 0.4 * lexical, row))

    ranked.sort(reverse=True)
    out: List[Dict[str, Any]] = []
    for score, row in ranked[:limit]:
        chunk = corpus.chunks[row]
        out.append({
            "skill": "answer",
            "text": chunk["text"],
            "section": chunk["section"],
            "page": chunk["page"],
            "filename": chunk["filename"],
            "similarity": round(max(min(score, 1.0), 0.0), 3),
            "literal": bool(terms) and any(
                re.search(rf"(?<!\w){re.escape(t)}(?!\w)", chunk["text"].lower()) for t in terms
            ),
        })
    return out


# Structured fields the platform already extracts, and how to phrase each answer.
# This maps English to OUR OWN data model — it is not a mapping between skills, and it
# is a fallback, not a whitelist: anything not listed is answered from the document.
# A postal code is part of an address, not part of an answer: "based in Hyderabad" is
# what a recruiter wants to read, not "based in Hyderabad 500034".
_POSTAL_TAIL = re.compile(r"[\s,-]+\d{5,6}\s*$")


def _readable_place(value: Optional[str]) -> Optional[str]:
    cleaned = _POSTAL_TAIL.sub("", (value or "").strip()).strip(" ,-")
    return cleaned or None


_STRUCTURED_READERS: Dict[str, Any] = {
    "location": lambda p: _readable_place(p.get("location")),
    "email": lambda p: p.get("email"),
    "phone": lambda p: p.get("phone"),
    "title": lambda p: p.get("title"),
    "seniority": lambda p: p.get("seniority_level"),
    "name": lambda p: p.get("name"),
    "experience": lambda p: (f"{p['total_years']:g} years" if p.get("total_years") is not None else None),
    "initials": lambda p: (
        "".join(part[0].upper() for part in (p.get("name") or "").split() if part) or None
    ),
}


def answer_about_person(corpus: Corpus, intent: QueryIntent) -> Dict[str, Any]:
    """
    Answer one factual question about one candidate, from that candidate's resume.

    Returns a fact record: the value when we have one, the passages that support it,
    and — crucially — an explicit `found: False` when the resume simply does not say.
    Inventing an answer, or falling back to a pool-wide search that returns a stranger,
    are both worse than admitting the document is silent.
    """
    person = intent.people[0]
    profile = corpus.resumes.get(person.resume_id) or {}

    value: Optional[str] = None
    if intent.attribute:
        reader = _STRUCTURED_READERS.get(intent.attribute)
        if reader:
            value = reader(profile)
        # A place harvested from the body of the resume is still a real answer even
        # when the header had no parseable contact line.
        if value is None and intent.attribute == "location":
            places = [
                corpus.lexicon.place_display.get(key, key)
                for key, ids in corpus.lexicon.places.items()
                if person.resume_id in ids
            ]
            value = ", ".join(sorted(places, key=len, reverse=True)[:2]) or None

    probe = intent.attribute_probe or intent.attribute or intent.raw
    evidence = search_within_resume(corpus, person.resume_id, probe)
    # Only passages that genuinely relate to the question are worth quoting.
    supporting = [e for e in evidence if e["similarity"] > 0.25 or e["literal"]]

    return {
        "resume_id": person.resume_id,
        "name": profile.get("name") or person.name,
        "attribute": intent.attribute or (intent.attribute_probe or "that"),
        "value": value,
        "found": bool(value) or bool(supporting),
        "evidence": supporting or evidence[:2],
        "profile": profile,
    }


# --- Comparison -------------------------------------------------------------
def compare_candidates(
    corpus: Corpus, intent: QueryIntent, limit: int = 4
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Put two or more named candidates side by side on the same axes.

    A comparison is NOT a search that happens to return two people. A search ranks the
    pool against a requirement; a comparison ranks a fixed, named set against *each
    other*, and the useful output is where they differ — who has what the other lacks,
    who proved it in real work, who has more relevant experience.

    Every axis is drawn from the union of both resumes' skills, so the comparison
    surfaces what the recruiter did not think to ask about. When the question named a
    skill ("who is better at Java"), that skill leads.
    """
    people = intent.people[:limit]
    if len(people) < 2:
        return [], {"mode": "compare", "compared": len(people)}

    profiles = [corpus.resumes[p.resume_id] for p in people if p.resume_id in corpus.resumes]
    if len(profiles) < 2:
        return [], {"mode": "compare", "compared": len(profiles)}

    # The axes: what was asked about first, then whatever either resume evidences.
    from app.services.ai.chat_lexicon import KNOWN_TECH

    axes: List[str] = list(intent.skills)
    for profile in profiles:
        for skill in (profile.get("skills") or []):
            if skill.lower() in KNOWN_TECH and skill not in axes:
                axes.append(skill)
    axes = axes[:8]

    floor = _chat_min_similarity(corpus)
    wanted = {p["resume_id"] for p in profiles}
    evidence_by_resume: Dict[int, List[Dict[str, Any]]] = {}
    for skill in axes:
        for rid, chunks in _search_skill(corpus, skill, wanted, len(wanted), floor).items():
            evidence_by_resume.setdefault(rid, []).extend(chunks)

    # Score each candidate on the SHARED axes, so the numbers are comparable.
    axis_intent = QueryIntent(raw=intent.raw, kind=intent.kind, skills=axes,
                              min_years=intent.min_years, max_years=intent.max_years)
    scored = [
        _score_candidate(profile, evidence_by_resume.get(profile["resume_id"], []),
                         axis_intent, corpus)
        for profile in profiles
    ]

    # What actually separates them — the only part of a comparison worth reading.
    for candidate in scored:
        others = [c for c in scored if c["resume_id"] != candidate["resume_id"]]
        proven = set(candidate["demonstrated_skills"])
        others_proven = set().union(*[set(o["demonstrated_skills"]) for o in others]) if others else set()
        others_any = set().union(*[set(o["matched_skills"]) for o in others]) if others else set()
        candidate["only_they_have"] = sorted(set(candidate["matched_skills"]) - others_any)
        candidate["proven_where_others_only_claim"] = sorted(proven - others_proven)
        candidate["others_have_that_they_lack"] = sorted(others_any - set(candidate["matched_skills"]))
        if not candidate["evidence"]:
            candidate["evidence"] = _structural_evidence(corpus, candidate["resume_id"], intent)

    scored.sort(key=lambda c: (c["match_percentage"], c.get("total_years") or 0), reverse=True)
    stats = {
        "mode": "compare",
        "compared": len(scored),
        "axes": axes,
        "pool_size": len(corpus.resumes),
        "after_prefilter": len(scored),
        "vector_searches": len(axes),
    }
    logger.info(f"Compared {[c['name'] for c in scored]} on {len(axes)} axes.")
    return scored, stats


# --- Profile overview -------------------------------------------------------
def profile_overview(corpus: Corpus, intent: QueryIntent, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Build a general overview for "tell me more about X" — no skill requirements.

    The recruiter asked about a person, so the answer describes the person: their role,
    seniority, the skills their resume actually evidences, and their strongest resume
    passages. Nothing is scored as missing, because nothing was requested.
    """
    out: List[Dict[str, Any]] = []
    for person in intent.people[:limit]:
        profile = corpus.resumes.get(person.resume_id)
        if not profile:
            continue
        rows = _resume_rows(corpus, person.resume_id)
        chunks = [corpus.chunks[r] for r in rows]
        # Lead with the sections that describe what someone has actually done.
        chunks.sort(key=lambda c: (section_proof(c["section"]), len(c["text"])), reverse=True)
        evidence = [{
            "skill": "profile",
            "text": c["text"],
            "section": c["section"],
            "page": c["page"],
            "filename": c["filename"],
            "similarity": 1.0,
            "literal": True,
        } for c in chunks[:5]]

        scored = _score_candidate(profile, [], intent, corpus)
        scored.update({
            "evidence": evidence,
            "match_percentage": 0,          # nothing was requested, so nothing to match
            "breakdown": {"skill_coverage": 0, "evidence_strength": 0, "experience_fit": 0},
            "sections_present": sorted({c["section"] for c in chunks}),
        })
        out.append(scored)
    return out


# --- Pool search ------------------------------------------------------------
def search_candidates(
    corpus: Corpus,
    intent: QueryIntent,
    limit: int = 5,
    restrict_ids: Optional[Set[int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run the staged search. Returns (ranked candidates, diagnostics).

    Diagnostics carry the whole funnel, not just a final count, so the caller can tell
    the recruiter which constraint emptied the result and offer a specific alternative.
    """
    facets = FacetSets(corpus, intent, restrict_ids)
    floor = _chat_min_similarity(corpus)
    stats: Dict[str, Any] = {
        "skills_searched": intent.skills,
        "places_searched": intent.place_labels,
        "vector_searches": 0,
        "embedding_engine": corpus.engine_name,
        "similarity_floor": round(floor, 3),
    }

    # Skill evidence is gathered over everything the previous turn allows, NOT over the
    # already place-filtered set — otherwise we could never say "12 people know Java,
    # just none of them in T. Nagar", which is the sentence the recruiter needs.
    searchable = facets.restricted & facets.by_years & facets.by_person
    evidence_by_resume: Dict[int, List[Dict[str, Any]]] = {}
    if intent.skills and searchable:
        found_per_skill: Dict[str, Set[int]] = {}
        for skill in intent.skills:
            stats["vector_searches"] += 1
            hits = _search_skill(corpus, skill, searchable, len(searchable), floor)
            found_per_skill[skill] = set(hits)
            for rid, chunks in hits.items():
                evidence_by_resume.setdefault(rid, []).extend(chunks)
        facets.by_skill = set().union(*found_per_skill.values()) if found_per_skill else set()
        facets.by_skill_all = (
            set.intersection(*found_per_skill.values()) if found_per_skill else set()
        )

    # "Who is the most experienced?" asks the pool to be ORDERED by a field we already
    # hold, not searched for evidence of a skill. Falling through to the skill path
    # returned nothing at all — and the assistant's own suggestion chip offered exactly
    # this question, so it was recommending a dead end.
    if intent.rank_by == "experience" and not intent.skills:
        ranked = [
            _score_candidate(corpus.resumes[rid], [], intent, corpus)
            for rid in facets.structural
            if rid in corpus.resumes
        ]
        for candidate in ranked:
            candidate["evidence"] = _structural_evidence(corpus, candidate["resume_id"], intent)
        # Unknown years sort last: absence of a stated figure is not seniority.
        ranked.sort(key=lambda c: (c["total_years"] is not None, c["total_years"] or 0), reverse=True)
        stats.update(facets.counts(intent))
        stats["candidates_scored"] = len(ranked)
        stats["ordered_by"] = "experience"
        return ranked[:limit], stats

    selected = facets.final
    candidates = [
        _score_candidate(corpus.resumes[rid], evidence_by_resume.get(rid, []), intent, corpus)
        for rid in selected
        if rid in corpus.resumes
    ]

    # A named lookup must return the person even when a specific skill is not evidenced
    # — "no Java evidence for Priya" is the correct, useful answer. This is deliberately
    # limited to people the recruiter actually named: applying it to every structurally
    # eligible resume is what filled a Java search with candidates who had none.
    if intent.people:
        present = {c["resume_id"] for c in candidates}
        for person in intent.people:
            profile = corpus.resumes.get(person.resume_id)
            if profile and person.resume_id not in present:
                candidates.append(
                    _score_candidate(profile, evidence_by_resume.get(person.resume_id, []), intent, corpus)
                )

    # A candidate selected on structure alone — "everyone in T. Nagar", "anyone with
    # 10+ years" — has no skill evidence by definition. Leaving it empty would see the
    # anti-hallucination gate delete a perfectly legitimate answer, so the passages
    # that justify the structural match stand in as its evidence.
    for candidate in candidates:
        if not candidate["evidence"]:
            candidate["evidence"] = _structural_evidence(corpus, candidate["resume_id"], intent)

    candidates.sort(
        key=lambda c: (c["match_percentage"], c["breakdown"]["skill_coverage"],
                       c["breakdown"]["evidence_strength"]),
        reverse=True,
    )

    stats.update(facets.counts(intent))
    stats["candidates_scored"] = len(candidates)
    if not candidates:
        stats["relaxation_options"] = relaxation_options(facets, intent)
        stats["blocking_facet"] = _blocking_facet(facets, intent)

    logger.info(
        f"Chat search [{intent.kind.value}]: pool={stats.get('pool_size')} "
        f"prefiltered={stats.get('after_prefilter')} searches={stats['vector_searches']} "
        f"scored={stats['candidates_scored']}"
    )
    return candidates[:limit], stats


def _structural_evidence(
    corpus: Corpus, resume_id: int, intent: QueryIntent
) -> List[Dict[str, Any]]:
    """Passages that justify a match made on place/years rather than on a skill."""
    probe = " ".join(intent.place_labels) or "experience summary"
    found = search_within_resume(corpus, resume_id, probe, limit=3)
    if found:
        return found
    rows = _resume_rows(corpus, resume_id)[:2]
    return [{
        "skill": "profile",
        "text": corpus.chunks[r]["text"],
        "section": corpus.chunks[r]["section"],
        "page": corpus.chunks[r]["page"],
        "filename": corpus.chunks[r]["filename"],
        "similarity": 1.0,
        "literal": True,
    } for r in rows]


def _blocking_facet(facets: FacetSets, intent: QueryIntent) -> Optional[str]:
    """Which single constraint emptied the result — the thing worth asking about."""
    base = facets.restricted
    if intent.places and not (base & facets.by_place):
        return "location"
    if (intent.min_years is not None or intent.max_years is not None) and not (base & facets.by_years):
        return "experience"
    if intent.skills and not (base & facets.by_skill):
        return "skills"
    if intent.places and intent.skills and not (facets.by_place & facets.by_skill & base):
        return "combination"
    return "combination" if intent.has_facets else None
