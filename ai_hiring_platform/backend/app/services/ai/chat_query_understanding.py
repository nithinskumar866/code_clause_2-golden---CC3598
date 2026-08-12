"""
Recruiter query understanding — turning a sentence into a typed, checkable request.

WHAT A QUESTION ACTUALLY IS
---------------------------
A recruiter's question is not a bag of words. It is an **intent** ("search the pool" /
"tell me a fact about this person" / "defend what you just said") plus a set of
**facets** that constrain it — skills, a place, an experience bar, a person. Read that
structure wrong and every later stage is confidently wrong:

    "who has now experience"              -> "now" read as a skill -> 306 matches
    "best candidate in t nagar knows java" -> "candidate" read as a person's NAME
                                              -> java silently dropped
    "where is naveen k from"               -> no such intent existed at all
    "support your point why you say ..."   -> re-searched instead of explaining

So this module resolves facets against the corpus lexicon *first* (`chat_lexicon` —
the pool decides what its own words mean), classifies the intent from grammar second,
and never lets an unresolved word become a requirement.

FOUR RULES KEEP IT HONEST
-------------------------
1. **A word is a facet only if the corpus says so.** A skill must be a known
   technology or a term this pool writes like one. A place must be somewhere a resume
   actually puts a person. A name must belong to a real, identified candidate.
2. **Ambiguity is a first-class outcome.** Several people answer to "Naveen"? That is
   not a ranking problem, it is a question to ask back.
3. **A constraint that resolves to nothing is reported, never dropped.** If no resume
   mentions T. Nagar, the answer says so. Silently ignoring a constraint answers a
   question the recruiter did not ask.
4. **The conversation is part of the question.** "where is he from" is unanswerable
   alone and perfectly clear after "tell me about Naveen" — so context is an input,
   not an afterthought.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.ai.chat_lexicon import (
    FUNCTION_WORDS,
    CorpusLexicon,
    NameResolution,
    Person,
)

# An experience bar has a DIRECTION, and reading it wrong is worse than missing it.
# "less than 5 years" was matched by the bare number and stored as a minimum, so a
# request for junior candidates returned an 11-year lead and called them a 100%
# experience fit. Upper bounds are therefore matched first and separately.
_MAX_YEARS_RE = re.compile(
    r"(?:less\s+than|under|below|fewer\s+than|no\s+more\s+than|at\s+most|"
    r"maximum|max|up\s?to|within|upto)\s+"
    r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b",
    re.I,
)
# "5 years or less", "5 yrs max" — the same bound written after the number.
_MAX_YEARS_TRAILING_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s+(?:or\s+(?:less|fewer|below|under)|max(?:imum)?)\b",
    re.I,
)
_MIN_YEARS_RE = re.compile(
    r"(?:(?:at\s+least|minimum|min|more\s+than|over|above|atleast)\s+)?"
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
    re.I,
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#./-]*")

# "show me 10 candidates", "top 3 java developers", "give 15 profiles". The recruiter
# said how many they wanted in the sentence; ignoring it and returning the default 5
# looks like the search silently truncated.
#
# Anchored to counting words on BOTH sides so it can never swallow a number that means
# something else — "10 years experience" and "React 18" are not requests for 10 or 18
# candidates.
_WANTED_COUNT_RE = re.compile(
    r"\b(?:top|best|first|show(?:\s+me)?|give(?:\s+me)?|list|display|need|want|fetch|"
    r"return|get)\s+(?:me\s+)?(\d{1,3})\b"
    r"|\b(\d{1,3})\s+(?:candidates?|profiles?|resumes?|people|matches|results?|names?)\b",
    re.I,
)
# Matches the API contract's ceiling (schemas/chat.py :: ChatRequest.limit).
_MAX_WANTED_COUNT = 20

# Morphologically technical: CamelCase, acronym, digit-bearing, or symbol-carrying
# (C++, CI/CD, node.js). Lets a technology the taxonomy has never heard of stay
# searchable, without admitting ordinary prose.
_MORPH_TECH = re.compile(
    r"[A-Za-z]+[A-Z0-9]|[A-Z]{2,}|[A-Za-z]*[0-9][A-Za-z0-9]*|[A-Za-z][A-Za-z0-9]*[+#/.][A-Za-z0-9+#/.]*"
)


class QueryKind(str, Enum):
    """What the recruiter is actually asking for."""

    SKILL_SEARCH = "skill_search"   # "need a java developer with 10 years"
    PROFILE = "profile"             # "tell me more about David Pillai"
    ATTRIBUTE = "attribute"         # "where is he from", "what are his initials"
    EXISTENCE = "existence"         # "is there anyone named Nithin?"
    COMPARE = "compare"             # "compare Priya and David"
    COUNT = "count"                 # "how many candidates know python?"
    META = "meta"                   # "why do you say he's the best?" — about the answer
    DEFINITION = "definition"       # "what is java" -> not an encyclopedia
    AMBIGUOUS = "ambiguous"         # nothing recognisable -> ask


@dataclass
class QueryIntent:
    """One recruiter question, fully resolved against the pool."""

    raw: str
    kind: QueryKind = QueryKind.AMBIGUOUS

    # --- facets -------------------------------------------------------------
    skills: List[str] = field(default_factory=list)
    min_years: Optional[float] = None
    # An UPPER bound on experience ("less than 5 years"). Held separately from
    # min_years because an experience bar has a direction, and collapsing the two made
    # a request for junior candidates rank the most senior person in the pool first.
    max_years: Optional[float] = None
    # How many candidates the recruiter asked to see, when they said so in words
    # ("show me 10 candidates"). None means "use the caller's default".
    wanted_count: Optional[int] = None
    # Lexicon keys for places named in the question, plus their display form.
    places: List[str] = field(default_factory=list)
    place_labels: List[str] = field(default_factory=list)
    # A place the recruiter named that no resume mentions. Reported, never dropped.
    unmatched_place: Optional[str] = None

    # --- people -------------------------------------------------------------
    people: List[Person] = field(default_factory=list)
    name_ambiguous: bool = False
    # The people ONE ambiguous name could refer to, and the words that were typed.
    # Held separately from `people` so a question naming two candidates, one of whom
    # is ambiguous, can still report the half it understood.
    ambiguous_options: List[Person] = field(default_factory=list)
    ambiguous_surface: Optional[str] = None
    unresolved_name: Optional[str] = None
    name_suggestions: List[Person] = field(default_factory=list)

    # Order the whole pool by something we hold rather than by skill evidence
    # ("who is the most experienced?"). Currently only "experience".
    rank_by: Optional[str] = None

    # --- attribute questions ------------------------------------------------
    # A structured field name when we hold one ("location", "email"), else None.
    attribute: Optional[str] = None
    # The phrase to retrieve with when the attribute is not a structured field.
    attribute_probe: str = ""

    # --- conversation -------------------------------------------------------
    # The question leans on the previous turn ("he", "that one", "of those").
    refers_to_previous: bool = False
    restricts_to_previous: bool = False

    # A bare facet with no request around it — the recruiter typed "java", not "who
    # knows java". It names a topic rather than asking a question, so the useful reply
    # is to narrow it down together rather than to dump a ranked list.
    underspecified: bool = False

    # --- bookkeeping --------------------------------------------------------
    # Content words the pool could not place. Naming them turns a useless "I don't
    # understand" into "I don't recognise 'now' as a skill here" — which tells the
    # recruiter what to change.
    unrecognised: List[str] = field(default_factory=list)
    corrections: List[Tuple[str, str]] = field(default_factory=list)
    semantic_probe: str = ""
    clarification: Optional[str] = None
    # How the intent was produced, for the diagnostics line: rules | rules+llm.
    parsed_by: str = "rules"

    @property
    def named_candidates(self) -> List[str]:
        """Names of the resolved people — the shape the API contract exposes."""
        return [p.name for p in self.people]

    @property
    def person_ids(self) -> List[int]:
        return [p.resume_id for p in self.people]

    @property
    def has_facets(self) -> bool:
        return (
            bool(self.skills or self.places or self.people or self.rank_by)
            or self.min_years is not None
            or self.max_years is not None
        )

    def is_lookup(self) -> bool:
        return self.kind in (QueryKind.PROFILE, QueryKind.COMPARE, QueryKind.ATTRIBUTE) and bool(self.people)


# --- Intent grammar ---------------------------------------------------------
_PROFILE_RE = re.compile(
    r"\b(?:tell|let)\s+me\s+(?:know\s+)?(?:more\s+)?(?:about|regarding)\b"
    r"|\b(?:more|details?|elaborat\w*|summar\w*|overview|background|profile|info(?:rmation)?)\s+"
    r"(?:on|about|of|for)\b"
    r"|\bwho\s+is\b|\bwhat\s+can\s+you\s+tell\s+me\s+about\b|\bbrief\s+me\b|\bshow\s+me\s+\w+'s\b",
    re.I,
)
_EXISTENCE_RE = re.compile(
    r"\bis\s+there\s+(?:any\s*(?:one|body)|a\s+candidate|someone)\b"
    r"|\bdo\s+(?:we|you)\s+have\s+(?:any\s*(?:one|body)|a\s+candidate|someone)\b"
    r"|\b(?:any\s*(?:one|body)|anybody)\s+(?:named|called|by\s+the\s+name)\b"
    r"|\bhave\s+we\s+got\s+(?:any\s*one|someone)\b",
    re.I,
)
_COMPARE_RE = re.compile(
    r"\bcompare\b|\bvs\.?\b|\bversus\b|\bbetter\s+(?:than|fit|match)\b|\bdifference\s+between\b", re.I
)
_COUNT_RE = re.compile(r"\bhow\s+many\b|\bcount\s+of\b|\bnumber\s+of\s+candidates\b", re.I)

# "who is the most experienced?", "rank by experience", "senior-most candidates".
# A legitimate question with NO skill in it: the pool is ORDERED by a field we already
# hold rather than searched for evidence. Without this it fell through to "nothing
# recognised" — and the assistant's own suggestion chip offered exactly this query, so
# it was recommending a dead end.
_RANK_EXPERIENCE_RE = re.compile(
    r"\b(?:most|highest|greatest|max(?:imum)?|top)\s+(?:\w+\s+){0,2}?(?:experience[d]?|years?)\b"
    r"|\brank(?:ed)?\s+by\s+(?:experience|years?|seniority)\b"
    r"|\b(?:senior|experienced)[\s-]?most\b"
    r"|\bwho\s+(?:is|are)\s+the\s+most\s+experienced\b",
    re.I,
)

# "what is java", "define kubernetes" — a question about a TECHNOLOGY, not the pool.
_DEFINITION_RE = re.compile(
    r"^\s*(?:what(?:'s| is| are)|define|explain|describe|meaning\s+of)\b"
    r"(?!.*\b(?:candidate|applicant|resume|profile|experience|background|skill)\b)",
    re.I,
)
_HOWTO_RE = re.compile(r"^\s*(?:how\s+(?:does|do|to)|why\s+(?:is|are|should|do))\b", re.I)

# A question ABOUT THE PREVIOUS ANSWER rather than a new search. Answering these by
# re-running retrieval is what made a candidate's score fall from 88% to 56% while
# the assistant was supposedly defending the 88%.
_META_RE = re.compile(
    r"\b(?:support|justify|defend|back)\s+(?:your|that|this|the)\b"
    r"|\bwhy\s+(?:do|did|would|are|is)\s+you\s+(?:say|said|think|pick|choose|chose|rank|rate|claim)\b"
    r"|\bwhy\s+(?:is|are|was|were)\s+(?:he|she|they|that|this|it)\b"
    r"|\bhow\s+(?:do|did)\s+you\s+(?:know|get|arrive|decide|calculate|score)\b"
    r"|\bon\s+what\s+basis\b|\bwhat\s+makes\s+(?:him|her|them|you)\b"
    r"|\bare\s+you\s+(?:sure|certain|confident)\b|\bprove\s+it\b|\bconvince\s+me\b"
    r"|\bexplain\s+(?:your|that|this|the)\s+(?:answer|reasoning|score|ranking|choice|logic)\b",
    re.I,
)

# Third-person / demonstrative reference to whoever the last answer was about.
_PERSON_PRONOUN = re.compile(r"\b(?:he|him|his|she|her|hers|they|them|their|theirs)\b", re.I)
_BARE_FOLLOWUP = re.compile(
    r"\b(?:tell\s+me\s+more|more\s+about\s+(?:him|her|them|this|that)|"
    r"details?\s+(?:on|about)\s+(?:him|her|them)|elaborate)\b", re.I,
)
_FOLLOWUP_SET = re.compile(
    r"\b(?:of\s+(?:those|them|these)|among\s+(?:those|them|these)|from\s+(?:those|them|these)|"
    r"narrow\s+(?:it\s+)?down|filter\s+(?:those|them)|which\s+of\s+(?:those|them|these)|"
    r"within\s+(?:those|them|these))\b",
    re.I,
)

# --- Attribute grammar ------------------------------------------------------
# These name STRUCTURED FIELDS the platform already extracts. This is a mapping from
# English to our own data model, not a mapping between skills, so Golden Rule 4 is
# untouched. Anything not listed here is answered by retrieving from the person's own
# resume, which is what makes "any question from the document" achievable.
_STRUCTURED_ATTRIBUTES: List[Tuple[str, re.Pattern]] = [
    ("location", re.compile(
        r"\bwhere\s+(?:is|are|does|do|did)\b.*\b(?:from|located|live|lives|based|stay|reside)\b"
        r"|\bwhere\s+(?:is|are)\b(?!.*\bexperience\b)"
        r"|\b(?:location|address|city|hometown|home\s+town|native|residence|residing|"
        r"place|locality|area|region|domicile)\b",
        re.I)),
    ("email", re.compile(r"\b(?:e-?mail|mail\s*(?:id|address)?)\b", re.I)),
    ("phone", re.compile(r"\b(?:phone|mobile|cell|contact\s+(?:number|details?)|number\s+to\s+call)\b", re.I)),
    ("experience", re.compile(
        r"\bhow\s+(?:many|much)\s+(?:years?|yrs?|experience)\b"
        r"|\b(?:total|overall|years?\s+of)\s+experience\b|\bhow\s+experienced\b", re.I)),
    ("title", re.compile(
        r"\b(?:designation|job\s+title|current\s+role|present\s+role|what\s+(?:is|was)\s+"
        r"(?:his|her|their|the)\s+(?:role|title|position))\b", re.I)),
    ("initials", re.compile(r"\binitials?\b", re.I)),
    ("name", re.compile(r"\b(?:full\s+name|complete\s+name|what\s+is\s+(?:his|her|their)\s+name)\b", re.I)),
    ("seniority", re.compile(r"\b(?:seniority|senior(?:ity)?\s+level|which\s+level)\b", re.I)),
]

# Grammar that says the subject of the sentence is a PERSON. Used to notice that a
# recruiter asked about somebody the pool does not contain — "is there anyone named
# Nithin?" must produce "no such person", never a search for whatever else is in the
# sentence. Fuzzy matching alone cannot do this: a name with no near neighbour in the
# pool has nothing to fuzzy-match against, and used to vanish silently.
_NAMED_AS_RE = re.compile(
    r"\b(?:named|called|by\s+the\s+name(?:\s+of)?)\s+([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*)?)",
    re.I,
)
_ASKS_ABOUT_PERSON = re.compile(
    r"\bwho\s+is\b|\bwho\s+are\b|\babout\b|\bprofile\s+of\b|\bdetails?\s+of\b"
    r"|\bwhere\s+is\b|\bwhere\s+does\b"
    r"|\b\w+'s\s+(?:skills?|experience|profile|education|background|location)\b"
    r"|\b\w+\s+(?:skills?|experience|profile|education|background|location)\s*\??$",
    re.I,
)

# Wh-grammar that signals a request for one FACT rather than a search of the pool.
_WH_FACT = re.compile(
    r"^\s*(?:where|when|which|what|whats|what's|who(?:se|m)?|how)\b|\bwhat\s+is\b|\bdoes\s+\w+\s+have\b",
    re.I,
)
# Possessive phrasing — "naveen's college", "his qualification".
_POSSESSIVE = re.compile(r"\b(\w+)'s\s+(\w+)", re.I)


def build_skill_vocabulary(resume_profiles: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """
    Backwards-compatible shim.

    The real vocabulary is now built by `chat_lexicon.CorpusLexicon`, which needs the
    corpus TEXT (not just the extracted skill lists) to tell a technology from an
    ordinary word. Kept so callers holding only profile records still work.
    """
    from app.services.ai.chat_lexicon import KNOWN_TECH

    vocab: Dict[str, str] = dict(KNOWN_TECH)
    for profile in resume_profiles:
        for skill in profile.get("skills") or []:
            if skill and skill.lower() not in FUNCTION_WORDS:
                vocab.setdefault(skill.lower(), skill)
    return vocab


def _extract_skills(
    text: str, lexicon: CorpusLexicon, exclude_tokens: Sequence[str] = ()
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Recognised skills plus any typo corrections, ignoring name and place tokens."""
    skills: List[str] = []
    corrections: List[Tuple[str, str]] = []
    seen = set()
    excluded = {t.lower() for t in exclude_tokens}

    lowered = (text or "").lower()
    consumed = lowered

    # Multi-word terms first ("machine learning", "spring boot") so their parts are not
    # re-read as separate skills.
    for term in sorted((t for t in lexicon.skill_terms if " " in t), key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", consumed):
            canonical = lexicon.skill_terms[term]
            if canonical.lower() not in seen:
                skills.append(canonical)
                seen.add(canonical.lower())
            consumed = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", " ", consumed)

    for token in _TOKEN_RE.findall(consumed):
        if token in excluded or token.isdigit() or len(token) < 2:
            continue
        canonical, corrected_from = lexicon.resolve_skill(token)
        if canonical is None:
            # An unknown token still counts when it *looks* technical in the original
            # casing — a framework newer than the taxonomy. Ordinary prose cannot pass.
            original = _original_casing(text, token)
            if (
                original
                and _MORPH_TECH.fullmatch(original)
                and token not in FUNCTION_WORDS
                and token not in lexicon.rejected_terms
                and not lexicon.is_name_token(token)
            ):
                canonical = original
            else:
                continue
        if canonical.lower() in seen:
            continue
        skills.append(canonical)
        seen.add(canonical.lower())
        if corrected_from:
            corrections.append((corrected_from, canonical))

    return skills[:8], corrections


def _original_casing(text: str, token: str) -> Optional[str]:
    """The token as the recruiter actually typed it — casing carries the signal."""
    match = re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text or "", re.I)
    return match.group(0) if match else None


def _detect_attribute(text: str) -> Optional[str]:
    for name, pattern in _STRUCTURED_ATTRIBUTES:
        if pattern.search(text):
            return name
    return None


def _attribute_words(text: str, intent: QueryIntent) -> List[str]:
    """
    What the question asks for, once everything identifying WHO is removed.

    "which college did naveen k attend" -> ["college", "attend"]. An EMPTY result is
    meaningful: "what is Priya Sharma" leaves nothing behind, which means no particular
    detail was requested and the recruiter wants the profile.
    """
    stripped = text or ""
    for person in intent.people:
        for token in person.tokens:
            stripped = re.sub(rf"(?<!\w){re.escape(token)}(?!\w)", " ", stripped, flags=re.I)
    return [
        w for w in _TOKEN_RE.findall(stripped.lower())
        if w not in FUNCTION_WORDS and len(w) > 2 and not w.isdigit()
    ]


def _attribute_probe(text: str, intent: QueryIntent, lexicon: CorpusLexicon) -> str:
    """
    The phrase to search the person's own resume with.

    Run against only that candidate's chunks, which is how an arbitrary document
    question gets a real, quotable answer instead of a shrug.
    """
    words = _attribute_words(text, intent)
    return " ".join(words[:8]) or (text or "").strip()


@dataclass
class Conversation:
    """
    What the previous turns established — an input to understanding, not a patch on it.

    Guardrails and the parser both consult this, which is why "where is he from" is
    now a legitimate, answerable question rather than an off-topic message.
    """

    subject: Optional[Person] = None
    last_candidate_ids: List[int] = field(default_factory=list)
    last_skills: List[str] = field(default_factory=list)
    last_min_years: Optional[float] = None
    last_places: List[str] = field(default_factory=list)
    has_answer: bool = False

    @property
    def has_subject(self) -> bool:
        return self.subject is not None


def parse_intent(
    query: str,
    lexicon: CorpusLexicon,
    context: Optional[Conversation] = None,
) -> QueryIntent:
    """
    Classify one recruiter question and resolve every facet against the pool.

    Order is deliberate: PEOPLE first (a person's name must never be read as a skill),
    then PLACES, then SKILLS from what is left, then the years bar, and only then the
    intent — because "tell me about X" must not turn the words of the request itself
    into requirements.
    """
    text = (query or "").strip()
    context = context or Conversation()
    intent = QueryIntent(raw=text)

    # --- conversational reference ------------------------------------------
    intent.refers_to_previous = bool(
        _PERSON_PRONOUN.search(text) or _BARE_FOLLOWUP.search(text)
    )
    intent.restricts_to_previous = bool(_FOLLOWUP_SET.search(text))

    # --- people -------------------------------------------------------------
    # Resolve EVERY person named, not just the best-scoring one — otherwise a
    # comparison silently loses its second candidate (see `resolve_mentions`).
    mentions: List[NameResolution] = lexicon.resolve_mentions(text)
    intent.people = [p for m in mentions if m.resolved for p in m.matched]
    # Only a genuinely ambiguous NAME asks a question back. A question that mentions
    # two different people is not ambiguous just because one of the two is.
    ambiguous = next((m for m in mentions if m.ambiguous), None)
    if ambiguous is not None:
        intent.name_ambiguous = True
        intent.ambiguous_options = list(ambiguous.matched)
        intent.ambiguous_surface = ambiguous.surface
        # The people we DID pin down stay resolved, so a comparison can say which half
        # it understood rather than starting over.
    unresolved = next((m for m in mentions if m.unresolved), None)
    if unresolved is not None:
        intent.unresolved_name = unresolved.unresolved
        intent.name_suggestions = list(unresolved.suggestions)

    # A pronoun inherits the subject of the conversation, exactly as a person would
    # read it. Only when the recruiter has not named someone new.
    if not intent.people and intent.refers_to_previous and context.has_subject:
        intent.people = [context.subject]
        intent.unresolved_name = None
        intent.name_suggestions = []

    name_tokens = [t for p in intent.people for t in p.tokens]

    # --- places -------------------------------------------------------------
    intent.places = lexicon.find_places(text)
    intent.place_labels = [lexicon.place_display.get(p, p) for p in intent.places]
    intent.unmatched_place = lexicon.unmatched_place_phrase(text, intent.places)
    place_tokens = [w for label in intent.places for w in label.split()]

    # --- how many to show ---------------------------------------------------
    if (m := _WANTED_COUNT_RE.search(text)):
        count = int(m.group(1) or m.group(2))
        if 1 <= count <= _MAX_WANTED_COUNT:
            intent.wanted_count = count

    # --- experience bar -----------------------------------------------------
    # Upper bounds are tested first: "less than 5 years" contains "5 years", so the
    # minimum pattern matches it too and would silently invert the recruiter's request.
    if (m := _MAX_YEARS_RE.search(text) or _MAX_YEARS_TRAILING_RE.search(text)):
        intent.max_years = float(m.group(1))
    elif (m := _MIN_YEARS_RE.search(text)):
        intent.min_years = float(m.group(1))

    # --- skills -------------------------------------------------------------
    intent.skills, intent.corrections = _extract_skills(
        text, lexicon, exclude_tokens=name_tokens + place_tokens
    )

    # --- intent classification ---------------------------------------------
    is_meta = bool(_META_RE.search(text)) and context.has_answer
    is_profile = bool(_PROFILE_RE.search(text))
    is_existence = bool(_EXISTENCE_RE.search(text))
    is_compare = bool(_COMPARE_RE.search(text))
    is_count = bool(_COUNT_RE.search(text))
    attribute = _detect_attribute(text)
    if _RANK_EXPERIENCE_RE.search(text):
        intent.rank_by = "experience"
        attribute = None if not intent.people else attribute

    # A question about the PREVIOUS ANSWER is not a new search, whatever words it
    # happens to contain.
    if is_meta:
        intent.kind = QueryKind.META
        intent.semantic_probe = " ".join(intent.skills) or text
        intent.parsed_by = "rules"
        return intent

    # A definition question about a technology is out of scope however it is phrased,
    # but only when nobody is named ("what is Priya's current role" is a fact request).
    if (
        not intent.people
        and not attribute
        and intent.skills
        and (_DEFINITION_RE.match(text) or _HOWTO_RE.match(text))
        and intent.min_years is None
        and intent.max_years is None
        and not intent.places
    ):
        intent.kind = QueryKind.DEFINITION
        return intent

    subject_known = bool(intent.people)

    if is_compare and len(intent.people) >= 1:
        intent.kind = QueryKind.COMPARE
    elif is_existence:
        intent.kind = QueryKind.EXISTENCE
    elif attribute and subject_known:
        # "where is Naveen from", "his email" — one fact about one person.
        intent.kind = QueryKind.ATTRIBUTE
        intent.attribute = attribute
        intent.skills, intent.corrections = [], []
    elif (
        subject_known
        and _asks_for_a_fact(text)
        and not is_profile
        and _attribute_words(text, intent)
    ):
        # A wh-question about a KNOWN person, answered from that person's own resume.
        #
        # It must actually ask something: "what is Priya Sharma" leaves nothing behind
        # once the name is removed, so that is a request for the profile.
        #
        # A technology in the sentence does NOT make this a pool search. "what has he
        # done with Azure" asks what THIS person did; ranking the whole pool by Azure
        # answers a question nobody asked. `_asks_for_a_fact` already excludes
        # "who has ..." phrasing, which is the genuine search form.
        intent.kind = QueryKind.ATTRIBUTE
        intent.attribute = None
    elif is_profile and (subject_known or intent.unresolved_name):
        # "tell me more about X" is a request for an overview; the words of the
        # request must never become requirements.
        intent.kind = QueryKind.PROFILE
        intent.skills, intent.corrections = [], []
    elif is_count and (intent.skills or intent.places):
        intent.kind = QueryKind.COUNT
    elif (intent.skills or intent.places or intent.rank_by
          or intent.min_years is not None or intent.max_years is not None):
        intent.kind = QueryKind.SKILL_SEARCH
    elif subject_known:
        intent.kind = QueryKind.PROFILE
    else:
        intent.kind = QueryKind.AMBIGUOUS

    # Nobody resolved, but the sentence is plainly about a person: name them as
    # unresolved so the answer can say who it could not find, and offer the closest
    # real candidates instead of quietly searching for something else.
    # A name that matched SEVERAL real people is resolved-but-ambiguous, not unknown.
    # Reporting it as "I couldn't find anyone called Naveen" when the pool holds two of
    # them is the opposite of the truth.
    if not intent.people and not intent.unresolved_name and not intent.name_ambiguous and intent.kind in (
        QueryKind.AMBIGUOUS, QueryKind.EXISTENCE, QueryKind.PROFILE, QueryKind.ATTRIBUTE
    ):
        if (phrase := _unresolved_person(text, intent, lexicon)):
            intent.unresolved_name = phrase
            intent.name_suggestions = lexicon.resolve_people(phrase).suggestions
            if intent.kind == QueryKind.ATTRIBUTE:
                intent.kind = QueryKind.AMBIGUOUS

    intent.underspecified = _is_underspecified(text, intent)

    if intent.kind == QueryKind.ATTRIBUTE:
        intent.attribute_probe = _attribute_probe(text, intent, lexicon)

    # A follow-up that narrows the previous result set inherits its constraints.
    if intent.restricts_to_previous and intent.kind == QueryKind.AMBIGUOUS and context.last_candidate_ids:
        intent.kind = QueryKind.SKILL_SEARCH

    if intent.kind == QueryKind.AMBIGUOUS:
        intent.unrecognised = _unrecognised_words(text, intent, lexicon)
        intent.clarification = _clarification_for(intent)

    probe = intent.skills or intent.place_labels or (
        [intent.people[0].name] if intent.people else [text]
    )
    intent.semantic_probe = " ".join(probe)
    return intent


def _unresolved_person(text: str, intent: QueryIntent, lexicon: CorpusLexicon) -> Optional[str]:
    """
    The name the recruiter typed that belongs to nobody in the pool.

    Two ways to spot one. Explicit grammar — "anyone named Nithin" — says outright that
    the following words are a name. Otherwise, in a sentence that is clearly ASKING
    ABOUT A PERSON, any content word left over once skills, places and function words
    are removed is the subject, and the subject is a person we do not have.

    Getting this right is what stops the assistant answering a question about a person
    it has never heard of by searching for whatever else was in the sentence.
    """
    if (m := _NAMED_AS_RE.search(text)):
        phrase = m.group(1).strip()
        if phrase and phrase.lower() not in FUNCTION_WORDS:
            return phrase

    if not _ASKS_ABOUT_PERSON.search(text):
        return None

    for raw in re.findall(r"[A-Za-z][A-Za-z.'-]+", text):
        low = raw.lower()
        if low in FUNCTION_WORDS or len(low) < 3:
            continue
        if low in lexicon.skill_terms or low in lexicon.rejected_terms:
            continue
        if any(low in place.split() or low == place for place in lexicon.places):
            continue
        if any(low == s.lower() for s in intent.skills):
            continue
        return raw
    return None


# Grammar that makes a message a REQUEST rather than a topic: an interrogative, an
# imperative, or a stated need. "who knows java" / "find me java people" / "I need java"
# all ask for something. A bare "java" does not.
_IS_A_REQUEST = re.compile(
    r"\b(?:who|which|what|whose|whom|where|when|how|any|anyone|anybody|is|are|does|do|"
    r"did|can|could|would|has|have|had)\b"
    r"|\b(?:find|show|give|get|list|search|need|want|looking|look|tell|compare|rank|"
    r"suggest|recommend|shortlist|filter|narrow)\b"
    r"|\?",
    re.I,
)


def _is_underspecified(text: str, intent: QueryIntent) -> bool:
    """
    True when the recruiter named a topic instead of asking a question.

    "java" is not the question "who knows java" — it is the start of one. Answering it
    with a ranked list picks a reading the recruiter never chose (best? most senior?
    project-proven? nearby?) and buries the four follow-ups they actually wanted. The
    honest move is to narrow it down together, with real counts.

    Deliberately narrow, so it never swallows a genuine search:
      * something must have resolved (a bare unknown word is AMBIGUOUS, handled above)
      * NO request grammar at all — no interrogative, imperative, or stated need
      * no second constraint, because two facets already express an intent
        ("java 5 years" is specific enough to answer)
    """
    if intent.kind not in (QueryKind.SKILL_SEARCH, QueryKind.COUNT):
        return False
    if (intent.min_years is not None or intent.max_years is not None
            or intent.people or intent.restricts_to_previous):
        return False
    # One facet only. "java aws" is still a topic, but "java in chennai" is a real query.
    if len(intent.skills) + len(intent.places) != 1:
        return False
    if _IS_A_REQUEST.search(text or ""):
        return False
    # Long prose without request grammar is unusual; treat it as a request anyway
    # rather than interrogating someone who clearly typed a sentence.
    return len((text or "").split()) <= 4


def _asks_for_a_fact(text: str) -> bool:
    """Wh- or possessive phrasing aimed at one detail rather than a shortlist."""
    if _POSSESSIVE.search(text):
        return True
    if not _WH_FACT.search(text):
        return False
    # "who has java" is a search, not a fact request.
    return not re.match(r"^\s*who(?:m)?\s+(?:has|have|knows?|worked)\b", text, re.I)


def _unrecognised_words(text: str, intent: QueryIntent, lexicon: CorpusLexicon) -> List[str]:
    """
    Content words that resolved to nothing at all.

    Reporting these is the difference between "I couldn't tell what you're asking" and
    "I don't recognise 'now' as a skill in your pool" — the second tells the recruiter
    exactly which word to change.
    """
    known = {s.lower() for s in intent.skills}
    known |= {w for label in intent.place_labels for w in label.lower().split()}
    known |= {t for person in intent.people for t in person.tokens}
    out: List[str] = []
    for raw in _TOKEN_RE.findall((text or "").lower()):
        if raw in FUNCTION_WORDS or raw in known or len(raw) < 3 or raw.isdigit():
            continue
        if raw not in out:
            out.append(raw)
    return out[:3]


def _clarification_for(intent: QueryIntent) -> str:
    """
    What to ask back when nothing resolved.

    Asking costs the recruiter one click. Guessing costs them their trust in every
    answer that follows, which is what happened when an unknown name was quietly
    matched against an unrelated candidate.
    """
    if intent.unresolved_name:
        return (
            f'I couldn\'t find anyone called "{intent.unresolved_name}" in the pool.'
        )
    if intent.unmatched_place:
        return (
            f"No resume in the pool mentions {intent.unmatched_place}. "
            f"Would you like me to search without the location, or try a nearby area?"
        )
    if intent.unrecognised:
        words = ", ".join(f'"{w}"' for w in intent.unrecognised)
        return (
            f"I don't recognise {words} as a skill, a place or a candidate in your pool, "
            f"so I'd only be guessing. Did you mean a technology, a location, or "
            f"someone's name?"
        )
    return (
        "I couldn't tell what you're asking for. I can search by skill "
        "(\"who knows Kubernetes?\"), by experience (\"10+ years in Java\"), by "
        "location, or answer a question about a specific candidate by name."
    )
