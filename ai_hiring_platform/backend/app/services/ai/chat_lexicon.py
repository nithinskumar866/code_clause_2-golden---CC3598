"""
Corpus-derived lexicons — what the *pool itself* says a word means.

WHY THIS MODULE EXISTS
----------------------
Every serious retrieval failure the assistant produced traces to one mistake: a word
in the recruiter's question was assigned a meaning the corpus does not support.

    "who has now experience"                 -> skill "Now"        -> 306 false matches
    "support your point why ..."             -> skill "Point"      -> 88% collapsed to 56%
    "best candidate who is in t nagar ..."   -> person "Candidate #124" (a placeholder
                                                name) -> the java requirement vanished
    "where is naveen k from"                 -> nobody, because the pool stores
                                                "K Naveen" and matching was positional

The old code answered "is this token a skill / a name / a place?" with regexes and a
harvested word list. That cannot work, because the same string is a technology in one
pool and ordinary prose in another. The question is only answerable **statistically,
against this corpus**.

So this module builds three lexicons once per corpus load, from the indexed chunk text
that is already in memory — no re-embedding, no new index, no maintained keyword list:

  SKILL TERMS     which words may act as a hard search requirement
  PLACES          which words name a location someone lives in
  PEOPLE          which words name a person, and *who*, order-independently

THE DISCRIMINATOR (skills)
--------------------------
A technology name is written as a proper noun: `Java`, `AWS`, `React`, `Kubernetes`.
An English word harvested by the "all-caps looks like an acronym" rule — `NOW`,
`POINT`, `WITHIN` — is written in lowercase virtually everywhere else in the same
corpus. So for every harvested term we measure the fraction of its occurrences that
are plain lowercase, and how many resumes it appears in. A term that is usually
lowercase *and* spread across much of the pool is ordinary English and can never be a
requirement. It costs one pass over the chunk text and generalises to any language,
any pool, any technology that has not been invented yet.

Terms in the shared `TECH_TAXONOMY` skip the test entirely — those are known
technologies by definition.

Golden Rule 4 is respected throughout: nothing here maps a skill to another skill or
enumerates technologies. The only fixed lists are English *function words* (grammar,
not domain knowledge) and the grammar of an address.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.constants import TECH_TAXONOMY
from app.core.logging import logger

# Canonical technology surface shared with the rest of the retrieval stack.
KNOWN_TECH: Dict[str, str] = {s.lower(): s for skills in TECH_TAXONOMY.values() for s in skills}

# --- Tuning -----------------------------------------------------------------
# A harvested term is ordinary English when it is usually written in lowercase AND
# occurs across a broad slice of the pool. Both conditions matter: a genuinely new
# framework written in lowercase in three resumes must stay searchable.
_ENGLISH_LOWER_FRACTION = 0.80
_ENGLISH_SPREAD_FRACTION = 0.10   # present in >10% of resumes
_MIN_CORPUS_TERM_LEN = 3          # corpus-harvested terms only; taxonomy is exempt

# How many different resumes must show a term inside an address region before it is
# read as a place on that evidence alone. One is not enough in a real pool: any line
# carrying a five-digit number looks address-shaped, which is how "loss ratio",
# "defect sheets" and "including test cases" all turned up as candidate locations.
#
# Corroboration only means something when there is something to corroborate against,
# so below _SMALL_POOL a single sighting stands — in a five-resume pool it is the
# strongest evidence the corpus can offer, and demanding a second would leave the
# gazetteer empty.
_MIN_OBSERVED_PLACE_RESUMES = 2
_SMALL_POOL = 25

# Longest run of capitals still readable as an acronym rather than a shouted word.
# AWS, SQL, API, HTML, REST and JSON fit; CHENNAI, DEVELOPER and SCIENCE do not.
# Taxonomy terms are exempt — they are admitted before this check ever runs.
_MAX_ACRONYM_LEN = 5


def _is_acronym(surface: str) -> bool:
    """Written in capitals and nothing but letters — HR, QA, ML, AWS, SQL."""
    s = (surface or "").strip()
    return bool(s) and s.isupper() and s.isalpha() and 2 <= len(s) <= _MAX_ACRONYM_LEN

# Morphologically technical: a name that announces itself as a technology whatever
# line it was found on, so it can never be read as a location.
#
# The acronym branch is deliberately bounded to five letters. An unbounded `[A-Z]{2,}`
# also matches a *shouted* word, and resumes shout their cities — "CHENNAI" would have
# been classified as technology and the whole fix would have inverted.
_MORPH_TECH = re.compile(
    r"[a-z][A-Z0-9]"        # internal capital or digit: BigQuery, PyTorch, log4j
    r"|[A-Z]{2,}[a-z]"      # capital run then lowercase: JUnit, IPython
    r"|\b[A-Z]{2,5}\b"      # short acronym: AWS, SQL, CI  (never a 7-letter city)
    r"|[0-9]"               # any digit at all: S3, EC2
    r"|[+#/]"               # tech punctuation: C++, C#, CI/CD
)

# Typo tolerance for skills. 0.82 accepts javaa->java, pythn->python while keeping
# java and javascript distinct.
_SKILL_TYPO_RATIO = 0.82
_MIN_TYPO_LEN = 4

# Name matching. A surname typo ("naven" -> "naveen") should still be offered as a
# suggestion, but never silently searched.
_NAME_FUZZY_RATIO = 0.84
_NAME_SUGGEST_RATIO = 0.68

# Resumes whose name could not be extracted are stored as "Candidate #17". Those are
# NOT names, and matching them against the ordinary English word "candidate" is what
# hijacked skill searches into profile lookups.
_PLACEHOLDER_NAME = re.compile(r"^\s*(?:candidate|resume|unknown)\s*#?\s*\d*\s*$", re.I)

# Case-sensitive token scan used for the English-vs-technology measurement.
_SURFACE_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#.]*")
# Lowercase token scan used for query parsing.
_QUERY_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#./-]*")

# English FUNCTION words — grammar, not domain knowledge. A word here can never be a
# skill, a place or a person, in any pool. Deliberately closed-class: determiners,
# pronouns, prepositions, conjunctions, auxiliaries, wh-words, and the handful of
# recruiting verbs/nouns that frame a request rather than name a requirement.
FUNCTION_WORDS: Set[str] = {
    # determiners / quantifiers
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "all", "each",
    "every", "both", "few", "many", "much", "more", "most", "less", "least", "several",
    "other", "others", "another", "such", "no", "none", "one", "ones", "two", "three",
    # pronouns
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their",
    "theirs", "who", "whom", "whose", "which", "what", "someone", "somebody", "anyone",
    "anybody", "everyone", "everybody", "himself", "herself", "themselves", "themself",
    # prepositions / conjunctions
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "within", "into",
    "onto", "by", "about", "above", "below", "under", "over", "between", "among",
    "through", "throughout", "during", "before", "after", "since", "until", "till",
    "while", "and", "or", "but", "nor", "so", "than", "then", "as", "if", "because",
    "though", "although", "whether", "against", "per", "via", "upon", "across",
    # auxiliaries / common verbs of request
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "done", "have", "has", "had", "having", "can", "could", "will", "would", "shall",
    "should", "may", "might", "must", "need", "needs", "needed", "want", "wants",
    "get", "gets", "got", "give", "gives", "let", "make", "makes", "made", "say",
    "says", "said", "tell", "tells", "told", "show", "shows", "shown", "find", "finds",
    "look", "looking", "search", "searching", "list", "listing", "know", "knows",
    "knowing", "known", "think", "thinks", "go", "goes", "come", "comes", "take",
    "takes", "put", "puts", "use", "uses", "used", "using", "work", "works",
    # discourse / time
    "now", "then", "here", "there", "when", "where", "why", "how", "again", "also",
    "just", "only", "still", "yet", "already", "ever", "never", "always", "sometimes",
    "today", "yesterday", "tomorrow", "currently", "current", "present", "recent",
    "recently", "ago", "soon", "later", "next", "previous", "last", "first", "second",
    "third", "fourth", "fifth", "very", "really", "quite", "rather", "too", "well",
    "good", "better", "best", "great", "bad", "worse", "worst", "new", "old", "same",
    "different", "sure", "okay", "ok", "yes", "yeah", "yep", "no", "not", "nope",
    "please", "thanks", "thank", "hi", "hello", "hey",
    # recruiting scaffolding — frames the request, never names a requirement
    "candidate", "candidates", "applicant", "applicants", "resume", "resumes", "cv",
    "cvs", "profile", "profiles", "person", "people", "guy", "guys", "someone",
    "experience", "experienced", "experiences", "year", "years", "yr", "yrs", "exp",
    "skill", "skills", "skilled", "expertise", "expert", "proficient", "proficiency",
    "knowledge", "background", "history", "detail", "details", "summary", "summarise",
    "summarize", "overview", "info", "information", "brief", "elaborate", "explain",
    "point", "points", "part", "parts", "level", "levels", "type", "types", "kind",
    "job", "jobs", "role", "roles", "position", "positions", "hire", "hiring", "hired",
    "recruit", "recruiting", "match", "matches", "matching", "fit", "fits", "suitable",
    "rank", "ranking", "compare", "comparison", "versus", "vs", "difference",
    "top", "bottom", "strong", "strongest", "strength", "weak", "weakest", "senior",
    "junior", "mid", "fresher", "lead", "leads", "leader", "leading",
    "project", "projects", "education", "degree", "company", "companies", "client",
    "clients", "team", "teams", "organization", "organisation",
}

# Sections that describe capability. Header/Summary/Education prose is dense with
# names, degrees and addresses that morphology mistakes for technology.
_SKILL_BEARING_SECTIONS = ("Skills", "Experience", "Projects", "Certifications")

# --- Address grammar (not a gazetteer) --------------------------------------
# A place is recognised by the SHAPE of the line it sits in — a postal code, an
# explicit "address" label, or the comma-separated tail of a contact block. Which
# actual places exist is then read off the resumes themselves, so the pool is its own
# gazetteer and a new city needs no code change.
_POSTAL = re.compile(r"\b\d{5,6}\b")
_ADDRESS_LABEL = re.compile(r"\b(?:address|located|location|based\s+(?:in|at)|residing|resident)\b", re.I)
_PLACE_PART = re.compile(r"^[A-Za-z][A-Za-z.'\-]*(?:\s+[A-Za-z][A-Za-z.'\-]*){0,2}$")
# Words that mean the capitalised phrase is an organisation or a heading, not a place.
_NOT_A_PLACE = re.compile(
    r"\b(?:college|university|institute|school|academy|ltd|limited|pvt|private|inc|"
    r"corp|corporation|technologies|technology|solutions|services|systems|consultancy|"
    r"software|resume|curriculum|vitae|profile|summary|objective|experience|education|"
    r"skills|projects?|certifications?|email|phone|mobile|contact|linkedin|github)\b",
    re.I,
)
# Locative phrasing in a QUESTION: "in t nagar", "from chennai", "based in pune",
# "candidates around velachery". Used to notice that the recruiter named a place we
# do not have, instead of silently dropping the constraint.
LOCATIVE_PHRASE = re.compile(
    r"\b(?:in|from|at|near|around|based\s+in|located\s+in|living\s+in|resident\s+of|"
    r"belongs?\s+to|belonging\s+to)\s+"
    r"([A-Za-z][A-Za-z.'\-]*(?:\s+[A-Za-z][A-Za-z.'\-]*){0,2})",
    re.I,
)


def normalise_name_token(token: str) -> str:
    """Lowercase, strip the punctuation people scatter through names (`S.` -> `s`)."""
    return re.sub(r"[^a-z0-9]", "", (token or "").lower())


def name_tokens(name: str) -> List[str]:
    """Order-independent token view of a person's name."""
    return [t for t in (normalise_name_token(p) for p in (name or "").split()) if t]


def is_placeholder_name(name: Optional[str]) -> bool:
    """True for the synthetic `Candidate #17` used when extraction found no name."""
    return bool(_PLACEHOLDER_NAME.match(name or ""))


@dataclass
class Person:
    """One identifiable human in the pool."""

    resume_id: int
    name: str
    tokens: List[str]
    title: Optional[str] = None
    total_years: Optional[float] = None
    location: Optional[str] = None

    @property
    def significant_tokens(self) -> List[str]:
        """Tokens long enough to identify someone. Single initials are not."""
        return [t for t in self.tokens if len(t) > 1]

    @property
    def initials(self) -> str:
        return "".join(t[0].upper() for t in self.tokens if t)


@dataclass
class NameResolution:
    """
    Outcome of looking for people in a question.

    `ambiguous` is the case the old code had no concept of: several real people answer
    to what was typed. The honest response is to ask which one, not to pick.
    """

    matched: List[Person] = field(default_factory=list)
    ambiguous: bool = False
    # What the recruiter typed that looks like a name but resolved to nobody.
    unresolved: Optional[str] = None
    suggestions: List[Person] = field(default_factory=list)
    # The token(s) that produced the match, for "I read 'naveen k' as 'K Naveen'".
    surface: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return bool(self.matched) and not self.ambiguous


class CorpusLexicon:
    """
    Everything the pool's own text says about words, built once and cached.

    Construction is a single pass over the indexed chunks that are already in memory,
    so it adds no I/O and no embedding work to a corpus load.
    """

    def __init__(self, chunks: Sequence[Dict[str, Any]], profiles: Sequence[Dict[str, Any]]) -> None:
        self._resume_count = max(len({c["resume_id"] for c in chunks}), 1)
        self.skill_terms: Dict[str, str] = dict(KNOWN_TECH)
        self.rejected_terms: Set[str] = set()
        # resume ids that list each term as a capability — the skill side of the
        # skill-versus-place adjudication in _build_places.
        self._skill_declarations: Dict[str, Set[int]] = defaultdict(set)
        self.places: Dict[str, Set[int]] = {}
        self.place_display: Dict[str, str] = {}
        self.people: List[Person] = []
        self._people_by_token: Dict[str, List[Person]] = defaultdict(list)
        self._all_name_tokens: Set[str] = set()

        self._build_word_statistics(chunks)
        self._build_skill_terms(profiles)
        self._build_places(chunks, profiles)
        self._build_people(profiles)

        logger.info(
            "Corpus lexicon: %d skill terms (%d harvested words rejected as ordinary "
            "English), %d places, %d identifiable people.",
            len(self.skill_terms), len(self.rejected_terms), len(self.places), len(self.people),
        )

    # --- word statistics ----------------------------------------------------
    def _build_word_statistics(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """
        Per term: how often it is written in plain lowercase, and how many resumes it
        occurs in. These two numbers are what separate `Java` from `now`.
        """
        total: Counter = Counter()
        lowercase: Counter = Counter()
        resumes: Dict[str, Set[int]] = defaultdict(set)

        for chunk in chunks:
            rid = chunk["resume_id"]
            for match in _SURFACE_TOKEN.finditer(chunk.get("text") or ""):
                surface = match.group(0)
                key = surface.lower()
                total[key] += 1
                if surface.islower():
                    lowercase[key] += 1
                resumes[key].add(rid)

        self._occurrences = total
        self._lowercase = lowercase
        self._document_frequency = {term: len(ids) for term, ids in resumes.items()}

    def _reads_as_english(self, term: str) -> bool:
        """
        True when this pool writes the term like an ordinary word rather than a name.

        Two independent signals must agree, so a lowercase-written niche technology in
        a handful of resumes is not thrown away.
        """
        key = term.lower()
        occurrences = self._occurrences.get(key, 0)
        if occurrences < 3:
            return False                       # too rare to judge; let it through
        lower_fraction = self._lowercase.get(key, 0) / occurrences
        spread = self._document_frequency.get(key, 0) / self._resume_count
        return lower_fraction >= _ENGLISH_LOWER_FRACTION and spread >= _ENGLISH_SPREAD_FRACTION

    # --- skills -------------------------------------------------------------
    def _build_skill_terms(self, profiles: Sequence[Dict[str, Any]]) -> None:
        """
        Admit a harvested term as a searchable requirement only if the corpus supports
        reading it as a technology. Known taxonomy terms are always admitted.
        """
        for profile in profiles:
            rid = profile.get("resume_id")
            for raw in profile.get("skills") or []:
                key = (raw or "").lower()
                if not key:
                    continue
                # How many resumes DECLARE this word as a capability. Weighed against
                # address evidence when a term looks like both a skill and a city.
                if rid is not None:
                    self._skill_declarations[key].add(int(rid))
                if key in self.skill_terms:
                    # The taxonomy stores terms in lowercase; resumes write them the way
                    # the industry does ("Java", "AWS", "React"). Prefer the corpus's
                    # capitalisation so the recruiter sees a technology name, not a word.
                    if self.skill_terms[key].islower() and not raw.islower():
                        self.skill_terms[key] = raw
                    continue
                # Two-letter terms are usually noise, but a whole class of real
                # requirements is exactly two letters — HR, QA, ML, AI, BI, UX. The
                # blanket three-character floor silently dropped every one of them, so
                # "communication skill for hr role" searched for communication ALONE
                # and returned five people with no HR background at 100%.
                #
                # An acronym announces itself by being written in capitals; ordinary
                # two-letter English is lowercase and already in FUNCTION_WORDS.
                too_short = len(key) < _MIN_CORPUS_TERM_LEN and not _is_acronym(raw)
                if too_short or key in FUNCTION_WORDS:
                    self.rejected_terms.add(key)
                    continue
                # An acronym is SHORT; a shouted word is long. Resumes capitalise
                # headings, cities and job titles, so the harvester's "run of capitals"
                # rule admitted CHENNAI, DEVELOPER, SCIENCE and ABILITIES as
                # technologies — which is why "react developer in chennai" was read as
                # three skill requirements and matched everyone at 88% depth.
                #
                # Applied here as well as at harvest time because profiles are cached
                # with the index: this runs on every load, so an existing store is
                # cleaned without re-embedding 305 resumes.
                if raw.isupper() and len(key) > _MAX_ACRONYM_LEN:
                    self.rejected_terms.add(key)
                    continue
                if self._reads_as_english(key):
                    self.rejected_terms.add(key)
                    continue
                self.skill_terms[key] = raw

    def resolve_skill(self, token: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Map one query token to a real skill.

        Returns `(canonical, corrected_from)`; `corrected_from` is set only when a typo
        was fixed, so the answer can state the assumption rather than silently search
        for something else.
        """
        low = (token or "").lower()
        if not low or low in FUNCTION_WORDS or low in self.rejected_terms:
            return None, None
        if low in self.skill_terms:
            return self.skill_terms[low], None
        # Typo correction runs against the TAXONOMY only. Correcting toward corpus
        # words is what turned the name "aravind" into the dairy brand "AAVIN".
        if len(low) >= _MIN_TYPO_LEN:
            close = difflib.get_close_matches(low, KNOWN_TECH.keys(), n=1, cutoff=_SKILL_TYPO_RATIO)
            if close:
                return KNOWN_TECH[close[0]], token
        return None, None

    # --- places -------------------------------------------------------------
    def _build_places(
        self, chunks: Sequence[Dict[str, Any]], profiles: Sequence[Dict[str, Any]]
    ) -> None:
        """
        Harvest the places this pool's people actually live in.

        Only address-shaped lines are read — a postal code, an explicit address label,
        or the comma-separated contact tail — so company names and project locations in
        body prose do not become residences.
        """
        # Two grades of evidence, kept apart because they are not equally trustworthy.
        #
        #   DECLARED — the term sits in a candidate's extracted `location` field. That
        #              field exists to hold a place, so anything in it is one.
        #   OBSERVED — the term sits in an address-shaped region of some line. Real
        #              addresses land here, but so does anything on a line that happens
        #              to carry a five-digit number, which is why one sighting is not
        #              enough to call something a city.
        declared: Dict[str, Set[int]] = defaultdict(set)
        observed: Dict[str, Set[int]] = defaultdict(set)
        display: Dict[str, str] = {}

        for profile in profiles:
            rid = profile.get("resume_id")
            if rid is None:
                continue
            for part in self._split_address(profile.get("location") or ""):
                declared[part.lower()].add(int(rid))
                display.setdefault(part.lower(), part)

        for chunk in chunks:
            text = chunk.get("text") or ""
            for line in text.split("\n"):
                for part in self._split_address(self._address_region(line)):
                    observed[part.lower()].add(chunk["resume_id"])
                    display.setdefault(part.lower(), part)

        for key in set(declared) | set(observed):
            if not self._could_be_a_place(key, display.get(key, key)):
                continue
            here = declared.get(key, set())
            there = observed.get(key, set())
            as_skill = len(self._skill_declarations.get(key, ()))

            # The `location` field is not as trustworthy as its name suggests: the
            # upstream extractor writes a skills line into it when a CV has no address
            # ("JUnit, Mockito" is a real value in this pool). So a declared location
            # still has to beat how often the corpus calls the same word a skill,
            # unless a separate address line corroborates it.
            declared_stands = bool(here) and (bool(there) or len(here | there) > as_skill)
            # Observed-only evidence needs a second resume AND has to outweigh the
            # skill reading, or a library sitting beside one address becomes a city.
            needed = (
                1 if self._resume_count < _SMALL_POOL else _MIN_OBSERVED_PLACE_RESUMES
            )
            observed_stands = len(there) >= needed and len(there) > as_skill
            if declared_stands or observed_stands:
                self.places[key] = here | there
                self.place_display.setdefault(key, display.get(key, key))

        # A word cannot be both. Whatever the corpus places, it no longer offers as a
        # searchable skill — otherwise "location bangalore" is answered by scoring
        # people on how well they *know* Bangalore.
        for key in self.places:
            self.skill_terms.pop(key, None)

    def _could_be_a_place(self, term: str, surface: str = "") -> bool:
        """
        Whether a term is eligible to be read as a location at all.

        Two exclusions, both about telling a technology from a city when both appear
        on address-shaped lines — which real CVs do constantly:

            Languages: Java, Python, SQL | Chennai 600017

        The address grammar cannot separate them, because the line genuinely *is* an
        address line. Without this guard "Java" became a location, and a search for
        Java developers matched on residence, dropped the skill, and scored everyone 0%.
        """
        low = (term or "").lower()
        if not low:
            return False
        words = low.split()
        # The taxonomy is authoritative about technologies.
        if low in KNOWN_TECH or (len(words) > 1 and all(w in KNOWN_TECH for w in words)):
            return False
        # So is shape: JUnit, BigQuery, C++, S3 announce themselves as technology no
        # matter what line they were found on. Tested against the corpus's own
        # spelling, since `low` has already been folded to lowercase.
        return not _MORPH_TECH.search(surface or term or "")

    def _is_a_technology(self, term: str) -> bool:
        """
        A technology is never a place. Adjudication has already removed genuine place
        names from the skill vocabulary, so anything still in it is a skill.
        """
        low = (term or "").lower()
        if low in self.skill_terms or low in KNOWN_TECH:
            return True
        words = low.split()
        return len(words) > 1 and all(w in self.skill_terms or w in KNOWN_TECH for w in words)

    def _address_region(self, line: str) -> str:
        """
        The part of a line that is actually an address — not the whole line.

        A CV contact line routinely carries both a skills list and a city:

            Languages: Java, Python, SQL | Chennai 600017

        Reading the whole line as an address turned Python and SQL into places. An
        address has recognisable position, not just recognisable shape: it sits AFTER
        an explicit label, or in the pipe-separated segment that carries the postal
        code, and a place name is one of the last few comma-parts before that code.
        Narrowing to that region is what separates the city from the skills beside it.
        """
        if not line:
            return ""
        if (m := _ADDRESS_LABEL.search(line)):
            return line[m.end():]

        segments = re.split(r"\s*[|·•]\s*", line)
        with_postal = [s for s in segments if _POSTAL.search(s)]
        if not with_postal:
            return ""
        segment = with_postal[-1]
        # A place is one of the last few comma-parts before the code, never the whole
        # of a long line that happens to end in one.
        parts = re.split(r"\s*,\s*", segment)
        return ", ".join(parts[-3:])

    def _split_address(self, line: str) -> List[str]:
        """Break an address line into the place names it contains."""
        out: List[str] = []
        # Strip the label, postal code and street-number noise before splitting.
        cleaned = _ADDRESS_LABEL.sub(" ", line or "")
        cleaned = _POSTAL.sub(" ", cleaned)
        for raw in re.split(r"[,|;/–—\-]+", cleaned):
            part = re.sub(r"\s+", " ", raw).strip(" .:-")
            if not part or len(part) < 3 or len(part) > 40:
                continue
            if "@" in part or any(ch.isdigit() for ch in part):
                continue
            if not _PLACE_PART.match(part) or _NOT_A_PLACE.search(part):
                continue
            words = part.split()
            # A place is a proper noun: at least one capitalised word, and not a run of
            # ordinary words that merely fits the shape.
            if not any(w[:1].isupper() for w in words):
                continue
            if all(w.lower() in FUNCTION_WORDS for w in words):
                continue
            out.append(" ".join(words))
        return out

    def find_places(self, query: str) -> List[str]:
        """
        Place names from the pool that appear in the question, longest first.

        A term that is also a technology is never read as a place here either, even if
        one slipped into the gazetteer from an older index: "java candidate" asks for a
        skill, and answering it with people who *live* somewhere called Java is not a
        near miss, it is a different question.
        """
        lowered = f" {re.sub(r'[^a-z0-9 ]+', ' ', (query or '').lower())} "
        lowered = re.sub(r"\s+", " ", lowered)
        found: List[str] = []
        for key in sorted(self.places, key=len, reverse=True):
            if self._is_a_technology(key):
                continue
            if f" {key} " in lowered and not any(key in seen for seen in found):
                found.append(key)
        return found

    def resumes_at(self, place_keys: Iterable[str]) -> Set[int]:
        """Resume ids mentioning any of these places."""
        out: Set[int] = set()
        for key in place_keys:
            out |= self.places.get(key, set())
        return out

    def unmatched_place_phrase(self, query: str, matched: Sequence[str]) -> Optional[str]:
        """
        A place the recruiter named that nobody's resume mentions.

        Detected from locative grammar ("in t nagar", "from chennai") so the assistant
        can say "no resume mentions T. Nagar" instead of quietly ignoring the
        constraint and answering a question that was never asked.
        """
        if matched:
            return None
        for match in LOCATIVE_PHRASE.finditer(query or ""):
            phrase = re.sub(r"\s+", " ", match.group(1)).strip(" .,")
            words = [w for w in phrase.split() if w]
            # Trim trailing scaffolding: "in t nagar and knows java" -> "t nagar".
            kept: List[str] = []
            for word in words:
                if word.lower() in FUNCTION_WORDS or word.lower() in self.skill_terms:
                    break
                kept.append(word)
            if not kept:
                continue
            phrase = " ".join(kept)
            if len(phrase) < 3 or phrase.lower() in FUNCTION_WORDS:
                continue
            # A known skill or a person is not a mislaid place.
            if phrase.lower() in self.skill_terms:
                continue
            if any(normalise_name_token(w) in self._all_name_tokens for w in kept):
                continue
            return phrase
        return None

    # --- people -------------------------------------------------------------
    def _build_people(self, profiles: Sequence[Dict[str, Any]]) -> None:
        for profile in profiles:
            name = profile.get("name") or ""
            rid = profile.get("resume_id")
            if rid is None or is_placeholder_name(name):
                continue                        # a placeholder is not a name
            tokens = name_tokens(name)
            if not tokens:
                continue
            person = Person(
                resume_id=int(rid),
                name=name,
                tokens=tokens,
                title=profile.get("title"),
                total_years=profile.get("total_years"),
                location=profile.get("location"),
            )
            self.people.append(person)
            for token in person.significant_tokens:
                self._people_by_token[token].append(person)
                self._all_name_tokens.add(token)

    def is_name_token(self, token: str) -> bool:
        return normalise_name_token(token) in self._all_name_tokens

    def resolve_mentions(self, query: str) -> List[NameResolution]:
        """
        Every person the question mentions — not just the best-scoring one.

        `resolve_people` answers "who is this about", which is the wrong question for
        "compare Najam Uddin and K Naveen": scoring picks the single best match, so
        Najam Uddin (two token hits) wins and K Naveen (one) is silently discarded.
        The comparison then runs against one person and reports 0%.

        The fix is to group by WHICH TYPED WORDS each person accounts for, rather than
        ranking people globally. Each distinct group of claimed tokens is one mention:

            "compare najam uddin and k naveen"
                {najam, uddin} -> Najam Uddin        (one mention, resolved)
                {naveen}       -> K Naveen           (one mention, resolved)

            "why anita is better than hemankshree"
                {anita}        -> Anita Iyer, Anita Patel   (ambiguous — ask)
                {hemankshree}  -> Hemankshree Tank          (resolved)

        That second case is why the old code asked "3 people answer to that name?" for a
        question that named two different people: it pooled everyone who tied on one hit
        into a single ambiguity. Ambiguity is now per NAME, not per question.

        Larger claims are consumed first, so a recruiter who typed both name parts gets
        the two-token match rather than everyone sharing the first name.
        """
        tokens = self._query_name_tokens(query)
        if not tokens:
            return []

        token_set = set(tokens)
        claims: Dict[frozenset, List[Person]] = {}
        for person in self.people:
            hit = set(person.significant_tokens) & token_set
            if hit:
                claims.setdefault(frozenset(hit), []).append(person)

        if not claims:
            suggestion = self._suggest_people(tokens)
            return [suggestion] if suggestion.unresolved else []

        mentions: List[NameResolution] = []
        consumed: Set[str] = set()
        for claimed in sorted(claims, key=len, reverse=True):
            if claimed <= consumed:
                continue
            people = claims[claimed]
            # Among people claiming the same words, prefer whoever the extra tokens
            # single out: "naveen k" separates "K Naveen" from "Naveen Kumar S".
            best_extra = max(len(set(p.tokens) & token_set) for p in people)
            winners = [p for p in people if len(set(p.tokens) & token_set) == best_extra]
            mentions.append(NameResolution(
                matched=winners,
                ambiguous=len(winners) > 1,
                surface=" ".join(sorted(claimed)) or None,
            ))
            consumed |= claimed
        return mentions

    def resolve_people(self, query: str) -> NameResolution:
        """
        Find who the question is about — order-independently, and honest about doubt.

        Matching is on token SETS, so "naveen k", "k naveen" and "Naveen K." all reach
        the same person. A token that is a function word, a known skill or a place can
        never contribute, which is what stops the literal word "candidate" from
        matching every resume whose name extraction failed.

        Three outcomes:
          * exactly one person            -> resolved
          * several equally good people   -> ambiguous, caller must ask which
          * a name-shaped word matching nobody -> unresolved + closest real people
        """
        tokens = self._query_name_tokens(query)
        if not tokens:
            return NameResolution()

        token_set = set(tokens)
        scored: List[Tuple[Tuple[int, int], Person]] = []
        for person in self.people:
            significant = set(person.significant_tokens)
            if not significant:
                continue
            hits = significant & token_set
            if not hits:
                continue
            # PRIMARY: how much of what the recruiter typed this person accounts for.
            # SECONDARY: initials too, so "naveen k" separates "K Naveen" from
            # "Naveen Kumar S" while a bare "naveen" leaves them tied.
            #
            # Scoring on how COMPLETELY the person's own name was covered instead
            # would silently prefer whoever has the shortest name — "naveen" would
            # pick "K Naveen" over "Naveen Kumar S" for no reason the recruiter gave.
            scored.append(((len(hits), len(set(person.tokens) & token_set)), person))

        if scored:
            best = max(score for score, _ in scored)
            winners = [p for score, p in scored if score == best]
            surface = " ".join(t for t in tokens if any(t in p.tokens for p in winners))
            # Two people the recruiter's words cannot separate is a question to ask,
            # not a ranking to guess at — including two resumes for the same name,
            # which the recruiter still needs to choose between.
            return NameResolution(
                matched=winners,
                ambiguous=len(winners) > 1,
                surface=surface or None,
            )

        # Nothing matched. Is any leftover word name-SHAPED enough to be worth asking
        # about? Only then do we offer "did you mean ...".
        return self._suggest_people(tokens)

    def _query_name_tokens(self, query: str) -> List[str]:
        """Query tokens that could plausibly be part of a person's name."""
        out: List[str] = []
        for raw in _QUERY_TOKEN.findall((query or "").lower()):
            token = normalise_name_token(raw)
            if not token or token.isdigit():
                continue
            if token in FUNCTION_WORDS or raw in FUNCTION_WORDS:
                continue
            if token in self.skill_terms or raw in self.skill_terms:
                continue
            out.append(token)
        return out

    def _suggest_people(self, tokens: Sequence[str]) -> NameResolution:
        """Closest real people to a name that matched nobody."""
        best_ratio = 0.0
        best_token = None
        suggestions: List[Person] = []

        for token in tokens:
            if len(token) < 3:
                continue
            close = difflib.get_close_matches(
                token, list(self._people_by_token), n=4, cutoff=_NAME_SUGGEST_RATIO
            )
            if not close:
                continue
            ratio = difflib.SequenceMatcher(None, token, close[0]).ratio()
            if ratio > best_ratio:
                best_ratio, best_token = ratio, token
                seen: Set[int] = set()
                suggestions = []
                for match in close:
                    for person in self._people_by_token[match]:
                        if person.resume_id not in seen:
                            seen.add(person.resume_id)
                            suggestions.append(person)

        if best_token is None:
            return NameResolution()
        # A very close single match is a typo we can act on with the recruiter's
        # consent; anything looser is only ever a suggestion.
        return NameResolution(unresolved=best_token, suggestions=suggestions[:5])


def build_lexicon(
    chunks: Sequence[Dict[str, Any]], profiles: Sequence[Dict[str, Any]]
) -> CorpusLexicon:
    return CorpusLexicon(chunks, profiles)
