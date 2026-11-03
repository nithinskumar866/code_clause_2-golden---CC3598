"""
Text matching primitives shared by the Match Score parameters.

Every parameter in the Match Score compares something the JD asks for against
something the resume shows — a skill, a job title, a degree, a city, an industry.
Those comparisons must be *reproducible*: the same pair of strings has to produce
the same number on every machine and every run, or three people implementing the
same specification can never reconcile their results.

So the primitives here are pure string algorithms. No embeddings, no model, no
network, no clock. Semantic equivalence (the JS ↔ JavaScript case) is layered on
top of these by `match_scoring_service`, which calls `skill_semantics_service`
only when the lexical layer has already failed to decide.

Golden Rule 4 holds: nothing here maps a specific skill to another specific skill.
Normalisation is a *rule* about how technology names are written — `React.js`,
`ReactJS` and `react js` are the same token under any spelling convention — and it
generalises to names nobody has entered yet.
"""
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, List, Set

# Suffixes that decorate a technology name without changing which technology it is.
# This is orthography, not a skill map: it is the same rule for `VueJS`, `NestJS`
# and for a framework released tomorrow.
_FRAMEWORK_SUFFIX = re.compile(r"(?:\.js|js|\.net|framework|library|lang)$")

# Punctuation that separates words rather than belonging to them.
_SEPARATOR = re.compile(r"[\s/\\,;:_\-–—()\[\]{}|+&]+")

# Characters that survive normalisation because they are part of real names:
# `c++`, `c#`, `node.js`, `ci/cd`. Everything else is dropped.
_KEEP = re.compile(r"[^a-z0-9+#.]")

# Grammatical filler that carries no matching signal in a title, degree or industry
# phrase. A closed class of English function words, not domain vocabulary.
STOP_WORDS: Set[str] = {
    "a", "an", "and", "the", "of", "in", "on", "at", "to", "for", "with", "or",
    "as", "by", "from", "into", "is", "are", "be", "been", "we", "you", "our",
    "your", "their", "its", "this", "that", "these", "those", "will", "shall",
    "must", "should", "any", "all", "other", "etc", "including", "such",
}


def normalise(term: str) -> str:
    """
    Reduce a term to its comparison key.

    `React.js`, `ReactJS`, `react-js` and `REACT` all become `react`; `Node.js` and
    `NodeJS` both become `node`. Accents are folded, so `Español` matches `Espanol`.
    Returns "" for anything that normalises away entirely.
    """
    if not term:
        return ""
    folded = unicodedata.normalize("NFKD", term).encode("ascii", "ignore").decode("ascii")
    low = _KEEP.sub("", folded.lower().strip())
    if not low:
        return ""
    # Never strip a suffix down to nothing: `js` on its own must stay `js`.
    stripped = _FRAMEWORK_SUFFIX.sub("", low)
    return stripped or low


def tokens(text: str) -> List[str]:
    """Meaningful lowercase words in a phrase, with grammatical filler removed."""
    parts = _SEPARATOR.split((text or "").lower())
    out: List[str] = []
    for p in parts:
        w = _KEEP.sub("", p).strip(".")
        if w and w not in STOP_WORDS and len(w) > 1:
            out.append(w)
    return out


def ratio(a: str, b: str) -> float:
    """Character-level similarity of two normalised terms, 0.0-1.0."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def is_misspelling(a: str, b: str, threshold: float = 0.85) -> bool:
    """
    Whether two terms differ only by a typo.

    Short terms are excluded deliberately: `go` and `god`, or `r` and `c`, sit above
    any sensible character-similarity threshold while naming completely different
    things. Requiring four characters keeps `pyton`→`python` and `javscript`→
    `javascript` while refusing to guess at two-letter tokens.
    """
    if len(a) < 4 or len(b) < 4:
        return False
    if abs(len(a) - len(b)) > 3:
        return False
    return ratio(a, b) >= threshold


# Character similarity below this is noise, not evidence. Two unrelated job titles
# ("Graphic Designer" vs "Backend Engineer") share enough letters to score around
# 0.5 on raw character overlap while having nothing to do with each other, so the
# character signal is only allowed to speak when it is claiming near-identity.
_NEAR_IDENTICAL = 0.80


def phrase_similarity(a: str, b: str) -> float:
    """
    How close two multi-word phrases are, 0.0-1.0 — used for job titles, degrees,
    industries and locations, where an exact string match almost never happens.

    Token containment is the primary signal: what share of the shorter phrase's
    words appear in the longer one. `Senior Backend Engineer` vs `Backend Engineer`
    scores 1.0, because every word the JD used is present. This is what makes titles
    work — seniority adjectives and company decoration are not a mismatch.

    Whole-phrase character similarity is a *rescue* for word-order differences and
    typos, and it only applies when it is claiming the phrases are near-identical.
    Taking the plain maximum of the two signals was wrong: it let letter overlap
    between unrelated phrases manufacture a score that no shared word supported.

    Word-level typos are tolerated inside the containment count, so `Sofware
    Engineer` still matches `Software Engineer`.
    """
    a_tokens, b_tokens = tokens(a), tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0

    short, long = (a_tokens, b_tokens) if len(a_tokens) <= len(b_tokens) else (b_tokens, a_tokens)
    long_norm = [normalise(t) for t in long]
    hits = 0
    for t in short:
        n = normalise(t)
        if n in long_norm or any(is_misspelling(n, l) for l in long_norm):
            hits += 1
    containment = hits / len(short)

    whole = ratio(normalise("".join(a_tokens)), normalise("".join(b_tokens)))
    return max(containment, whole) if whole >= _NEAR_IDENTICAL else containment


# --- Academic qualifications ------------------------------------------------
# The grammar of how degrees are written. A closed vocabulary of AWARD NAMES, not a
# mapping between degrees and jobs, and shared by the JD and resume extractors so
# the two can never disagree about what counts as a qualification.
#
# Split by case-sensitivity on purpose. `B.E` and `M.E` are real degrees; `be` and
# `me` are ordinary English words, and matching them case-insensitively without a
# dot turned the sentence "It will be fun" into an education requirement. So the
# two-letter forms are accepted only when dotted, or when written in capitals.
_DEGREE_ANY_CASE = re.compile(
    r"\b(?:b\.\s?e\b|b\.?\s?tech\b|b\.?\s?sc\b|b\.?\s?c\.?\s?a\b|b\.?\s?com\b|bachelors?\b|"
    r"m\.\s?e\b|m\.?\s?tech\b|m\.?\s?sc\b|m\.?\s?c\.?\s?a\b|m\.?\s?b\.?\s?a\b|masters?\b|"
    r"ph\.?\s?d\b|doctorate\b|diploma\b|post[\s-]?graduate\b)",
    re.I,
)
_DEGREE_CAPITALS = re.compile(r"\b(?:BE|ME|BA|BS|MS|MA|BSC|MSC|BTECH|MTECH)\b")


def mentions_a_degree(text: str) -> bool:
    """Whether a line names an academic qualification."""
    if not text:
        return False
    return bool(_DEGREE_ANY_CASE.search(text) or _DEGREE_CAPITALS.search(text))


def mentions_term(text: str, term: str) -> bool:
    """
    Whether a passage literally names a term, as a whole word or phrase.

    `PostgreSQL` in "Designed PostgreSQL schemas" counts; `java` in "JavaScript" does
    not. Names of one or two letters only count when capitalised as written in the
    term's proper form, because `go` and `r` are ordinary text in lowercase.
    """
    needle = (term or "").strip()
    if not needle or not text:
        return False
    if len(needle) <= 2:
        forms = f"{re.escape(needle.upper())}|{re.escape(needle.capitalize())}"
        flags = 0
    else:
        forms, flags = re.escape(needle), re.I
    pattern = rf"(?<![\w+#.])(?:{forms})(?![\w+#]|\.\w)"
    if re.search(pattern, text, flags):
        return True
    # `ReactJS` and `React.js` name the same technology as `react`.
    key = normalise(needle)
    return len(key) > 2 and " " not in needle and any(
        normalise(word) == key for word in re.findall(r"[\w+#.]+", text)
    )


# Levels of the awards `_DEGREE_ANY_CASE` recognises, so `B.Tech` satisfies "Bachelor's".
_DEGREE_LEVELS = (
    (3, re.compile(r"\b(?:ph\.?\s?d|doctorate)\b", re.I)),
    (2, re.compile(r"\b(?:m\.\s?[eas]|m\.?\s?tech|m\.?\s?sc|m\.?\s?c\.?\s?a|m\.?\s?b\.?\s?a|masters?|post[\s-]?graduate)\b", re.I)),
    (1, re.compile(r"\b(?:b\.\s?[eas]|b\.?\s?tech|b\.?\s?sc|b\.?\s?c\.?\s?a|b\.?\s?com|bachelors?)\b", re.I)),
)
_DEGREE_LEVELS_CAPITALS = {"BE": 1, "BA": 1, "BS": 1, "BSC": 1, "BTECH": 1, "ME": 2, "MA": 2, "MS": 2, "MSC": 2, "MTECH": 2}
_DEGREE_WORDS = {
    "degree", "degrees", "related", "field", "fields", "equivalent", "similar", "relevant",
    "discipline", "qualification", "required", "preferred", "minimum", "bachelor", "bachelors",
    "master", "masters", "doctorate", "phd", "ph.d", "diploma", "postgraduate",
}


def degree_level(text: str) -> int:
    """1 for a bachelor's, 2 for a master's, 3 for a doctorate, 0 when none is named."""
    if not text:
        return 0
    levels = [level for level, pattern in _DEGREE_LEVELS if pattern.search(text)]
    levels += [level for word, level in _DEGREE_LEVELS_CAPITALS.items() if re.search(rf"\b{word}\b", text)]
    return max(levels, default=0)


def qualification_similarity(required: str, held: str) -> float:
    """
    How far a resume's qualification meets a JD's, 0.0-1.0.

    Word overlap fails here because the same degree is written many ways: "B.Tech in
    Computer Science" shares only two words with "Bachelor's degree in Computer Science
    or a related field". So the level and the field are judged separately and weighted
    equally. Falls back to plain word overlap when either side names no recognised award.
    """
    required_level, held_level = degree_level(required), degree_level(held)
    if not required_level or not held_level:
        return phrase_similarity(required, held)

    def field(text: str) -> Set[str]:
        return {
            t for t in tokens(text)
            if t not in _DEGREE_WORDS and not degree_level(t) and not degree_level(t.upper())
        }

    wanted, have = field(required), field(held)
    field_score = 1.0 if not wanted else len(wanted & have) / len(wanted)
    level_score = 1.0 if held_level >= required_level else 0.0
    return 0.5 * level_score + 0.5 * field_score


def any_phrase_similarity(needle: str, haystack: Iterable[str]) -> float:
    """Best `phrase_similarity` between one phrase and a collection of candidates."""
    best = 0.0
    for candidate in haystack:
        best = max(best, phrase_similarity(needle, candidate))
        if best >= 1.0:
            break
    return best
