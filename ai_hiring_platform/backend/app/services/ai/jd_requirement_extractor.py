import re
from typing import List, Dict, Any
from app.core.constants import TECH_TAXONOMY
from app.core.config import settings
from app.core.logging import logger

# ---------------------------------------------------------------------------
# Requirement extraction is DETERMINISTIC by design: the JD is a raw document,
# and Golden Rule 2 forbids the LLM from inspecting raw documents. Extraction is
# therefore a two-layer algorithm, not a keyword lookup:
#
#   Layer 1 (precision) — match known technologies from TECH_TAXONOMY.
#   Layer 2 (generalization) — pull requirement phrases straight from the JD text
#     using linguistic *grammar* cues (not a skill list), so unknown tools, soft
#     skills, domain knowledge, and seniority are captured for ANY job description.
#
# Downstream everything already generalizes (embeddings retrieve any string;
# skill_semantics classifies unseen skills via centroid) — this extractor was the
# one component that did not. Now it does, without hardcoding skill examples.
# ---------------------------------------------------------------------------

# Flattened taxonomy for the precision layer (longest first for greedy matching).
_TAXONOMY_TERMS = sorted(
    {s for skills in TECH_TAXONOMY.values() for s in skills}, key=len, reverse=True
)

# A skill token: starts with a letter, may carry tech punctuation (C++, CI/CD, .NET, node.js).
_WORD = r"[A-Za-z][A-Za-z0-9+#./-]*"

# Possession cues — grammar that signals "a required competency follows/precedes".
# These are language patterns, not domain examples, so they transfer to any field.
_CUE = (
    r"(?:experience|experienced|proficiency|proficient|expertise|expert|knowledge|"
    r"knowledgeable|familiarity|familiar|skilled|competency|competence|understanding|"
    r"fluency|fluent|versed|hands-on|background|ability|exposure)"
)
_CONNECTOR = r"(?:with|in|of|using|on|across|w/)"

# "<cue> <connector> <object>"  e.g. "knowledge of Snowflake and Airflow".
# Capture a generous object run; it is split on connectors then length-capped per skill.
_CUE_AFTER = re.compile(rf"{_CUE}\s+{_CONNECTOR}\s+((?:{_WORD}[\s,/&]+){{0,6}}{_WORD})", re.I)
# "<object> <cue>"             e.g. "Kubernetes experience", "React proficiency"
_CUE_BEFORE = re.compile(rf"((?:{_WORD}\s+){{0,2}}{_WORD})\s+{_CUE}\b", re.I)
# Capitalised / acronym tech phrases in requirement lines (Snowflake, React Native, AWS)
_CAP_PHRASE = re.compile(r"\b([A-Z][A-Za-z0-9+#./-]*(?:\s+(?:&\s+)?[A-Z][A-Za-z0-9+#./-]*){0,3})\b")
# Seniority: "5+ years", "3 years" -> a first-class requirement.
_YEARS = re.compile(r"(\d+)\s*\+?\s*years?", re.I)
# Splits an object phrase into individual skills.
_SPLIT = re.compile(r"\s*(?:,|/|&|\band\b|\bor\b)\s*", re.I)

# Structure / filler / role words that must never stand alone as a requirement.
# Grammar and job-posting boilerplate — not a skill blocklist.
_STOP = {
    "a", "an", "the", "and", "or", "to", "for", "of", "in", "on", "with", "as", "is",
    "are", "be", "we", "you", "our", "your", "their", "this", "that", "these", "those",
    "it", "its", "who", "will", "can", "must", "should", "have", "has", "nice", "plus",
    "etc", "strong", "good", "great", "excellent", "solid", "deep", "hands", "proven",
    # possession-cue vocabulary (grammar, not skills) — must not become requirements
    "exposure", "familiarity", "familiar", "proficiency", "proficient", "expertise",
    "expert", "competency", "competence", "fluency", "fluent", "versed", "experienced",
    "knowledgeable", "skilled", "background",
    "years", "year", "role", "team", "teams", "company", "candidate", "candidates",
    "requirements", "responsibilities", "qualifications", "skills", "skill", "ability",
    "experience", "knowledge", "understanding", "work", "working", "using", "looking",
    "preferred", "required", "related", "similar", "field", "degree", "software",
    "engineer", "engineers", "developer", "developers", "architect", "analyst",
    "manager", "designer", "scientist", "administrator", "specialist", "lead",
    "intern", "consultant", "senior", "junior", "mid", "level", "position", "job",
}

_MAX_WORDS = 4     # a requirement phrase is at most this many words
_MAX_RESULTS = 40  # guard against runaway extraction on very large JDs


def _clean_phrase(raw: str) -> str:
    """Normalise a candidate phrase: trim junk tokens, cap length, drop if empty/filler."""
    tokens = [t for t in re.split(r"\s+", raw.strip()) if t]
    # Strip leading/trailing pure-stopword tokens (keep interior, e.g. "ci/cd").
    while tokens and re.sub(r"[^a-z0-9]", "", tokens[0].lower()) in _STOP:
        tokens.pop(0)
    while tokens and re.sub(r"[^a-z0-9]", "", tokens[-1].lower()) in _STOP:
        tokens.pop()
    tokens = tokens[:_MAX_WORDS]
    if not tokens:
        return ""
    phrase = " ".join(tokens).strip(" -,/&")
    # Reject if every token is filler or it is a lone very-short non-tech token.
    words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in phrase.split()]
    if all(w in _STOP or len(w) < 2 for w in words):
        return ""
    return phrase


def _requirement_lines(jd_text: str) -> List[str]:
    """Lines likely to state a requirement: bullets, cue-bearing lines, or lines under
    a Requirements/Qualifications/Skills/Responsibilities heading. Filters out prose
    like 'We are looking for...' so capitalised-phrase extraction stays low-noise."""
    lines = [ln.strip() for ln in jd_text.split("\n") if ln.strip()]
    out, in_req_block = [], False
    header = re.compile(r"^\s*(requirements|qualifications|skills|responsibilities|"
                        r"what you.ll (need|do)|must have|nice to have|tech stack)\b", re.I)
    bullet = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+")
    cue_line = re.compile(_CUE, re.I)
    for ln in lines:
        if header.search(ln):
            in_req_block = True
            continue
        if bullet.search(ln) or cue_line.search(ln) or in_req_block:
            out.append(ln)
    return out


def extract_requirements(jd_text: str) -> List[str]:
    """
    Extract job requirements from a JD via a taxonomy precision layer plus a
    grammar-driven generalization layer. Returns a deterministic, deduplicated,
    sorted list that includes skills OUTSIDE the known taxonomy.
    """
    logger.info("Extracting requirements from Job Description text...")

    # key = lowercase form (dedupe), value = display form. Taxonomy wins on ties.
    found: dict = {}

    def add(display: str):
        display = display.strip()
        if not display:
            return
        key = display.lower()
        if key not in found and len(found) < _MAX_RESULTS:
            found[key] = display

    # --- Layer 1: known-technology precision match (canonical taxonomy casing) ---
    for term in _TAXONOMY_TERMS:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", jd_text, re.I):
            add(term)

    # --- Layer 2: grammar-driven extraction from requirement lines ---
    req_lines = _requirement_lines(jd_text)
    joined = "\n".join(req_lines)

    for m in _CUE_AFTER.finditer(joined):
        for part in _SPLIT.split(m.group(1)):
            add(_clean_phrase(part))
    for m in _CUE_BEFORE.finditer(joined):
        add(_clean_phrase(m.group(1)))
    for ln in req_lines:
        for m in _CAP_PHRASE.finditer(ln):
            add(_clean_phrase(m.group(1)))

    # Seniority requirement (captures "5+ years experience" — invisible to taxonomy).
    yrs = [int(m.group(1)) for m in _YEARS.finditer(jd_text)]
    if yrs:
        add(f"{max(yrs)}+ years experience")

    # Drop a bare single-word phrase that is merely a fragment of a longer kept
    # phrase (e.g. "financial" vs "financial modeling"), unless it is a real known
    # skill in its own right (e.g. keep "react" even alongside "react native").
    known = {t.lower() for t in _TAXONOMY_TERMS}
    multiword_tokens = {
        tok for v in found for tok in v.split() if len(v.split()) > 1
    }
    result = sorted(
        (
            v for k, v in found.items()
            if len(v.split()) > 1 or k in known or k not in multiword_tokens
        ),
        key=lambda s: s.lower(),
    )
    logger.info(f"Extracted {len(result)} requirements: {result}")
    return result


# ---------------------------------------------------------------------------
# Requirement prioritization: is a requirement must-have or nice-to-have?
# Determined dynamically from the JD's own language — priority headers
# ("Nice to have:", "Requirements:") set a running context, and inline cues on a
# line override it. No skill is hardcoded as important; the JD decides. (F1 will
# add an LLM refinement pass over this deterministic baseline when a key is set.)
# ---------------------------------------------------------------------------
_NICE_HEADER = re.compile(
    r"^\s*(nice[\s-]*to[\s-]*have|preferred|bonus|good[\s-]*to[\s-]*have|desirable|"
    r"pluses?|optional|nice)\b.*$", re.I)
_MUST_HEADER = re.compile(
    r"^\s*(requirements?|required|must[\s-]*have|qualifications|minimum qualifications|"
    r"what you.ll need|essential|responsibilities|you have)\b.*$", re.I)
_NICE_CUE = re.compile(
    r"\b(nice[\s-]*to[\s-]*have|preferred|preferable|bonus|a plus|good[\s-]*to[\s-]*have|"
    r"desirable|desired|optional|advantageous|ideally|would be (a )?(plus|great|nice))\b", re.I)
_MUST_CUE = re.compile(
    r"\b(must[\s-]*have|must|required|require|essential|mandatory|minimum|need to have)\b", re.I)


def classify_priorities(jd_text: str, requirements: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Classify each requirement as 'must' or 'nice' and assign a scoring weight, based
    purely on the JD's wording. Returns {requirement: {"importance", "weight"}}.
    Defaults to 'must' (a listed requirement is needed unless stated otherwise).
    """
    lines = [ln.strip() for ln in jd_text.split("\n") if ln.strip()]

    # Per-line priority: headers switch the running context; inline cues override.
    line_priority = []
    context = "must"
    for ln in lines:
        if _NICE_HEADER.match(ln):
            context = "nice"
        elif _MUST_HEADER.match(ln):
            context = "must"
        if _NICE_CUE.search(ln):
            line_priority.append((ln, "nice"))
        elif _MUST_CUE.search(ln):
            line_priority.append((ln, "must"))
        else:
            line_priority.append((ln, context))

    out: Dict[str, Dict[str, Any]] = {}
    for req in requirements:
        pat = re.compile(rf"(?<!\w){re.escape(req)}(?!\w)", re.I)
        seen = {pri for (ln, pri) in line_priority if pat.search(ln)}
        # 'must' dominates a conflict (required somewhere ⇒ treat as required);
        # only a purely nice-to-have mention downgrades. No mention ⇒ must.
        importance = "nice" if seen == {"nice"} else "must"
        weight = (settings.REQUIREMENT_WEIGHT_NICE if importance == "nice"
                  else settings.REQUIREMENT_WEIGHT_MUST)
        out[req] = {"importance": importance, "weight": weight}
    return out
