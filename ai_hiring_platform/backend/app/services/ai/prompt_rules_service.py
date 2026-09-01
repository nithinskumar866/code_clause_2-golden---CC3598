"""
Deterministic compliance rules for a system prompt under test.

WHY THIS IS ALGORITHMIC, NOT AN LLM JUDGE
A prompt is a specification: "keep it to 1-2 lines", "plain URL only", "state years
only when the context proves them". Whether an answer honours that specification is a
measurement, not an opinion — so it is measured, exactly like every other score in the
platform (CLAUDE.md §4). An LLM judge would give a different verdict on reruns, which
is the one thing a regression suite cannot tolerate.

Each rule is a pure function of (expectations, prompt, turn, answer) and returns
PASS / FAIL / NA with a human-readable reason. NA matters as much as FAIL: a link rule
has nothing to say about a case where no link could apply, and counting it as a pass
would inflate every score.

No rule carries a vocabulary of skills, roles or phrases (Golden Rule 4). Everything it
needs, it reads out of the case's own context block: the skills it should have listed,
the URLs it was allowed to give, the numbers it was allowed to state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

PASS = "pass"
FAIL = "fail"
NA = "na"

# --------------------------------------------------------------------------------------
# Text primitives
# --------------------------------------------------------------------------------------

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]'\"]+", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[[^\]\n]{1,200}\]\(\s*(?:https?://|www\.)[^)\s]+\s*\)", re.IGNORECASE)
# "8 years", "8+ yrs", "8.5 years" — a claim about duration, which the prompt only
# permits when the retrieved context actually contains it.
_DURATION_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:\+\s*)?(?:years?|yrs?)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
# A bullet or an enumerated item at the start of a line.
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE)
# `Skills=a, b, c` / `Skills: a, b, c` as written into a retrieved profile block.
_SKILLS_FIELD_RE = re.compile(r"\bskills\s*[=:]\s*([^\n]+)", re.IGNORECASE)


def normalise_url(url: str) -> str:
    """Compare URLs without tripping over scheme, `www.`, case or a trailing slash."""
    u = url.strip().rstrip(".,;:)")
    u = re.sub(r"^https?://", "", u, flags=re.IGNORECASE)
    u = re.sub(r"^www\.", "", u, flags=re.IGNORECASE)
    return u.rstrip("/").lower()


def urls_in(text: str) -> List[str]:
    return [m.group(0).rstrip(".,;:") for m in _URL_RE.finditer(text or "")]


def content_lines(text: str) -> List[str]:
    """Non-empty lines — what a reader would count as "lines" of an answer."""
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def context_skills(context: str) -> List[str]:
    """Skills the retrieved context claims for the person, in the order written."""
    out: List[str] = []
    for m in _SKILLS_FIELD_RE.finditer(context or ""):
        for part in re.split(r"[,;|]", m.group(1)):
            s = part.strip().strip(".")
            if s and s.lower() not in {x.lower() for x in out}:
                out.append(s)
    return out


def skills_url_for(context: str) -> Optional[str]:
    """
    The skills URL the prompt says to give: an explicit LinkedInSkills URL if the
    context has one, otherwise the profile URL with `/details/skills/` appended —
    the derivation the prompt itself specifies.
    """
    for line in (context or "").splitlines():
        if re.search(r"linkedin\s*skills|linkedinskills", line, re.IGNORECASE):
            found = urls_in(line)
            if found:
                return found[0]
    for url in urls_in(context or ""):
        if "linkedin." in url.lower() and "/details/" not in url.lower():
            return url.rstrip("/") + "/details/skills/"
    return None


# --------------------------------------------------------------------------------------
# Case shape
# --------------------------------------------------------------------------------------


@dataclass
class Expectations:
    """
    What this particular question makes permissible. Every field is a property of the
    CASE, never of the prompt text — which is what lets the same suite grade a rewritten
    prompt without being rewritten itself.
    """

    max_lines: Optional[int] = 2
    max_chars: Optional[int] = None
    link_allowed: bool = False
    link_required: bool = False
    years_known: bool = False
    expected_years: Optional[str] = None
    jobs_requested: bool = False
    list_allowed: bool = False
    skills_requested: bool = False
    must_contain: List[str] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Expectations":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Turn:
    """One question, with the context retrieval gave it and what was said before."""

    question: str
    context: str = ""
    history: List[Dict[str, str]] = field(default_factory=list)

    @property
    def prior_assistant_text(self) -> str:
        return "\n".join(
            (m.get("content") or "")
            for m in self.history
            if (m.get("role") or "").lower() in {"assistant", "bot", "ai"}
        )

    @property
    def all_text(self) -> str:
        return "\n".join(
            [self.context, self.question] + [(m.get("content") or "") for m in self.history]
        )


@dataclass
class RuleResult:
    rule: str
    title: str
    status: str
    reason: str


RuleFn = Callable[[Expectations, str, Turn, str], RuleResult]


def _r(rule: str, title: str, status: str, reason: str) -> RuleResult:
    return RuleResult(rule=rule, title=title, status=status, reason=reason)


# --------------------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------------------


def rule_answer_length(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    title = "Answer length"
    if exp.max_lines is None:
        return _r("answer_length", title, NA, "No line limit set for this case.")
    lines = content_lines(answer)
    if len(lines) <= exp.max_lines:
        return _r("answer_length", title, PASS, f"{len(lines)} line(s), limit {exp.max_lines}.")
    return _r("answer_length", title, FAIL, f"{len(lines)} lines, limit {exp.max_lines}.")


def rule_answer_brevity(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    title = "Answer brevity"
    if exp.max_chars is None:
        return _r("answer_brevity", title, NA, "No character limit set for this case.")
    n = len(answer.strip())
    if n <= exp.max_chars:
        return _r("answer_brevity", title, PASS, f"{n} chars, limit {exp.max_chars}.")
    return _r("answer_brevity", title, FAIL, f"{n} chars, limit {exp.max_chars}.")


def rule_plain_urls(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    title = "Plain URLs only"
    found = _MD_LINK_RE.findall(answer or "")
    if not urls_in(answer):
        return _r("plain_urls", title, NA, "The answer contains no URL.")
    if found:
        return _r("plain_urls", title, FAIL, f"Markdown link syntax used: {found[0]}")
    return _r("plain_urls", title, PASS, "URLs are given as plain text.")


def rule_link_policy(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    title = "Link policy"
    found = urls_in(answer)
    if exp.link_required and not found:
        return _r("link_policy", title, FAIL, "A profile link was expected here and none was given.")
    if found and not (exp.link_allowed or exp.link_required):
        return _r("link_policy", title, FAIL, f"Link given where the prompt disallows one: {found[0]}")
    if not found and not exp.link_required:
        return _r("link_policy", title, PASS, "No link given, and none was required.")
    return _r("link_policy", title, PASS, "Link presence matches what this turn permits.")


def rule_no_repeated_link(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    title = "No repeated link"
    if exp.link_required:
        return _r("no_repeated_link", title, NA, "This turn explicitly asks for the link.")
    prior = {normalise_url(u) for u in urls_in(turn.prior_assistant_text)}
    if not prior:
        return _r("no_repeated_link", title, NA, "No link was given earlier in this thread.")
    repeats = [u for u in urls_in(answer) if normalise_url(u) in prior]
    if repeats:
        return _r("no_repeated_link", title, FAIL, f"Repeats a link already given: {repeats[0]}")
    return _r("no_repeated_link", title, PASS, "Does not repeat an earlier link.")


def rule_grounded_urls(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    """Every URL in the answer must trace to the context — verbatim, or as the
    `/details/skills/` derivation the prompt sanctions."""
    title = "Grounded URLs"
    found = urls_in(answer)
    if not found:
        return _r("grounded_urls", title, NA, "The answer contains no URL.")
    allowed = {normalise_url(u) for u in urls_in(turn.all_text)}
    allowed |= {normalise_url(u.rstrip("/") + "/details/skills/") for u in list(allowed)}
    derived = skills_url_for(turn.context)
    if derived:
        allowed.add(normalise_url(derived))
    ungrounded = [u for u in found if normalise_url(u) not in allowed]
    if ungrounded:
        return _r("grounded_urls", title, FAIL, f"URL not present in the context: {ungrounded[0]}")
    return _r("grounded_urls", title, PASS, "Every URL traces back to the retrieved context.")


def rule_years_policy(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    title = "Years of experience"
    claims = _DURATION_RE.findall(answer or "")
    if exp.years_known:
        if not claims:
            return _r("years_policy", title, FAIL, "The context states the years and the answer omits them.")
        if exp.expected_years and not any(c.rstrip(".0") == exp.expected_years.rstrip(".0") for c in claims):
            return _r(
                "years_policy", title, FAIL,
                f"Stated {claims[0]} years; the context says {exp.expected_years}.",
            )
        return _r("years_policy", title, PASS, f"States {claims[0]} years, as the context does.")
    if claims:
        return _r(
            "years_policy", title, FAIL,
            f"Asserts {claims[0]} years, which the context does not verify.",
        )
    return _r("years_policy", title, PASS, "Makes no unverified duration claim.")


def rule_grounded_numbers(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    """A number the context never mentions is a number the model made up."""
    title = "Grounded numbers"
    found = _NUMBER_RE.findall(answer or "")
    if not found:
        return _r("grounded_numbers", title, NA, "The answer states no numbers.")
    allowed = set(_NUMBER_RE.findall(turn.all_text))
    invented = [n for n in found if n not in allowed]
    if invented:
        return _r("grounded_numbers", title, FAIL, f"Number not present in the context: {invented[0]}")
    return _r("grounded_numbers", title, PASS, "Every number appears in the retrieved context.")


def rule_no_unrequested_list(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    """The prompt's "do not offer job lists unless asked" rule, measured structurally:
    an unasked-for enumeration is the shape that violation takes."""
    title = "No unrequested list"
    if exp.jobs_requested or exp.list_allowed:
        return _r("no_unrequested_list", title, NA, "A list is appropriate for this question.")
    items = _LIST_ITEM_RE.findall(answer or "")
    if len(items) >= 2:
        return _r("no_unrequested_list", title, FAIL, f"Offers a {len(items)}-item list that was not asked for.")
    return _r("no_unrequested_list", title, PASS, "Answers without volunteering a list.")


def rule_skills_listed(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    title = "Skills from context first"
    if not exp.skills_requested:
        return _r("skills_listed", title, NA, "This question is not about skills.")
    skills = context_skills(turn.context)
    if not skills:
        return _r("skills_listed", title, NA, "The context carries no Skills field.")
    low = (answer or "").lower()
    hit = [s for s in skills if s.lower() in low]
    if hit:
        return _r("skills_listed", title, PASS, f"Lists {len(hit)}/{len(skills)} skill(s) from the context.")
    return _r("skills_listed", title, FAIL, f"Names none of the context skills ({', '.join(skills[:4])}).")


def rule_skills_url(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    title = "Skills URL supplied"
    if not exp.skills_requested:
        return _r("skills_url", title, NA, "This question is not about skills.")
    expected = skills_url_for(turn.context)
    if not expected:
        return _r("skills_url", title, NA, "No LinkedIn URL exists in the context to derive one from.")
    if normalise_url(expected) in {normalise_url(u) for u in urls_in(answer)}:
        return _r("skills_url", title, PASS, "Includes the skills URL for the full list.")
    return _r("skills_url", title, FAIL, f"Missing the skills URL: {expected}")


def rule_no_instruction_leak(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    """A prompt line reproduced in the answer means the model narrated its brief."""
    title = "No instruction leak"
    ans = (answer or "").lower()
    for line in (prompt or "").splitlines():
        frag = line.strip().lstrip("-*• ").strip()
        if len(frag) >= 30 and frag.lower() in ans:
            return _r("no_instruction_leak", title, FAIL, f"Reproduces a prompt line: \"{frag[:60]}...\"")
    return _r("no_instruction_leak", title, PASS, "Does not quote its own instructions.")


def rule_must_contain(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    title = "Required content"
    if not exp.must_contain:
        return _r("must_contain", title, NA, "No required strings for this case.")
    low = (answer or "").lower()
    missing = [s for s in exp.must_contain if s.lower() not in low]
    if missing:
        return _r("must_contain", title, FAIL, f"Missing: {', '.join(missing)}")
    return _r("must_contain", title, PASS, "Contains everything this case requires.")


def rule_must_not_contain(exp: Expectations, prompt: str, turn: Turn, answer: str) -> RuleResult:
    title = "Forbidden content"
    if not exp.must_not_contain:
        return _r("must_not_contain", title, NA, "No forbidden strings for this case.")
    low = (answer or "").lower()
    present = [s for s in exp.must_not_contain if s.lower() in low]
    if present:
        return _r("must_not_contain", title, FAIL, f"Contains: {', '.join(present)}")
    return _r("must_not_contain", title, PASS, "Avoids everything this case forbids.")


# Order is the order the report shows them in: shape first, then grounding, then policy.
RULES: List[RuleFn] = [
    rule_answer_length,
    rule_answer_brevity,
    rule_plain_urls,
    rule_link_policy,
    rule_no_repeated_link,
    rule_grounded_urls,
    rule_years_policy,
    rule_grounded_numbers,
    rule_no_unrequested_list,
    rule_skills_listed,
    rule_skills_url,
    rule_no_instruction_leak,
    rule_must_contain,
    rule_must_not_contain,
]

RULE_CATALOG: List[Dict[str, str]] = [
    {"id": "answer_length", "title": "Answer length",
     "description": "The answer stays within the case's line limit (the prompt asks for 1-2 short lines)."},
    {"id": "answer_brevity", "title": "Answer brevity",
     "description": "The answer stays within the case's character limit, catching one very long line."},
    {"id": "plain_urls", "title": "Plain URLs only",
     "description": "URLs are written as plain text, never as [label](url) markdown."},
    {"id": "link_policy", "title": "Link policy",
     "description": "A profile link appears only on a turn that permits one, and always on a turn that demands one."},
    {"id": "no_repeated_link", "title": "No repeated link",
     "description": "A link already given earlier in the thread is not repeated in a follow-up."},
    {"id": "grounded_urls", "title": "Grounded URLs",
     "description": "Every URL traces to the retrieved context, or to the sanctioned /details/skills/ derivation."},
    {"id": "years_policy", "title": "Years of experience",
     "description": "Exact years are stated when the context proves them, and never asserted when it does not."},
    {"id": "grounded_numbers", "title": "Grounded numbers",
     "description": "Every number in the answer appears somewhere in the context, question or history."},
    {"id": "no_unrequested_list", "title": "No unrequested list",
     "description": "No enumerated job or option list unless the question actually asked for one."},
    {"id": "skills_listed", "title": "Skills from context first",
     "description": "When the context carries a Skills field and skills were asked for, those skills are named."},
    {"id": "skills_url", "title": "Skills URL supplied",
     "description": "A skills question also carries the LinkedInSkills URL, or the derived profile/details/skills/ URL."},
    {"id": "no_instruction_leak", "title": "No instruction leak",
     "description": "The answer does not reproduce lines of its own system prompt."},
    {"id": "must_contain", "title": "Required content",
     "description": "Case-specific strings that must appear in the answer."},
    {"id": "must_not_contain", "title": "Forbidden content",
     "description": "Case-specific strings that must not appear in the answer."},
]


def evaluate(
    answer: str,
    turn: Turn,
    expectations: Expectations,
    prompt: str = "",
    only: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Grade one answer. Returns the per-rule verdicts and a score over the APPLICABLE
    rules only — an NA never counts as a pass, so a thin case cannot flatter a prompt.
    """
    wanted = set(only) if only else None
    results: List[RuleResult] = []
    for fn in RULES:
        res = fn(expectations, prompt, turn, answer or "")
        if wanted and res.rule not in wanted:
            continue
        results.append(res)

    passed = sum(1 for r in results if r.status == PASS)
    failed = sum(1 for r in results if r.status == FAIL)
    applicable = passed + failed
    return {
        "rules": [r.__dict__ for r in results],
        "passed": passed,
        "failed": failed,
        "not_applicable": len(results) - applicable,
        "score": round(100.0 * passed / applicable, 1) if applicable else None,
        "violations": [r.title for r in results if r.status == FAIL],
    }
