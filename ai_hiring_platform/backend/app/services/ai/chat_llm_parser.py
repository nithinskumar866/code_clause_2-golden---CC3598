"""
LLM-assisted query understanding — the model segments, the corpus decides.

WHY THE MODEL IS NOT ALLOWED TO SAY WHAT A WORD MEANS
-----------------------------------------------------
The obvious design is to ask the LLM "what skills is this recruiter asking for?" and
use the answer. That fails in exactly the way the old regex parser failed, only less
predictably: an 8B model will happily report `now` as a skill, invent `Java 8` when
the recruiter typed `java`, or answer with a candidate name that is not in the pool.
A hallucinated *requirement* is just as damaging as a hallucinated *fact* — it silently
changes the question before any retrieval happens.

So the division of labour here is deliberately narrow:

    THE MODEL   marks up the sentence — "this span is a person, this span is a place,
                this span names a technology, this is a question about your last
                answer". It contributes linguistic judgement, which is what it is good
                at, and it must quote spans VERBATIM from the recruiter's own words.

    THE CORPUS  resolves every span through `chat_lexicon`. A person span becomes a
                real candidate or nothing. A skill span becomes a known technology or
                nothing. A place span becomes somewhere a resume actually puts someone,
                or is reported as unmatched.

Nothing the model emits reaches retrieval unvalidated, so the worst a bad parse can do
is fall back to the deterministic reading. That is why this can run on a small model
without putting the answer at risk.

WHEN IT RUNS
------------
Only when the deterministic parser is genuinely stuck — no facets resolved, or a word
that looks like a name matched nobody. A question the rules already understand is never
sent to the model: it would add latency and a chance of being wrong to a parse that is
already correct and reproducible. With no LLM configured, or on any error, the
deterministic intent is returned untouched.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.services.ai import llm_service
from app.services.ai.chat_lexicon import CorpusLexicon
from app.services.ai.chat_query_understanding import (
    Conversation,
    QueryIntent,
    QueryKind,
    parse_intent,
)

_SYSTEM = """You label the parts of a recruiter's question. You do NOT answer it.

Return ONLY a JSON object, no prose, with exactly these keys:
{
  "asking_for": one of "search", "person", "fact", "compare", "count", "about_last_answer", "unclear",
  "person": the person's name exactly as the user typed it, or "",
  "place": the city/area exactly as the user typed it, or "",
  "technologies": list of technology names exactly as the user typed them, may be empty,
  "years": a number if the user asked for a minimum years of experience, else null,
  "asked_detail": the specific detail wanted about a person (e.g. "location", "college", "notice period"), or ""
}

Rules:
- Copy spans VERBATIM from the user's message. Never translate, expand, correct or invent a word.
- "technologies" means software, tools, languages or platforms ONLY. Ordinary English words
  like "now", "point", "best", "experience", "candidate" are NEVER technologies.
- "asking_for" is "about_last_answer" when the user is challenging or asking you to justify
  what you just told them.
- "asking_for" is "fact" when the user wants one specific detail about one person.
- If you are unsure about a field, use "" or null. Guessing is worse than leaving it empty.

Output the JSON object and nothing else."""

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _call(llm: Any, message: str, context_line: str) -> Optional[Dict[str, Any]]:
    prompt = f"{context_line}User message: \"{message[:400]}\"\n\nJSON:"
    varnames = getattr(getattr(llm, "complete", None), "__code__", None)
    if varnames and "system" in varnames.co_varnames:
        response = llm.complete(prompt, system=_SYSTEM, temperature=0.0)
    else:
        response = llm.complete(f"{_SYSTEM}\n\n{prompt}")
    raw = (getattr(response, "text", "") or "").strip()
    match = _JSON_BLOCK.search(raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _spans_present(message: str, span: str) -> bool:
    """
    The span must really be in the recruiter's sentence.

    This is the anti-hallucination check that makes the whole arrangement safe: a model
    that invents "Kubernetes" for a question that never mentioned it fails here, before
    the word can become a search requirement.
    """
    span = (span or "").strip()
    if not span:
        return False
    return span.lower() in (message or "").lower()


def refine_intent(
    message: str,
    intent: QueryIntent,
    lexicon: CorpusLexicon,
    context: Optional[Conversation] = None,
) -> QueryIntent:
    """
    Give a stuck parse a second reading, then re-resolve it against the corpus.

    Returns the ORIGINAL intent unless the model produced spans that (a) genuinely
    appear in the recruiter's message and (b) resolve to something real in the pool.
    """
    if not _worth_refining(intent):
        return intent

    llm = llm_service.get_llm()
    if llm is None:
        return intent

    context = context or Conversation()
    context_line = ""
    if context.has_subject:
        context_line = (
            f"The conversation is currently about the candidate {context.subject.name}. "
            f"A pronoun refers to them.\n"
        )

    try:
        parsed = _call(llm, message, context_line)
    except Exception as e:
        logger.warning(f"LLM query parse failed; keeping the deterministic reading: {e}")
        return intent
    if not parsed:
        return intent

    # Rebuild a query from VALIDATED spans only, then run it back through the
    # deterministic parser so exactly one code path decides the final intent.
    rebuilt = _rebuild_query(message, parsed, lexicon)
    if rebuilt is None:
        return intent

    refined = parse_intent(rebuilt, lexicon, context)
    if not refined.has_facets and refined.kind == QueryKind.AMBIGUOUS:
        return intent

    refined.raw = intent.raw          # the recruiter's own words stay the record
    refined.parsed_by = "rules+llm"
    # An attribute question keeps the detail the model isolated, which is usually a
    # cleaner retrieval probe than the whole sentence.
    detail = parsed.get("asked_detail")
    if refined.kind == QueryKind.ATTRIBUTE and isinstance(detail, str) and _spans_present(message, detail):
        refined.attribute_probe = detail.strip()
    return refined


def _worth_refining(intent: QueryIntent) -> bool:
    """Only a parse the rules could not settle is worth a model call."""
    if intent.kind == QueryKind.AMBIGUOUS:
        return True
    if intent.unresolved_name and not intent.people:
        return True
    if intent.unmatched_place:
        return True
    return False


def _rebuild_query(message: str, parsed: Dict[str, Any], lexicon: CorpusLexicon) -> Optional[str]:
    """
    Turn validated spans back into a sentence the deterministic parser can read.

    Rebuilding rather than assigning fields directly keeps ONE implementation of "what
    does this question mean" — the model can influence the reading, never bypass it.
    """
    parts: List[str] = []

    person = parsed.get("person")
    if _spans_present(message, person) and lexicon.resolve_people(person).matched:
        parts.append(person.strip())

    place = parsed.get("place")
    if _spans_present(message, place) and lexicon.find_places(place):
        parts.append(f"in {place.strip()}")

    technologies = parsed.get("technologies")
    if isinstance(technologies, list):
        for tech in technologies[:6]:
            if not isinstance(tech, str) or not _spans_present(message, tech):
                continue
            canonical, _ = lexicon.resolve_skill(tech.strip())
            if canonical:
                parts.append(canonical)

    years = parsed.get("years")
    if isinstance(years, (int, float)) and 0 < float(years) <= 60:
        parts.append(f"{years:g} years")

    if not parts:
        return None

    asking = (parsed.get("asking_for") or "").strip().lower()
    detail = parsed.get("asked_detail") if _spans_present(message, parsed.get("asked_detail")) else ""
    if asking == "about_last_answer":
        return f"why do you say that about {' '.join(parts)}"
    if asking == "fact" and detail:
        return f"what is the {detail.strip()} of {' '.join(parts)}"
    if asking == "person":
        return f"tell me about {' '.join(parts)}"
    if asking == "compare":
        return f"compare {' and '.join(parts)}"
    if asking == "count":
        return f"how many candidates have {' '.join(parts)}"
    return f"who has {' '.join(parts)}"
