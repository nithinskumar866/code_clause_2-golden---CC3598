"""
Chatbot guardrails — the safety layer wrapped around recruiter Q&A.

Guardrails run on BOTH sides of the model, because they defend against different
failure modes:

  INPUT  (before any retrieval or LLM call)
    * scope        — the assistant answers questions about the candidate pool, and
                     nothing else. "What is HR?", "what time is it", trivia, coding
                     help, and general chit-chat are declined.
    * injection    — attempts to override instructions, extract the system prompt, or
                     switch persona ("ignore previous instructions", "you are now...").
    * secrets      — probes for API keys, env vars, DB contents, internal config or
                     source code are refused regardless of phrasing.
    * fairness     — filtering candidates on protected attributes (age, gender,
                     religion, caste, marital status, nationality...) is refused. This
                     is both a legal exposure and the single biggest hallucination
                     trigger, since resumes almost never state these attributes, so any
                     answer would be invented.
    * abuse        — hostile input gets a calm, respectful, non-escalating reply.

  OUTPUT (after reasoning, before the recruiter sees it)
    * grounding    — every candidate in an answer must trace to a real indexed resume
                     and carry at least one retrieved evidence chunk. Candidates the
                     model invented, or that carry no evidence, are dropped.
    * leakage      — the answer is scrubbed of anything resembling internal
                     credentials/config, in case the LLM echoes its own context.

Design notes: matching is deterministic (explicit patterns over an inspectable rule
set), never an LLM judging itself — a hallucinating model cannot be trusted to police
its own hallucinations. Candidate contact details (email/phone) are deliberately NOT
redacted: recruiters need to be able to reach the person, which is the legitimate
purpose of the tool.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- Refusal categories -----------------------------------------------------
SCOPE = "out_of_scope"
INJECTION = "prompt_injection"
SECRETS = "sensitive_system_data"
FAIRNESS = "discriminatory_filter"
ABUSE = "abusive_input"
EMPTY = "empty_query"

_REFUSALS: Dict[str, str] = {
    SCOPE: (
        "I can only help with questions about the candidates in your resume pool — "
        "their skills, experience, projects, education and fit for a role. "
        "Try something like \"who has 10+ years in Java?\" or "
        "\"how much Python experience does Priya have?\""
    ),
    INJECTION: (
        "I can't change my instructions or reveal how I'm configured. "
        "I'm happy to keep helping you search and evaluate candidates."
    ),
    SECRETS: (
        "I can't share internal system details such as credentials, configuration, "
        "database contents or source code. I can help you with candidate information "
        "from the resume pool instead."
    ),
    FAIRNESS: (
        "I can't screen or rank candidates using personal attributes such as age, "
        "gender, religion, caste, marital status or nationality — that would be "
        "discriminatory, and resumes don't reliably state them. "
        "I can rank on job-relevant evidence: skills, years of experience, projects "
        "and qualifications."
    ),
    ABUSE: (
        "I'd like to keep this respectful so I can be useful to you. "
        "Ask me anything about the candidates in your pool and I'll help."
    ),
    EMPTY: "Ask me anything about your candidate pool — skills, experience, or fit for a role.",
}


@dataclass
class GuardResult:
    """Verdict for one recruiter message."""

    allowed: bool
    category: Optional[str] = None
    message: Optional[str] = None
    triggers: List[str] = field(default_factory=list)

    @classmethod
    def block(cls, category: str, trigger: str) -> "GuardResult":
        return cls(allowed=False, category=category, message=_REFUSALS[category], triggers=[trigger])

    @classmethod
    def allow(cls) -> "GuardResult":
        return cls(allowed=True)


# --- Input patterns ---------------------------------------------------------
_INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instruction|prompt|rule|direction)",
    r"\bdisregard\s+(?:all\s+|any\s+|your\s+|the\s+)*(?:previous|prior|above|earlier)?\s*(?:instruction|prompt|rule|direction)",
    r"\b(?:system|initial|original)\s+prompt\b",
    r"\breveal\s+(?:your|the)\s+(?:prompt|instruction|rule|configuration)",
    r"\b(?:what\s+are|show\s+me)\s+your\s+(?:instruction|rule|prompt|guardrail)s?\b",
    r"\byou\s+are\s+now\b|\bpretend\s+to\s+be\b|\bact\s+as\s+(?:if|a\s+)?(?!a\s+recruiter)",
    r"\b(?:developer|god|dan|jailbreak|admin)\s+mode\b",
    r"\bbypass\s+(?:your\s+)?(?:guardrail|filter|restriction|safety)",
    r"\bwithout\s+(?:any\s+)?(?:restriction|guardrail|filter|limitation)s?\b",
]

_SECRET_PATTERNS = [
    # No leading \b: an underscore is a word character, so "OPENAI_API_KEY" has no
    # boundary before "API" and a \b-anchored pattern would miss the most obvious probe.
    r"api[\s_-]?key", r"secret[\s_-]?key", r"access[\s_-]?token",
    r"\bpassword\b", r"\bcredential", r"\benv(?:ironment)?\s+(?:var|file)",
    r"\b\.env\b", r"\bconnection\s+string\b",
    r"\bdatabase\s+(?:schema|dump|password|credential|url)\b",
    r"\bdrop\s+table\b|\bdelete\s+from\b|\bselect\s+\*\s+from\b",
    r"\bsource\s+code\b", r"\byour\s+(?:model|architecture|weights|training\s+data)\b",
    r"\bwhich\s+(?:llm|model)\s+(?:are\s+you|do\s+you)\b",
    r"\b(?:company|internal|trade)\s+secret", r"\bconfidential\s+(?:company|internal|business)",
    r"\bserver\s+(?:config|address|ip)\b", r"\bruntime\s+config",
]

# Protected attributes. Blocked when used to FILTER/SELECT candidates.
_PROTECTED = (
    r"(?:age|ages|aged|young|younger|older|elderly|dob|date\s+of\s+birth|"
    r"gender|male|female|man|woman|men|women|boy|girl|"
    r"religion|religious|hindu|muslim|christian|sikh|jain|buddhist|"
    r"caste|race|racial|ethnic(?:ity)?|colou?r|"
    r"marital\s+status|married|unmarried|single|divorced|pregnan\w*|"
    r"nationality|citizenship|visa\s+status|"
    r"disab(?:led|ility)|handicap\w*|health\s+condition|medical\s+history|"
    r"caste|sexual\s+orientation|gay|lesbian|lgbt\w*|"
    r"political|photo(?:graph)?|appearance|looks)"
)
_FAIRNESS_PATTERNS = [
    # "only male candidates", "candidates under 30", "prefer young engineers"
    rf"\b(?:only|just|prefer(?:ably)?|filter|exclude|avoid|remove|no)\s+\w{{0,12}}\s*{_PROTECTED}\b",
    rf"\b{_PROTECTED}\s+(?:candidate|applicant|resume|profile|people|person)s?\b",
    rf"\b(?:candidate|applicant|resume|profile)s?\s+(?:who\s+are\s+|that\s+are\s+|being\s+)?{_PROTECTED}\b",
    r"\b(?:under|below|above|over|younger\s+than|older\s+than)\s+\d{1,2}\s*(?:years?\s*old|yrs?\s*old|$)",
    rf"\bwhat\s+is\s+(?:the\s+)?{_PROTECTED}\s+of\b",
    rf"\b(?:his|her|their|the\s+candidate'?s?)\s+{_PROTECTED}\b",
    rf"\bshortlist\s+\w{{0,12}}\s*{_PROTECTED}\b",
]

_ABUSE_PATTERNS = [
    r"\b(?:fuck|shit|bitch|bastard|asshole|dumbass|idiot|stupid|moron|retard|useless|garbage)\b",
    r"\byou\s+(?:are|'re)\s+(?:so\s+)?(?:useless|stupid|worthless|trash|dumb)\b",
    r"\bshut\s+up\b",
]

# Vocabulary that makes a question legitimately about the candidate pool.
_IN_SCOPE_TERMS = [
    r"\bcandidate|applicant|resume|cv|profile|talent|pool|shortlist|screen",
    r"\bhir(?:e|ing)|recruit|interview|role|position|job|vacancy|opening|jd",
    r"\bexperience|experienced|yrs?\b|years?\b|senior|junior|mid[\s-]?level|fresher",
    r"\bskill|expert|proficien|knowledge|worked\s+with|hands[\s-]?on",
    r"\bproject|education|degree|qualification|certification|university|college",
    r"\bmatch|fit|suitab|best|top|rank|compare|better|strongest|recommend",
    r"\bwho\b|\bwhich\b|\bany(?:one|body)\b|\bhow\s+many\b|\bhow\s+much\b|\bfind\b|\bshow\b|\bsearch\b",
    r"\bemail|phone|contact|reach|call\b",
    r"\bstack|tech|technolog|language|framework|tool|platform|database",
    # Job-title grammar — "react developer", "backend engineer", "data scientist".
    r"\b(?:engineer|developer|programmer|architect|analyst|manager|designer|scientist|"
    r"administrator|consultant|specialist|lead|director|intern|devops|sre|tester|qa)s?\b",
    # Follow-up references, which carry no hiring nouns of their own but are only ever
    # meaningful in the context of a previous candidate answer.
    r"\bof\s+(?:those|them|these)\b|\bamong\s+(?:those|them|these)\b|\bnarrow\s+(?:it\s+)?down\b",
    r"\btell\s+me\s+more\b|\bmore\s+about\b|\bthe\s+(?:first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)\b",
    # Asking about a person by name. The pool IS people, so "tell me about <someone>"
    # is in scope even when that someone turns out not to exist — the honest reply is
    # "I have nobody by that name", not "that is off topic". Explicit trivia patterns
    # are checked earlier and still win, so "who is the father of the nation" is
    # unaffected.
    r"\btell\s+me\s+about\b|\bwho\s+is\b|\bwho\s+are\b|\bhow\s+is\b",
    # Questions about a candidate's own details. A resume states where someone lives,
    # what they studied and how to reach them, so asking is squarely in scope — this
    # is the vocabulary of "where is he from", not of general knowledge.
    r"\blocation\b|\baddress\b|\bcity\b|\bhometown\b|\bnative\b|\bbased\s+in\b|\blives?\s+in\b",
    r"\bcollege\b|\buniversity\b|\bschool\b|\bgraduat|\bstudied\b|\bcgpa\b|\bmarks\b",
    r"\binitials?\b|\bfull\s+name\b|\bdesignation\b|\bnotice\s+period\b|\bcertif",
    # Asking the assistant to justify what it just said is part of the conversation,
    # not an off-topic aside.
    r"\bwhy\s+(?:do|did)\s+you\b|\bsupport\s+your\b|\bjustify\b|\bon\s+what\s+basis\b",
    r"\bare\s+you\s+sure\b|\bprove\s+it\b|\bwhat\s+makes\s+(?:him|her|them)\b",
]

# A message that only makes sense as a continuation — it carries a pronoun or a
# demonstrative and nothing else to go on. These are IN SCOPE when the conversation
# has already established who is being discussed, and only then.
_CONTINUATION_ONLY = re.compile(
    r"\b(?:he|him|his|she|her|hers|they|them|their|theirs|that|this|those|these|it)\b", re.I
)


def _mentions_known_technology(text: str) -> bool:
    """
    True when the message names a technology the platform knows about.

    A bare "Kubernetes?" is a legitimate recruiter question even though it contains no
    hiring noun at all, so the shared tech taxonomy — not a second hand-written list —
    is consulted before declaring a message off-topic.
    """
    from app.core.constants import TECH_TAXONOMY

    lowered = (text or "").lower()
    return any(
        re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lowered)
        for skills in TECH_TAXONOMY.values()
        for term in skills
    )

# Unambiguously general-knowledge / assistant-trivia questions.
_OUT_OF_SCOPE_PATTERNS = [
    r"^\s*(?:hi|hello|hey|yo|sup|thanks|thank\s+you|ok(?:ay)?|cool|nice)\s*[!.?]*\s*$",
    r"\bwhat(?:'s| is)\s+(?:the\s+)?(?:time|date|day|weather|temperature)\b",
    r"\bwhat(?:'s| is)\s+(?:an?\s+)?(?:hr|human\s+resources?|recruitment|hiring)\s*\??$",
    # General-knowledge "who/what is …" trivia. Deliberately broad: the assistant only
    # knows the resume pool, so a question about the world at large is always wrong for
    # it to attempt. "who is the father of the nation" previously slipped through and
    # was answered with four unrelated candidates.
    r"\bwho\s+(?:is|was|are|were)\s+(?:the\s+)?(?:father|mother|founder|inventor|president|"
    r"prime\s+minister|king|queen|author|creator|discoverer|leader)\b",
    r"\bwho\s+(?:invented|discovered|founded|wrote|created|built)\b",
    r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:meaning|definition|purpose|history|capital|population)\s+of\b",
    r"\bcapital\s+of\b|\bpopulation\s+of\b",
    r"\btell\s+me\s+a\s+(?:joke|story|poem)\b",
    r"\bwrite\s+(?:me\s+)?(?:a\s+)?(?:poem|song|essay|python|java|code|script|program)\b",
    r"\btranslate\b.*\binto\b",
    r"\bhow\s+(?:do\s+i|to)\s+(?:cook|drive|invest|lose\s+weight)\b",
    r"^\s*\d+\s*[-+*/]\s*\d+\s*=?\s*$",
    r"\bwhat\s+can\s+you\s+do\b.*\b(?:besides|other\s+than|apart\s+from)\b",
]


def _matches(patterns: List[str], text: str) -> Optional[str]:
    for p in patterns:
        if re.search(p, text, re.I):
            return p
    return None


def check_query(
    query: str,
    known_names: Optional[List[str]] = None,
    conversation_has_subject: bool = False,
) -> GuardResult:
    """
    Screen one recruiter message before any retrieval or LLM call happens.

    Order matters: injection and secrets are checked first (they may be phrased in
    perfectly on-topic language), then fairness, then abuse, and scope last — so a
    genuine hiring question is never rejected as "off topic" by an earlier rule.

    `conversation_has_subject` says whether an earlier turn established who is being
    discussed. Scope is a property of the CONVERSATION, not of an isolated string:
    "where is he from" carries no hiring noun at all, yet immediately after "tell me
    about Naveen" it is the most natural question a recruiter could ask. Judging it
    alone is what made the assistant refuse its own follow-ups.
    """
    text = (query or "").strip()
    if not text:
        return GuardResult.block(EMPTY, "empty")
    if len(text) > 2000:
        return GuardResult.block(SCOPE, "length")

    if (hit := _matches(_INJECTION_PATTERNS, text)):
        return GuardResult.block(INJECTION, hit)
    if (hit := _matches(_SECRET_PATTERNS, text)):
        return GuardResult.block(SECRETS, hit)
    if (hit := _matches(_FAIRNESS_PATTERNS, text)):
        return GuardResult.block(FAIRNESS, hit)
    if (hit := _matches(_ABUSE_PATTERNS, text)):
        return GuardResult.block(ABUSE, hit)

    # Scope: an explicit off-topic pattern blocks, UNLESS the message also carries
    # genuine hiring vocabulary or names someone in the pool (e.g. a candidate really
    # called "Hr" would otherwise be unreachable).
    # Recruiters type first names ("how is priya", "say abt priya"), so matching only
    # the full name declared those out of scope and refused a perfectly valid question
    # about a real candidate. Match on name PARTS as well.
    names_hit = False
    if known_names:
        lowered = text.lower()
        parts = set()
        for n in known_names:
            if not n:
                continue
            parts.add(n.lower())
            parts.update(p.lower() for p in n.split() if len(p) > 2)
        names_hit = any(
            re.search(rf"(?<!\w){re.escape(p)}(?!\w)", lowered) for p in parts
        )

    # An explicit off-topic pattern wins over incidental scope vocabulary: "write me a
    # python script" mentions a technology and the word "list", but it is a coding
    # request, not a recruiting question. Only naming a real candidate overrides it.
    if (hit := _matches(_OUT_OF_SCOPE_PATTERNS, text)) and not names_hit:
        return GuardResult.block(SCOPE, hit)

    in_scope = (
        names_hit
        or _matches(_IN_SCOPE_TERMS, text) is not None
        or _mentions_known_technology(text)
        # A continuation of an established subject inherits that subject's scope.
        or (conversation_has_subject and _CONTINUATION_ONLY.search(text) is not None)
    )
    if not in_scope:
        return GuardResult.block(SCOPE, "no_hiring_intent")

    return GuardResult.allow()


# --- Output guardrails ------------------------------------------------------
# Anything credential-shaped that must never survive into a recruiter-visible answer,
# in case the LLM echoes part of its own runtime context.
_LEAK_PATTERNS = [
    (re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{16,}\b"), "[redacted]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "[redacted]"),
    (re.compile(r"\b(?:api[_\s-]?key|token|password|secret)\s*[:=]\s*\S+", re.I), "[redacted]"),
    (re.compile(r"\b[a-z]+://[^\s]*(?:proxy\.runpod|localhost|127\.0\.0\.1)[^\s]*", re.I), "[internal endpoint]"),
]


def scrub_output(text: str) -> str:
    """Strip credential-shaped content from a generated answer."""
    out = text or ""
    for pattern, replacement in _LEAK_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def ground_candidates(
    candidates: List[Dict[str, Any]],
    valid_resume_ids: set,
) -> List[Dict[str, Any]]:
    """
    Anti-hallucination gate on the answer.

    A candidate survives only if it (a) refers to a resume that actually exists in the
    index and (b) carries at least one retrieved evidence chunk. Anything the reasoning
    layer invented, or asserted without evidence, is discarded rather than shown.
    """
    grounded = []
    for c in candidates:
        if c.get("resume_id") not in valid_resume_ids:
            continue
        if not c.get("evidence"):
            continue
        grounded.append(c)
    return grounded


def refusal_message(category: str) -> str:
    return _REFUSALS.get(category, _REFUSALS[SCOPE])
