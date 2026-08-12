"""
API contract for the recruiter chatbot (`/api/v1/chat`).

Backend owns this contract; `frontend/src/types/index.ts` mirrors it.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatEvidence(BaseModel):
    """One retrieved resume chunk backing a claim — the audit trail for a match."""

    skill: str
    text: str
    section: str
    page: int = 1
    filename: str = ""
    similarity: float
    # True when the chunk contains the skill term verbatim, rather than merely being
    # semantically near it. Literal mentions are what certify a skill.
    literal: bool = False


class ChatScoreBreakdown(BaseModel):
    """
    Decomposition of the match percentage, so the number is explainable.

    `skill_coverage` is the mean *depth* of the requested skills — how well each is
    substantiated by the sections it appears in — not a found/not-found flag.
    """

    skill_coverage: int
    evidence_strength: int
    experience_fit: int


class ChatCandidate(BaseModel):
    resume_id: int
    name: Optional[str] = None
    title: Optional[str] = None
    filename: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    total_years: Optional[float] = None
    seniority_level: Optional[str] = None
    match_percentage: int
    breakdown: ChatScoreBreakdown
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    demonstrated_skills: List[str] = []
    listed_only_skills: List[str] = []
    # Per-skill substantiation: how deeply proven (0-100) and in which resume sections.
    skill_depth: Dict[str, int] = {}
    skill_sections: Dict[str, List[str]] = {}
    all_skills: List[str] = []
    sections_present: List[str] = []
    experience_note: str = ""
    # Whether this candidate actually satisfies a requested location, and how to say
    # it. Surfaced so a card can never look like a match on a constraint it failed.
    location_match: bool = True
    location_note: str = ""
    # One-line headline judgement for this candidate, e.g. "Strong fit — 88%, proven
    # in real work". Written by the LLM when reachable, deterministic otherwise.
    verdict: str = ""
    # The card body as SCANNABLE POINTS rather than a paragraph. A recruiter skims a
    # shortlist; prose forces them to parse a sentence to find one fact.
    highlights: List[str] = []
    reasoning: str = ""
    evidence: List[ChatEvidence] = []


class ChatIntent(BaseModel):
    """How the question was understood — surfaced so recruiters can see the parse."""

    kind: str = "skill_search"
    skills: List[str] = []
    min_years: Optional[float] = None
    # An upper bound ("less than 5 years"). Separate from min_years so the UI can show
    # which direction the recruiter's experience bar actually points.
    max_years: Optional[float] = None
    # Places named in the question that the pool actually knows about.
    places: List[str] = []
    # The specific detail a fact question asked for ("location", "initials"), if any.
    attribute: Optional[str] = None
    named_candidates: List[str] = []
    # [[typed, corrected]] — e.g. [["javaa", "java"]], so the UI can say what it assumed.
    corrections: List[List[str]] = []
    unresolved_name: Optional[str] = None
    # A place the recruiter named that no resume mentions — reported, never dropped.
    unmatched_place: Optional[str] = None
    is_followup: bool = False
    # "rules" or "rules+llm" — whether the language model helped read the question.
    parsed_by: str = "rules"


class ChatFact(BaseModel):
    """
    A direct answer to a direct question about one candidate.

    `found` is false when the resume genuinely does not state the detail. That is a
    real answer, not a failure: substituting a pool-wide search would hand the
    recruiter a different person, which is worse than an honest gap.
    """

    resume_id: int
    name: Optional[str] = None
    attribute: str = ""
    value: Optional[str] = None
    found: bool = False
    evidence: List[ChatEvidence] = []


class ChatClarificationOption(BaseModel):
    """One concrete choice offered with a question back."""

    label: str
    query: str = ""
    action: str = "ask"
    resume_id: Optional[int] = None


class ChatClarification(BaseModel):
    """
    A question the assistant asked back, with the options that answer it.

    Raised only when the assistant genuinely cannot proceed — several real people share
    the name, or every constraint combination is empty. Each option carries a real,
    measured outcome so the recruiter chooses between facts rather than guesses.
    """

    question: str
    options: List[ChatClarificationOption] = []
    # True when "I don't know which — show me all of them" is a sensible reply.
    allow_all: bool = False
    all_label: str = ""


class ChatSuggestion(BaseModel):
    """
    A suggested next step shown under an answer.

    `action` is `ask` (re-query the assistant with `query`) or `navigate` (hand off to
    the AI Analysis page carrying `resume_id`, so JD evaluation reuses the existing
    flow rather than being duplicated in the chat).
    """

    label: str
    query: str = ""
    action: str = "ask"
    resume_id: Optional[int] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="The recruiter's question.")
    session_id: str = Field("default", description="Conversation key enabling follow-up questions.")
    limit: int = Field(5, ge=1, le=20, description="Maximum candidates to return.")
    use_llm: bool = Field(True, description="Set false to force the deterministic engine.")
    embedding_engine: Optional[str] = Field(
        None, description="'bge' (local) or 'gpu'. Defaults to EMBEDDING_ENGINE; falls back to 'bge' if unreachable."
    )


class ChatResponse(BaseModel):
    answer: str
    # What SHAPE this answer is, so the UI renders it as what it actually is:
    #   candidates    — a ranked shortlist
    #   fact          — one detail about one person
    #   explanation   — a defence of the previous answer
    #   clarification — a question back, with options
    #   refusal | empty
    answer_type: str = "candidates"
    candidates: List[ChatCandidate] = []
    fact: Optional[ChatFact] = None
    clarification: Optional[ChatClarification] = None
    refused: bool = False
    refusal_category: Optional[str] = None
    # True when the assistant asked a question back instead of guessing an answer.
    needs_clarification: bool = False
    intent: Optional[ChatIntent] = None
    suggestions: List[ChatSuggestion] = []
    diagnostics: Dict[str, Any] = {}
    elapsed_ms: int = 0


class CorpusStatus(BaseModel):
    indexed_resumes: int
    indexed_chunks: int
    database_resumes: Optional[int] = None
    in_sync: bool
    cached: bool
    # Which embedding engine's index this describes. Each engine has its own index
    # because vectors from different models cannot be compared.
    engine: str = "bge"
    engine_requested: str = "bge"
    # True when the requested engine could not serve and the local one stood in.
    engine_fell_back: bool = False
    dimension: int = 384
    store: Optional[str] = None


class CorpusSyncResult(BaseModel):
    added: int
    updated: int
    removed: int
    unchanged: int
    total_resumes: int
    total_chunks: int
    engine: str = "bge"
