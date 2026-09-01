"""
API contract for the Prompt Lab (Golden Rule 9 — the backend owns these shapes and the
frontend mirrors them in `types/index.ts`).

A case is deliberately self-describing: the question, the context retrieval would have
supplied, the thread so far, and what this particular turn makes permissible. That last
part is what lets one suite grade a rewritten prompt without itself being rewritten.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CaseExpectations(BaseModel):
    """What this turn permits. Every field is a property of the question, not the prompt."""

    max_lines: Optional[int] = Field(2, description="Maximum non-empty lines; null disables the check.")
    max_chars: Optional[int] = Field(None, description="Maximum characters; null disables the check.")
    link_allowed: bool = Field(False, description="A profile link is permitted on this turn.")
    link_required: bool = Field(False, description="A profile link must appear on this turn.")
    years_known: bool = Field(False, description="The context states exact years, so the answer must too.")
    expected_years: Optional[str] = Field(None, description="The exact figure the context carries.")
    jobs_requested: bool = Field(False, description="The user actually asked to search/list jobs.")
    list_allowed: bool = Field(False, description="An enumerated answer is appropriate here.")
    skills_requested: bool = Field(False, description="The user asked about the person's skills.")
    must_contain: List[str] = Field(default_factory=list)
    must_not_contain: List[str] = Field(default_factory=list)


class PromptCase(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    question: str
    context: str = ""
    history: List[Dict[str, str]] = Field(
        default_factory=list, description="Prior turns as {role, content}."
    )
    expectations: CaseExpectations = Field(default_factory=CaseExpectations)


class PromptVariant(BaseModel):
    label: str = "A"
    prompt: str


class RunRequest(BaseModel):
    variants: List[PromptVariant] = Field(..., min_length=1, max_length=4)
    cases: List[PromptCase] = Field(..., min_length=1)
    temperature: float = 0.2
    rules: Optional[List[str]] = Field(None, description="Limit grading to these rule ids.")
    suite_id: Optional[int] = Field(None, description="Record the run against this suite's history.")
    persist: bool = Field(True, description="Store the run so it shows up in the regression history.")


class ScoreRequest(BaseModel):
    """Grade an answer that already exists — the playground, or a pasted transcript."""

    prompt: str = ""
    case: PromptCase
    answer: str
    rules: Optional[List[str]] = None


class RuleVerdict(BaseModel):
    rule: str
    title: str
    status: str = Field(..., description="pass | fail | na")
    reason: str


class CaseResult(BaseModel):
    case_id: str
    name: str
    question: str
    answer: str
    error: Optional[str] = None
    latency_ms: int = 0
    rules: List[RuleVerdict]
    passed: int
    failed: int
    not_applicable: int
    score: Optional[float] = None
    violations: List[str] = Field(default_factory=list)


class RuleBreakdown(BaseModel):
    rule: str
    title: str
    passed: int
    failed: int
    not_applicable: int
    score: Optional[float] = None


class VariantResult(BaseModel):
    label: str
    prompt_hash: str
    cases: List[CaseResult]
    total_cases: int
    clean_cases: int
    passed: int
    failed: int
    score: Optional[float] = None
    by_rule: List[RuleBreakdown]
    errors: List[str] = Field(default_factory=list)


class RuleDelta(BaseModel):
    rule: str
    title: str
    baseline_score: Optional[float] = None
    variant_score: Optional[float] = None
    delta: Optional[float] = None


class RunResult(BaseModel):
    variants: List[VariantResult]
    deltas: List[RuleDelta] = Field(default_factory=list)
    llm_available: bool
    model: str


class ScoreResult(BaseModel):
    rules: List[RuleVerdict]
    passed: int
    failed: int
    not_applicable: int
    score: Optional[float] = None
    violations: List[str] = Field(default_factory=list)


class RuleInfo(BaseModel):
    id: str
    title: str
    description: str


class SuiteBase(BaseModel):
    name: str
    description: str = ""
    prompt: str
    cases: List[PromptCase]


class SuiteCreate(SuiteBase):
    pass


class SuiteUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    prompt: Optional[str] = None
    cases: Optional[List[PromptCase]] = None


class Suite(SuiteBase):
    id: int
    created_at: str
    updated_at: str


class RunSummary(BaseModel):
    """One row of the regression history — enough to spot a drop without loading detail."""

    id: int
    suite_id: Optional[int] = None
    label: str
    prompt_hash: str
    model: str
    total_cases: int
    clean_cases: int
    passed: int
    failed: int
    score: Optional[float] = None
    created_at: str


class RunDetail(RunSummary):
    results: Dict[str, Any]
