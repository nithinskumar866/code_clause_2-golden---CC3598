"""
Retrieval: the question in, evidence out. This path never crawls.

That separation is the point of the whole architecture. Answer latency depends on
Qdrant and the embedding model, never on whether a company's website is up, slow, or
rate-limiting us today. Ingestion is a scheduled concern; answering is not.

Two things shape the results beyond plain similarity:

**Company scoping.** When the question names a registered company, the search is
filtered to it. Without that filter, a question about Acme's products competes against
every other company's product page and can be answered with a competitor's copy — the
failure is invisible, because the answer is fluent and the citation is a real URL.

**Per-company diversity.** For a pool-wide question, an unfiltered top-k is routinely
eight chunks from the one company whose site is most verbose. Capping the chunks any
single company contributes is what turns "which companies do X" into an answer about
several companies instead of a deep read of one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.embed import engine
from app.extract import classify
from app.sources import registry
from app.store import qdrant

# Question shapes that clearly point at one kind of page. Used only to BOOST, never to
# filter: a leadership question answered from an About page is still a good answer, and
# a hard filter here would discard it.
_INTENT_HINTS = (
    ("leadership", ("ceo", "founder", "cto", "cfo", "coo", "chairman", "director",
                    "who leads", "who runs", "leadership", "management", "executive")),
    ("careers", ("hiring", "careers", "jobs", "openings", "work culture", "benefits",
                 "life at", "employee")),
    ("products", ("product", "platform", "pricing", "features", "app", "tool")),
    ("services", ("service", "solution", "offering", "consulting", "capability")),
    ("clients", ("client", "customer", "case study", "portfolio", "who do they work")),
    ("contact", ("contact", "address", "office", "location", "phone", "email")),
)

_BOOST = 0.03  # small on purpose: a nudge in ordering, never an override of meaning


@dataclass
class Evidence:
    """One retrieved chunk, with everything needed to cite it."""

    company_id: str
    company_name: str
    text: str
    page_url: str
    page_title: str
    page_type: str
    section: str
    score: float
    source_type: str
    crawled_at: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)


@dataclass
class RetrievalResult:
    question: str
    evidence: List[Evidence] = field(default_factory=list)
    company: Optional[Dict[str, Any]] = None
    scope: str = "pool"        # 'company' when narrowed to one, else 'pool'
    intent: str = ""
    searched: bool = True

    @property
    def empty(self) -> bool:
        return not self.evidence


def detect_intent(question: str) -> str:
    """Which facet the question is reaching for, or '' when it is not obvious."""
    low = f" {(question or '').lower()} "
    for label, phrases in _INTENT_HINTS:
        if any(phrase in low for phrase in phrases):
            return label
    return ""


def retrieve(
    question: str,
    company_id: Optional[str] = None,
    limit: int = 8,
    per_company_cap: int = 3,
    min_similarity: Optional[float] = None,
) -> RetrievalResult:
    """
    Find the evidence that answers one question.

    `company_id` given explicitly always wins over the name detected in the text — an
    API caller that has already resolved the company should not have that decision
    silently overridden by a company name mentioned in passing.
    """
    question = (question or "").strip()
    if not question:
        return RetrievalResult(question=question, searched=False)

    company = registry.get(company_id) if company_id else registry.resolve(question)
    scope = "company" if company else "pool"
    intent = detect_intent(question)

    flt = qdrant.match_filter(company_id=company["company_id"]) if company else None

    # Over-fetch, then trim. The diversity cap and the intent boost both need more
    # candidates than the caller asked for in order to have anything to choose between.
    hits = qdrant.search(
        engine.embed_query(question),
        limit=max(limit * 4, 24),
        flt=flt,
        min_similarity=min_similarity,
    )

    scored: List[Evidence] = []
    for hit in hits:
        payload = hit["payload"]
        score = hit["score"]
        if intent and payload.get("page_type") == intent:
            score += _BOOST
        scored.append(
            Evidence(
                company_id=payload.get("company_id", ""),
                company_name=payload.get("company_name", ""),
                text=payload.get("text", ""),
                page_url=payload.get("page_url", ""),
                page_title=payload.get("page_title", ""),
                page_type=payload.get("page_type", classify.OTHER),
                section=payload.get("section", ""),
                score=round(float(score), 4),
                source_type=payload.get("source_type", "website"),
                crawled_at=payload.get("crawled_at"),
            )
        )

    scored.sort(key=lambda e: e.score, reverse=True)

    # The diversity cap only applies to a pool-wide question. When the caller has
    # already narrowed to one company, depth on that company IS the answer.
    if scope == "company":
        return RetrievalResult(
            question=question,
            evidence=scored[:limit],
            company=company,
            scope=scope,
            intent=intent,
        )

    seen: Dict[str, int] = {}
    kept: List[Evidence] = []
    for item in scored:
        if seen.get(item.company_id, 0) >= per_company_cap:
            continue
        seen[item.company_id] = seen.get(item.company_id, 0) + 1
        kept.append(item)
        if len(kept) >= limit:
            break

    return RetrievalResult(
        question=question, evidence=kept, company=company, scope=scope, intent=intent
    )


def group_by_company(evidence: List[Evidence]) -> List[Dict[str, Any]]:
    """Reassemble flat chunk hits into per-company results for the UI."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in evidence:
        entry = grouped.setdefault(
            item.company_id,
            {
                "company_id": item.company_id,
                "company_name": item.company_name,
                "relevance": 0.0,
                "matches": [],
            },
        )
        entry["relevance"] = max(entry["relevance"], item.score)
        entry["matches"].append(
            {
                "page_url": item.page_url,
                "page_title": item.page_title,
                "page_type": item.page_type,
                "section": item.section,
                "text": item.text,
                "score": item.score,
            }
        )
    return sorted(grouped.values(), key=lambda c: c["relevance"], reverse=True)
