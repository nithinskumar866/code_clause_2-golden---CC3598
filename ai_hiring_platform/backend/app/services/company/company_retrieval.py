"""
Retrieval over the company knowledge base.

Three ways in, tried in this order, because they answer three different questions:

1. **A named company.** "What does TCS build?" names its own answer. Resolving the name
   against the company index and filtering the search to that company is not a shortcut
   past RAG — the vector search still decides WHICH of the seven fields answers the
   question. It only stops a similarity search from confidently returning Infosys.
2. **A named person.** People carry no vectors (they live in the company payload), so a
   bare name is matched against the people index and answered from the company that
   employs them. This is the acknowledged cost of payload-only people: exact match, not
   semantic.
3. **Open search.** Nothing recognisable named — the question goes straight at the
   vectors across all 60 companies and the hits are grouped back into companies.

Grouping is the important part. A search returns FIELDS, but a recruiter thinks in
COMPANIES, so hits are folded back onto their company and a company's relevance is its
best field plus a discounted contribution from its other matching fields. Summing raw
scores would rank a company that matches weakly on six fields above one that answers the
question exactly on one.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.logging import logger
from app.services.ai import embedding_engines
from app.services.company import qdrant_store

# How much a company's second, third… matching field adds on top of its best one. Set
# below 1 so breadth is a tie-breaker rather than the ranking: a company that genuinely
# answers the question in one field must outrank one that half-answers it in five.
_SECONDARY_FIELD_WEIGHT = 0.25

# A company index this old is refetched. Companies change only when the loader runs, so
# this is about picking up a reload, not about freshness within a session.
_INDEX_TTL_SECONDS = 300

_index_lock = threading.Lock()
_index_cache: Dict[str, Any] = {"at": 0.0, "companies": [], "people": [], "df": {}, "docs": 0}


# --- The company / people index --------------------------------------------
def _company_index(force: bool = False) -> Dict[str, Any]:
    """
    Names in the store plus the corpus statistics that make them usable, cached.

    `df[token]` is the number of companies whose text uses that token. It is what lets
    the module decide, without a maintained keyword list, whether a word inside a
    company name identifies that company or is just English: "tcs" appears in one
    company's prose, "services" appears in most of them. The pool defines its own
    vocabulary, exactly as `chat_lexicon` does for resumes.
    """
    with _index_lock:
        fresh = (time.time() - float(_index_cache["at"])) < _INDEX_TTL_SECONDS
        if fresh and not force and _index_cache["companies"]:
            return _index_cache

        companies: Dict[str, str] = {}
        people: List[Dict[str, str]] = []
        seen_people: set = set()
        tokens_by_company: Dict[str, set] = {}
        try:
            for payload in qdrant_store.scroll_all(with_text=True):
                company_id = str(payload.get("company_id") or "")
                if not company_id:
                    continue
                companies[company_id] = str(payload.get("company_name") or company_id)
                tokens_by_company.setdefault(company_id, set()).update(
                    _word_set(str(payload.get("text") or ""))
                )
                for person in payload.get("people") or []:
                    name = str(person.get("name") or person.get("person") or "").strip()
                    key = (company_id, name.lower())
                    if name and key not in seen_people:
                        seen_people.add(key)
                        people.append(
                            {
                                "name": name,
                                "company_id": company_id,
                                "company_name": companies[company_id],
                            }
                        )
        except Exception as e:
            logger.warning(f"Could not refresh the company index: {e}")

        df: Dict[str, int] = {}
        for tokens in tokens_by_company.values():
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        _index_cache.update(
            {
                "at": time.time(),
                "companies": [
                    {"company_id": k, "company_name": v}
                    for k, v in sorted(companies.items(), key=lambda kv: kv[1])
                ],
                "people": people,
                "df": df,
                "docs": len(companies),
            }
        )
        return _index_cache


def refresh_index() -> None:
    """Force the next question to re-read the name index. Called after a load."""
    _company_index(force=True)


def _word_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def find_companies(message: str) -> List[Dict[str, str]]:
    """
    Which stored companies the message names.

    Matched on distinctive word overlap rather than substring: "Tata Consultancy
    Services (TCS)" must be found by "TCS" and by "tata consultancy", while the word
    "services" alone — which appears in most company names in the sheet — must match
    nothing on its own.

    A single overlapping word is enough WHEN that word is rare in the corpus. Requiring
    a fixed share of the name to match is what made "what products does TCS build"
    fail: "tcs" is one of three name words, but it is also a word that appears in
    exactly one company's text, which makes it a stronger identifier than two generic
    ones would be. Rarity is measured over the pool, never asserted by a list.
    """
    words = _word_set(message)
    if not words:
        return []

    index = _company_index()
    companies = index["companies"]
    df: Dict[str, int] = index["df"]
    docs = max(int(index["docs"]), 1)

    # A word shared by many company NAMES carries no identifying power.
    frequency: Dict[str, int] = {}
    for company in companies:
        for word in _word_set(company["company_name"]):
            frequency[word] = frequency.get(word, 0) + 1
    common = {w for w, n in frequency.items() if n > max(2, len(companies) * 0.1)}

    # A word used in many companies' TEXT is ordinary English in this pool, whatever it
    # happens to spell inside a name.
    def rare(word: str) -> bool:
        return len(word) >= 2 and df.get(word, 0) <= max(1, docs * 0.1)

    hits: List[Tuple[float, Dict[str, str]]] = []
    for company in companies:
        name_words = _word_set(company["company_name"])
        distinctive = (name_words - common) or name_words
        overlap = distinctive & words
        if not overlap:
            continue
        share = len(overlap) / max(len(distinctive), 1)
        # Either most of the name matched, or one genuinely rare word did.
        if share >= 0.5 or any(rare(word) for word in overlap):
            hits.append((share + 0.5 * len(overlap), company))

    hits.sort(key=lambda pair: -pair[0])
    return [company for _, company in hits][:4]


def find_people(message: str) -> List[Dict[str, str]]:
    """Which stored people the message names. Exact word match — see module docstring."""
    words = _word_set(message)
    if not words:
        return []
    out = []
    for person in _company_index()["people"]:
        name_words = _word_set(person["name"])
        if name_words and name_words <= words:
            out.append(person)
    return out[:4]


# --- Searching --------------------------------------------------------------
def _embed_query(message: str) -> List[float]:
    engine = embedding_engines.gpu_engine_strict()
    return engine.embed_query(message)[0].tolist()


def _group_by_company(hits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold field-level hits into companies, best field first."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        payload = hit["payload"]
        company_id = str(payload.get("company_id") or "")
        if not company_id:
            continue
        entry = grouped.setdefault(
            company_id,
            {
                "company_id": company_id,
                "company_name": payload.get("company_name") or company_id,
                "fields": payload.get("fields") or {},
                "people": payload.get("people") or [],
                "industries": payload.get("industries_list") or [],
                "matches": [],
            },
        )
        entry["matches"].append(
            {
                "field": payload.get("field") or "",
                "label": qdrant_store.FIELD_LABELS.get(
                    payload.get("field") or "", str(payload.get("field") or "").replace("_", " ")
                ),
                "text": payload.get("text") or "",
                "score": round(float(hit["score"]), 4),
            }
        )

    companies = []
    for entry in grouped.values():
        entry["matches"].sort(key=lambda m: -m["score"])
        best = entry["matches"][0]["score"]
        rest = sum(m["score"] for m in entry["matches"][1:])
        entry["relevance"] = round(best + _SECONDARY_FIELD_WEIGHT * rest, 4)
        entry["best_field"] = entry["matches"][0]["field"]
        companies.append(entry)

    companies.sort(key=lambda c: -c["relevance"])
    return companies


def search(
    message: str,
    limit: int = 5,
    company_ids: Optional[Sequence[str]] = None,
    min_similarity: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Answer-ready retrieval for one question.

    Returns `{companies, mode, named_companies, named_people, stats}`. Never raises for
    an empty result — an unanswerable question returns zero companies and the caller
    phrases the miss.
    """
    started = time.perf_counter()
    named_companies = list(company_ids or []) or [
        c["company_id"] for c in find_companies(message)
    ]
    named_people = find_people(message)
    if not named_companies and named_people:
        named_companies = list(dict.fromkeys(p["company_id"] for p in named_people))

    mode = "company" if named_companies else "open"
    vector = _embed_query(message)

    # A named company gets no similarity floor: the recruiter already told us who they
    # mean, so returning "nothing relevant" about a company they explicitly asked about
    # would be a worse answer than its weakest field.
    hits = qdrant_store.search(
        vector,
        limit=max(int(settings.COMPANY_TOP_K), limit * 4),
        company_ids=named_companies or None,
        min_similarity=0.0 if named_companies else min_similarity,
    )

    # Open search that found nothing is a real miss. A named company that found nothing
    # is not — fall back to reading its record directly.
    if not hits and named_companies:
        hits = [h for cid in named_companies for h in qdrant_store.fetch_company(cid)]

    companies = _group_by_company(hits)[:limit]
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        f"Company retrieval [{mode}] '{message[:60]}' -> {len(companies)} companies "
        f"from {len(hits)} field hits in {elapsed_ms} ms."
    )
    return {
        "companies": companies,
        "mode": mode,
        "named_companies": named_companies,
        "named_people": named_people,
        "stats": {
            "field_hits": len(hits),
            "companies_returned": len(companies),
            "companies_indexed": len(_company_index()["companies"]),
            "elapsed_ms": elapsed_ms,
            "floor": 0.0 if named_companies else float(
                settings.COMPANY_MIN_SIMILARITY if min_similarity is None else min_similarity
            ),
        },
    }
