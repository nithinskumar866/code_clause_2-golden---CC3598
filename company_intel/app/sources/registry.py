"""
The company registry: who we track, and where their content lives.

One point per company in `ci_companies`, carrying no meaningful vector. This is the
only place that decides what `company_id` means, and the only place that knows how to
remove a company completely — registry point, crawl state, and every chunk — so that
deletion cannot leave orphaned vectors answering questions about a company nobody
tracks any more.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence

from app.core.logging import logger
from app.crawl import urls as urlutil
from app.store import qdrant

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def make_company_id(domain: str) -> str:
    """
    A stable id derived from the registrable domain: `www.acme.co.uk` -> `acme-co-uk`.

    Derived from the domain rather than the display name because names are edited —
    "Acme" becomes "Acme Group" — and an id that moves takes every stored chunk's
    `company_id` out of sync with the registry.
    """
    registrable = urlutil.registrable_domain(domain)
    return _SLUG_STRIP.sub("-", registrable.lower()).strip("-")


def _now() -> float:
    return time.time()


def register(
    name: str,
    domain: str,
    seed_urls: Optional[Sequence[str]] = None,
    linkedin_urls: Optional[Sequence[str]] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    deny_patterns: Optional[Sequence[str]] = None,
    enabled: bool = True,
    render: bool = False,
) -> Dict[str, Any]:
    """
    Add or update one company. Idempotent on `company_id`.

    Re-registering preserves `created_at` so the registry keeps an honest record of when
    the company entered the system, not when its details were last edited.
    """
    domain = (domain or "").strip()
    if not domain:
        raise ValueError("A company needs a domain.")

    company_id = make_company_id(domain)
    if not company_id:
        raise ValueError(f"Could not derive a company id from domain {domain!r}.")

    # Seeds default to the domain root. Normalising them here means a typo'd seed is
    # rejected at registration rather than silently producing a crawl of nothing.
    raw_seeds = list(seed_urls or []) or [f"https://{urlutil.registrable_domain(domain)}"]
    seeds: List[str] = []
    for seed in raw_seeds:
        candidate = seed if "://" in seed else f"https://{seed}"
        normalized = urlutil.normalize(candidate)
        if normalized and normalized not in seeds:
            seeds.append(normalized)
    if not seeds:
        raise ValueError("None of the supplied seed URLs are fetchable web addresses.")

    existing = get(company_id)
    payload = {
        "company_id": company_id,
        "name": (name or "").strip() or urlutil.registrable_domain(domain),
        "domain": urlutil.registrable_domain(domain),
        "seed_urls": seeds,
        # Stored, never fetched. See `docs/linkedin.md` — these are surfaced to the
        # recruiter as links to open themselves, not as crawl targets.
        "linkedin_urls": [u.strip() for u in (linkedin_urls or []) if u and u.strip()],
        "allow_patterns": [p.strip() for p in (allow_patterns or []) if p and p.strip()],
        "deny_patterns": [p.strip() for p in (deny_patterns or []) if p and p.strip()],
        "enabled": bool(enabled),
        # Render this company's pages in a headless browser. Off by default; turned on
        # for single-page apps, which a plain fetch reads as one shell repeated.
        "render": bool(render),
        # Set by the crawler when it observes that most pages are duplicates of each
        # other. Diagnosis, not configuration — it explains a thin corpus rather than
        # changing behaviour, and is what the UI shows to suggest turning on `render`.
        "client_rendered": (existing or {}).get("client_rendered", False),
        "created_at": (existing or {}).get("created_at") or _now(),
        "updated_at": _now(),
    }
    qdrant.upsert_meta(qdrant.COMPANIES(), f"company:{company_id}", payload)
    logger.info(f"Registered company '{payload['name']}' ({company_id}) with {len(seeds)} seed(s).")
    return payload


def mark_client_rendered(company_id: str, value: bool) -> None:
    """Record the crawler's verdict that this site serves one shell for every path."""
    company = get(company_id)
    if not company or bool(company.get("client_rendered")) == value:
        return
    company["client_rendered"] = value
    company["updated_at"] = _now()
    qdrant.upsert_meta(qdrant.COMPANIES(), f"company:{company_id}", company)
    if value:
        logger.info(
            f"{company_id}: pages are client-rendered — enable `render` to read them."
        )


def get(company_id: str) -> Optional[Dict[str, Any]]:
    for point in qdrant.scroll(
        qdrant.COMPANIES(), flt=qdrant.match_filter(company_id=company_id), limit=2
    ):
        return point["payload"]
    return None


def list_all(enabled_only: bool = False) -> List[Dict[str, Any]]:
    flt = qdrant.match_filter(enabled=True) if enabled_only else None
    companies = [p["payload"] for p in qdrant.scroll(qdrant.COMPANIES(), flt=flt)]
    return sorted(companies, key=lambda c: (c.get("name") or "").lower())


def resolve(text: str) -> Optional[Dict[str, Any]]:
    """
    Find the company a question names, or None.

    Matching is on the registered name and domain only. This never guesses from partial
    similarity: answering a question about Acme with Acme Logistics' data because the
    names overlap is worse than treating the question as a pool-wide search.
    """
    haystack = f" {(text or '').lower()} "
    best: Optional[Dict[str, Any]] = None
    for company in list_all():
        for candidate in (company.get("name") or "", company.get("domain") or ""):
            candidate = candidate.lower().strip()
            if len(candidate) < 3:
                continue
            if f" {candidate} " in haystack or candidate in haystack.replace(" ", ""):
                # Prefer the longest match, so "Acme Logistics" beats "Acme".
                if best is None or len(candidate) > len(best.get("name") or ""):
                    best = company
    return best


def delete(company_id: str) -> Dict[str, int]:
    """
    Remove a company and everything derived from it, across all three collections.

    Content first: if the process dies half way, an orphaned registry entry is a
    harmless empty company, whereas orphaned chunks would keep answering questions with
    data the operator believed they had deleted.
    """
    flt = qdrant.match_filter(company_id=company_id)
    if flt is None:
        raise ValueError("A company id is required.")

    chunks = qdrant.count(qdrant.CONTENT(), flt)
    pages = qdrant.count(qdrant.SOURCES(), flt)

    qdrant.delete_by_filter(qdrant.CONTENT(), flt)
    qdrant.delete_by_filter(qdrant.SOURCES(), flt)
    qdrant.delete_by_filter(qdrant.COMPANIES(), flt)

    logger.info(f"Deleted company {company_id}: {chunks} chunks, {pages} pages.")
    return {"chunks_deleted": chunks, "pages_deleted": pages}
