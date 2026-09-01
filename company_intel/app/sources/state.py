"""
Crawl state: one record per URL, in `ci_sources`.

This is the memory that makes refresh cheap. For every page ever fetched it holds the
`ETag`, the `Last-Modified` string, the hash of the extracted text, when it was last
crawled and when it is next due. A refresh run is therefore a filtered scroll for
"due and live" rather than a re-crawl of everything.

It also holds the failure count, which is what stops a dead URL from being retried
forever. After CRAWL_MAX_FAILURES consecutive failures a page is retired: kept for the
record, excluded from scheduling, and visible in the company's status so an operator
can see that a page went away rather than wondering why answers got thinner.

None of this is vector data. It lives in Qdrant because Qdrant is the only datastore
this service is allowed, and a filtered scroll over a few thousand payloads is a
perfectly good key-value read at this scale.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.extract import classify
from app.process.hashing import url_hash
from app.store import qdrant

# `status` values.
LIVE = "live"        # fetched successfully at least once, scheduled for refresh
PENDING = "pending"  # discovered, never yet fetched
RETIRED = "retired"  # failed too many times, or blocked by robots — not scheduled
SKIPPED = "skipped"    # deliberately out of scope (deny pattern, off-site)
DUPLICATE = "duplicate"  # byte-identical to another URL already indexed for this company


def _now() -> float:
    return time.time()


def _key(company_id: str, url: str) -> str:
    return f"source:{company_id}:{url_hash(url)}"


def get(company_id: str, url: str) -> Optional[Dict[str, Any]]:
    flt = qdrant.match_filter(company_id=company_id, url_hash=url_hash(url))
    for point in qdrant.scroll(qdrant.SOURCES(), flt=flt, limit=2):
        return point["payload"]
    return None


def list_for_company(company_id: str) -> List[Dict[str, Any]]:
    flt = qdrant.match_filter(company_id=company_id)
    return [p["payload"] for p in qdrant.scroll(qdrant.SOURCES(), flt=flt)]


def due(limit: int = 0, now: Optional[float] = None) -> Iterator[Dict[str, Any]]:
    """Live URLs whose refresh has come due, oldest first."""
    flt = qdrant.due_filter(now if now is not None else _now())
    records = [p["payload"] for p in qdrant.scroll(qdrant.SOURCES(), flt=flt)]
    records.sort(key=lambda r: float(r.get("next_due_at") or 0.0))
    cap = limit or settings.REFRESH_BATCH
    for record in records[:cap]:
        yield record


def _write(company_id: str, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    qdrant.upsert_meta(qdrant.SOURCES(), _key(company_id, url), payload)
    return payload


def _base(company_id: str, url: str, page_type: str) -> Dict[str, Any]:
    existing = get(company_id, url) or {}
    return {
        "company_id": company_id,
        "url": url,
        "url_hash": url_hash(url),
        "page_type": page_type or existing.get("page_type") or classify.OTHER,
        "status": existing.get("status") or PENDING,
        "etag": existing.get("etag"),
        "last_modified": existing.get("last_modified"),
        "page_hash": existing.get("page_hash"),
        "first_seen_at": existing.get("first_seen_at") or _now(),
        "last_crawled_at": existing.get("last_crawled_at"),
        "last_changed_at": existing.get("last_changed_at"),
        "next_due_at": existing.get("next_due_at") or 0.0,
        "fail_count": int(existing.get("fail_count") or 0),
        "last_error": existing.get("last_error") or "",
        "chunk_count": int(existing.get("chunk_count") or 0),
        "http_status": existing.get("http_status"),
    }


def record_discovered(company_id: str, url: str, page_type: str) -> Dict[str, Any]:
    """Note a URL the frontier found but has not fetched yet."""
    payload = _base(company_id, url, page_type)
    if payload["status"] == PENDING:
        payload["next_due_at"] = 0.0  # due immediately
    return _write(company_id, url, payload)


def record_success(
    company_id: str,
    url: str,
    page_type: str,
    page_hash: str,
    chunk_count: int,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    changed: bool = True,
    http_status: Optional[int] = None,
) -> Dict[str, Any]:
    """
    A page was read successfully. Schedules its next refresh by page type.

    `changed=False` (a 304, or an identical hash) still counts as a success and still
    resets the schedule — the page was verified current, which is exactly what the
    refresh was for.
    """
    payload = _base(company_id, url, page_type)
    now = _now()
    payload.update(
        {
            "status": LIVE,
            "page_hash": page_hash or payload.get("page_hash"),
            "etag": etag if etag is not None else payload.get("etag"),
            "last_modified": last_modified if last_modified is not None else payload.get("last_modified"),
            "last_crawled_at": now,
            "next_due_at": now + classify.refresh_days(page_type) * 86400.0,
            "fail_count": 0,
            "last_error": "",
            "chunk_count": int(chunk_count),
            "http_status": http_status,
        }
    )
    if changed:
        payload["last_changed_at"] = now
    return _write(company_id, url, payload)


def record_failure(
    company_id: str,
    url: str,
    page_type: str,
    error: str,
    http_status: Optional[int] = None,
    fatal: bool = False,
) -> Dict[str, Any]:
    """
    A fetch failed. Retries back off, and a page that keeps failing is retired.

    `fatal=True` retires immediately — used for robots.txt refusals, where retrying is
    not a transient-error question but a permission one that will not change on its own.
    """
    payload = _base(company_id, url, page_type)
    payload["fail_count"] = int(payload.get("fail_count") or 0) + 1
    payload["last_error"] = (error or "")[:400]
    payload["http_status"] = http_status
    payload["last_crawled_at"] = _now()

    if fatal or payload["fail_count"] >= settings.CRAWL_MAX_FAILURES:
        payload["status"] = RETIRED
        payload["next_due_at"] = 0.0
        logger.info(f"Retiring {url} after {payload['fail_count']} failure(s): {error}")
    else:
        # Exponential backoff in hours, so a site having a bad afternoon is not hammered.
        backoff_hours = min(72.0, 2.0 ** payload["fail_count"])
        payload["status"] = LIVE if payload.get("page_hash") else PENDING
        payload["next_due_at"] = _now() + backoff_hours * 3600.0
    return _write(company_id, url, payload)


def find_duplicate(company_id: str, page_hash: str, exclude_url: str) -> Optional[str]:
    """
    Another URL of this company already holding exactly this content, if any.

    Client-rendered sites are the reason this exists. A single-page app serves the same
    HTML shell for every path, so twelve distinct URLs extract to one identical body —
    and the per-URL hash check cannot see it, because it only ever compares a page to
    its OWN previous version. Measured on mphasis.com: twelve pages, one distinct
    content hash, 132 chunks that were all the same homepage.

    The damage is not merely wasted space. Twelve copies of one passage outrank a
    unique passage that says something more relevant, so the duplicate crowds the real
    answer out of the results.
    """
    if not page_hash:
        return None
    mine = url_hash(exclude_url)
    flt = qdrant.match_filter(company_id=company_id, page_hash=page_hash)
    for point in qdrant.scroll(qdrant.SOURCES(), flt=flt, limit=50):
        payload = point["payload"]
        if payload.get("url_hash") != mine and payload.get("status") in (LIVE, PENDING):
            return payload.get("url")
    return None


def record_duplicate(company_id: str, url: str, page_type: str, page_hash: str,
                     canonical_url: str) -> Dict[str, Any]:
    """Note a page whose content is already indexed under another URL."""
    payload = _base(company_id, url, page_type)
    payload.update({
        "status": DUPLICATE,
        "page_hash": page_hash,
        "chunk_count": 0,
        "last_crawled_at": _now(),
        "next_due_at": 0.0,
        "last_error": f"Identical content to {canonical_url}",
    })
    return _write(company_id, url, payload)


def record_skipped(company_id: str, url: str, reason: str) -> Dict[str, Any]:
    """A URL deliberately left out of scope, kept so the decision is auditable."""
    payload = _base(company_id, url, classify.OTHER)
    payload.update({"status": SKIPPED, "last_error": reason[:400], "next_due_at": 0.0})
    return _write(company_id, url, payload)


def summarize(company_id: str) -> Dict[str, Any]:
    """Per-company crawl health, for the status endpoint."""
    records = list_for_company(company_id)
    by_status: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for record in records:
        by_status[record.get("status") or "unknown"] = by_status.get(record.get("status") or "unknown", 0) + 1
        by_type[record.get("page_type") or "other"] = by_type.get(record.get("page_type") or "other", 0) + 1

    crawled = [r for r in records if r.get("last_crawled_at")]
    failures = [
        {"url": r["url"], "error": r.get("last_error"), "fail_count": r.get("fail_count")}
        for r in records
        if r.get("status") == RETIRED or int(r.get("fail_count") or 0) > 0
    ]
    return {
        "pages_known": len(records),
        "pages_by_status": by_status,
        "pages_by_type": by_type,
        "chunks": sum(int(r.get("chunk_count") or 0) for r in records),
        "last_crawled_at": max((float(r["last_crawled_at"]) for r in crawled), default=None),
        "next_due_at": min(
            (float(r.get("next_due_at") or 0.0) for r in records if r.get("status") == LIVE),
            default=None,
        ),
        "failures": failures[:20],
    }
