"""
Periodic refresh: re-read only what has come due.

The chat path never crawls, so freshness is entirely this module's job. It reads the
`ci_sources` collection for live URLs whose `next_due_at` has passed, re-ingests each
one, and lets `ingest_page` decide — via ETag, then content hash — whether any real
work is needed. Most due pages cost one conditional request and nothing else.

Cadence is per page type, not global (see `classify.refresh_days`). One shared interval
is wrong in both directions simultaneously: it re-reads a leadership page that changes
once a year, and lets a newsroom sit stale for a week.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.ingest import pipeline
from app.sources import registry, state


@dataclass
class RefreshReport:
    pages_checked: int = 0
    pages_updated: int = 0
    pages_unchanged: int = 0
    pages_failed: int = 0
    chunks_written: int = 0
    duration_seconds: float = 0.0
    companies: List[str] = field(default_factory=list)
    details: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pages_checked": self.pages_checked,
            "pages_updated": self.pages_updated,
            "pages_unchanged": self.pages_unchanged,
            "pages_failed": self.pages_failed,
            "chunks_written": self.chunks_written,
            "duration_seconds": round(self.duration_seconds, 2),
            "companies": self.companies,
            "details": self.details[:100],
        }


def run(limit: Optional[int] = None, company_id: Optional[str] = None) -> RefreshReport:
    """
    Process one batch of due pages.

    Batched rather than exhaustive so a run has a bounded cost and a predictable
    duration. Anything left over is simply still due at the next run, because due-ness
    lives in the store and not in this process's memory.
    """
    started = time.time()
    report = RefreshReport()
    cap = limit or settings.REFRESH_BATCH

    due_records = list(state.due(limit=cap))
    if company_id:
        due_records = [r for r in due_records if r.get("company_id") == company_id]

    # Companies are looked up once per run, not once per page: a batch of fifty pages
    # from one company would otherwise be fifty identical registry scrolls.
    companies: Dict[str, Optional[Dict[str, Any]]] = {}

    for record in due_records:
        cid = record.get("company_id") or ""
        if cid not in companies:
            companies[cid] = registry.get(cid)
        company = companies[cid]

        if not company:
            # Crawl state outlived its company. Nothing to refresh into, so retire it
            # rather than letting it come due again on every run forever.
            state.record_failure(
                cid, record["url"], record.get("page_type", ""),
                "Company is no longer registered.", fatal=True,
            )
            continue
        if not company.get("enabled", True):
            continue

        outcome = pipeline.ingest_page(
            company, record["url"], page_type=record.get("page_type")
        )
        report.pages_checked += 1
        if cid not in report.companies:
            report.companies.append(cid)

        if outcome.status == "indexed":
            report.pages_updated += 1
            report.chunks_written += outcome.chunks
        elif outcome.status in ("unchanged", "not_modified", "empty"):
            report.pages_unchanged += 1
        else:
            report.pages_failed += 1

        report.details.append(
            {
                "company_id": cid,
                "url": outcome.url,
                "status": outcome.status,
                "chunks": outcome.chunks,
                "detail": outcome.detail,
            }
        )

    report.duration_seconds = time.time() - started
    if report.pages_checked:
        logger.info(
            f"Refresh: {report.pages_checked} checked, {report.pages_updated} updated, "
            f"{report.pages_unchanged} unchanged, {report.pages_failed} failed, "
            f"{report.duration_seconds:.1f}s."
        )
    return report


def pending_count() -> int:
    """How many pages are currently due — the number a status endpoint should show."""
    return sum(1 for _ in state.due(limit=100_000))
