"""
Ingestion: the only place that writes to the content collection.

    fetch → 304? stop · extract → hash → unchanged? stop
          → classify → chunk → embed → replace the page's chunks → record state

Two short-circuits carry the whole refresh economy. A `304 Not Modified` ends the work
before the body is even transferred; an unchanged content hash ends it before chunking
and embedding, which are the expensive steps. On a corpus that is mostly static between
refreshes, the great majority of pages exit at one of those two points.

The write itself is **delete-then-upsert, scoped to one page**. Deterministic ids
overwrite chunks that still exist, but they cannot remove chunks that no longer do: a
page edited from nine chunks down to four would leave chunks five to nine in the index
permanently, still matching questions, still cited with a real URL, quoting text that
was deleted from the site months ago. Deleting the page's chunks by `url_hash` first is
what makes an update an update rather than an accumulation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.crawl import browser, fetcher
from app.crawl.frontier import Frontier
from app.embed import engine
from app.extract import blocked, classify, html_text
from app.process import chunker
from app.process.hashing import content_hash, url_hash
from app.sources import registry, state
from app.store import qdrant

SOURCE_WEBSITE = "website"


@dataclass
class PageOutcome:
    url: str
    status: str            # indexed | unchanged | not_modified | empty | blocked | failed
    page_type: str = ""
    chunks: int = 0
    detail: str = ""
    # Links parsed from this page, carried out so the frontier never has to re-fetch it
    # to discover them. Excluded from the API response — see `CrawlReport.as_dict`.
    links: List[str] = field(default_factory=list)


@dataclass
class CrawlReport:
    company_id: str
    company_name: str = ""
    pages_fetched: int = 0
    pages_indexed: int = 0
    pages_unchanged: int = 0
    pages_duplicate: int = 0
    pages_failed: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    budget_exhausted: bool = False
    client_rendered: bool = False
    advice: str = ""
    duration_seconds: float = 0.0
    outcomes: List[PageOutcome] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "pages_fetched": self.pages_fetched,
            "pages_indexed": self.pages_indexed,
            "pages_unchanged": self.pages_unchanged,
            # Reported separately because a crawl of a single-page app can return
            # "12 pages" while holding one page's worth of information.
            "pages_duplicate": self.pages_duplicate,
            "pages_failed": self.pages_failed,
            "chunks_written": self.chunks_written,
            "chunks_deleted": self.chunks_deleted,
            "budget_exhausted": self.budget_exhausted,
            "client_rendered": self.client_rendered,
            "advice": self.advice,
            "duration_seconds": round(self.duration_seconds, 2),
            "outcomes": [
                {k: v for k, v in vars(o).items() if k != "links"} for o in self.outcomes
            ],
            # Capped: a large site can reject thousands of off-site links, and a
            # response listing all of them helps nobody.
            "skipped_sample": dict(list(self.skipped.items())[:25]),
            "skipped_total": len(self.skipped),
        }


# --- one page ---------------------------------------------------------------
def ingest_page(
    company: Dict[str, Any],
    url: str,
    page_type: Optional[str] = None,
    force: bool = False,
) -> PageOutcome:
    """
    Fetch, process and store one page. The unit of work for both crawl and refresh.

    `force=True` skips both short-circuits. It exists for the case where the stored
    chunks are wrong but the page is unchanged — a chunker fix, or a model change —
    where the whole point is to redo work the hashes say is unnecessary.
    """
    company_id = company["company_id"]
    previous = state.get(company_id, url) or {}
    guessed_type = page_type or previous.get("page_type") or classify.classify(url)

    if company.get("render"):
        # Rendering has no ETag to send, so it always pays a full load. robots.txt is
        # still checked first — rendering reads JavaScript, it does not bypass rules.
        from app.crawl import robots

        if not robots.allowed(url):
            state.record_failure(
                company_id, url, guessed_type, "Disallowed by robots.txt", fatal=True
            )
            return PageOutcome(
                url=url, status="blocked", page_type=guessed_type,
                detail="Disallowed by robots.txt",
            )
        result = browser.fetch(url)
        # Rendering must never return less than not rendering. Measured on ltm.com, a
        # render yielded 426 characters where a plain fetch returned 2217 — a page
        # whose content is server-sent and whose JavaScript replaces it with a shell.
        # Falling back costs one extra request on the pages where it happens, and
        # prevents rendering from quietly making a corpus worse.
        if result.ok:
            rendered_text = html_text.extract(result.html, result.url).text
            if len(rendered_text) < settings.BROWSER_MIN_TEXT_CHARS:
                plain = fetcher.fetch(url)
                if plain.ok and len(html_text.extract(plain.html, plain.url).text) > len(rendered_text):
                    logger.info(f"Render returned less than a plain fetch for {url} — using plain.")
                    result = plain
        if not result.ok:
            result = fetcher.fetch(url)
    else:
        result = fetcher.fetch(
            url,
            etag=None if force else previous.get("etag"),
            last_modified=None if force else previous.get("last_modified"),
        )

    if result.status == "blocked":
        state.record_failure(company_id, url, guessed_type, result.error, fatal=True)
        return PageOutcome(url=url, status="blocked", page_type=guessed_type, detail=result.error)

    if result.status == "not_modified":
        state.record_success(
            company_id,
            url,
            guessed_type,
            page_hash=previous.get("page_hash") or "",
            chunk_count=int(previous.get("chunk_count") or 0),
            etag=previous.get("etag"),
            last_modified=previous.get("last_modified"),
            changed=False,
            http_status=304,
        )
        return PageOutcome(url=url, status="not_modified", page_type=guessed_type)

    if not result.ok:
        # 401/403 is the site refusing automated access — a permission answer, not a
        # transient one. Retrying it on a backoff schedule spends the site's budget and
        # ours forever on a decision that will not change by itself.
        forbidden = result.http_status in (401, 403)
        detail = result.error
        if forbidden:
            detail = (
                f"{result.error} — this site blocks automated access "
                f"(often an edge WAF refusing datacenter traffic). It cannot be crawled."
            )
        state.record_failure(
            company_id, url, guessed_type, detail,
            http_status=result.http_status, fatal=forbidden,
        )
        return PageOutcome(url=url, status="failed", page_type=guessed_type, detail=detail)

    # A redirect means the page we got is not at the URL we asked for, and that landed
    # URL is what gets cited. Normalising it keeps tracking parameters picked up from a
    # redirect chain out of the citation a recruiter is shown — one live crawl stored
    # `.../home.html?utm_source=chatgpt.com` as a source link.
    from app.crawl.urls import normalize as _normalize

    landed_url = _normalize(result.url) or result.url
    page = html_text.extract(result.html, landed_url)
    page_type_final = classify.classify(landed_url, title=page.title, h1=page.h1)

    # A 200 that carries a bot wall instead of content. Recorded as a fatal failure
    # rather than indexed: these pages extract and embed perfectly well, which is
    # exactly why they poison a corpus silently.
    if blocked.looks_blocked(page.text):
        detail = blocked.reason(page.text)
        state.record_failure(
            company_id, url, page_type_final, detail,
            http_status=result.http_status, fatal=True,
        )
        return PageOutcome(
            url=landed_url, status="blocked", page_type=page_type_final, detail=detail
        )

    if page.empty:
        # Recorded as a success, not a failure: the fetch worked and there is genuinely
        # nothing to index. Treating it as a failure would retry it forever.
        state.record_success(
            company_id,
            url,
            page_type_final,
            page_hash=content_hash(page.text),
            chunk_count=0,
            etag=result.etag,
            last_modified=result.last_modified,
            changed=False,
            http_status=result.http_status,
        )
        return PageOutcome(
            url=url,
            status="empty",
            page_type=page_type_final,
            detail="No extractable content.",
            links=page.links,
        )

    fresh_hash = content_hash(page.text)
    if not force and fresh_hash and fresh_hash == previous.get("page_hash"):
        state.record_success(
            company_id,
            url,
            page_type_final,
            page_hash=fresh_hash,
            chunk_count=int(previous.get("chunk_count") or 0),
            etag=result.etag,
            last_modified=result.last_modified,
            changed=False,
            http_status=result.http_status,
        )
        return PageOutcome(url=url, status="unchanged", page_type=page_type_final)

    # Is this page's content already indexed under a different URL of this company?
    # Checked before chunking, because on a single-page app EVERY path returns the same
    # shell and the whole crawl budget otherwise buys one page stored a dozen times.
    twin = state.find_duplicate(company_id, fresh_hash, url)
    if twin:
        # Drop anything previously stored for this URL, so a page that BECOMES a
        # duplicate does not leave its old passages behind competing with the original.
        qdrant.delete_by_filter(
            qdrant.CONTENT(), qdrant.match_filter(company_id=company_id, url_hash=url_hash(url))
        )
        state.record_duplicate(company_id, url, page_type_final, fresh_hash, twin)
        return PageOutcome(
            url=landed_url,
            status="duplicate",
            page_type=page_type_final,
            detail=f"Identical content to {twin}",
            links=page.links,
        )

    chunks = chunker.chunk_page(page)
    if not chunks:
        state.record_success(
            company_id, url, page_type_final, page_hash=fresh_hash, chunk_count=0,
            etag=result.etag, last_modified=result.last_modified,
            changed=True, http_status=result.http_status,
        )
        return PageOutcome(url=url, status="empty", page_type=page_type_final,
                           detail="Nothing survived chunking.", links=page.links)

    vectors = engine.embed_documents([c.text for c in chunks])
    page_key = url_hash(url)
    crawled_at = time.time()

    points = [
        {
            "id": qdrant.point_id("chunk", page_key, str(chunk.index)),
            "vector": vector,
            "payload": {
                "company_id": company_id,
                "company_name": company.get("name", ""),
                "source_type": SOURCE_WEBSITE,
                "page_url": landed_url,
                "url_hash": page_key,
                "page_type": page_type_final,
                "page_title": page.title,
                "section": chunk.section,
                # The prose alone. The heading path is in `section`, so an answer can
                # quote the page without the heading spliced into the sentence.
                "text": chunk.body,
                "chunk_index": chunk.index,
                "page_hash": fresh_hash,
                "crawled_at": crawled_at,
                "lang": page.lang,
            },
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    page_filter = qdrant.match_filter(company_id=company_id, url_hash=page_key)
    qdrant.delete_by_filter(qdrant.CONTENT(), page_filter)
    qdrant.upsert(qdrant.CONTENT(), points)

    state.record_success(
        company_id,
        url,
        page_type_final,
        page_hash=fresh_hash,
        chunk_count=len(points),
        etag=result.etag,
        last_modified=result.last_modified,
        changed=True,
        http_status=result.http_status,
    )
    return PageOutcome(
        url=landed_url,
        status="indexed",
        page_type=page_type_final,
        chunks=len(points),
        links=page.links,
    )


# --- a whole company --------------------------------------------------------
def crawl_company(
    company_id: str,
    max_pages: Optional[int] = None,
    max_depth: Optional[int] = None,
    force: bool = False,
) -> CrawlReport:
    """
    Discover and ingest a company's site, breadth-limited and priority-ordered.

    Links are harvested from every fetched page, but only pages that were actually
    fetched can contribute them — so a 304 or an unchanged page ends that branch of
    discovery. That is the right trade for a refresh, and it is why a periodic full
    re-discovery (`force=True`) is worth running occasionally: it is the only way a
    page linked solely from an unchanged page is ever found.
    """
    started = time.time()
    company = registry.get(company_id)
    if not company:
        raise ValueError(f"No company registered with id {company_id!r}.")

    report = CrawlReport(company_id=company_id, company_name=company.get("name", ""))
    if company.get("render"):
        logger.info(f"{company_id}: rendering pages in a headless browser.")
    frontier = Frontier(
        domain=company["domain"],
        allow_patterns=list(company.get("allow_patterns") or []),
        deny_patterns=list(company.get("deny_patterns") or []),
        max_pages=max_pages or settings.CRAWL_MAX_PAGES,
        max_depth=max_depth or settings.CRAWL_MAX_DEPTH,
    )

    for seed in company.get("seed_urls") or []:
        frontier.offer(seed, depth=0)

    logger.info(f"Crawling {company.get('name')} ({company_id}) — {frontier.queued} seed(s).")

    while True:
        candidate = frontier.next()
        if candidate is None:
            break

        state.record_discovered(company_id, candidate.url, candidate.page_type)
        outcome = ingest_page(company, candidate.url, candidate.page_type, force=force)
        # A redirect means the page we read lives at a different address than the one we
        # asked for. Marking it seen stops that address being queued again when some
        # other page links to it directly.
        if outcome.url != candidate.url:
            frontier.mark_seen(outcome.url)
            # ...and when it lands on a WHOLE DIFFERENT domain, that domain is this
            # company's site now. ltimindtree.com redirects to ltm.com, and without
            # this the crawl reads one page and rejects every link on it as off-site.
            frontier.adopt_domain_of(outcome.url)
        report.outcomes.append(outcome)
        report.pages_fetched += 1

        if outcome.status == "indexed":
            report.pages_indexed += 1
            report.chunks_written += outcome.chunks
        elif outcome.status in ("unchanged", "not_modified", "empty", "duplicate"):
            report.pages_unchanged += 1
            if outcome.status == "duplicate":
                report.pages_duplicate += 1
        else:
            report.pages_failed += 1

        # Only a page we actually read can extend the frontier — a 304 or an unchanged
        # hash returns no links, which is the cost of the short-circuit and the reason a
        # periodic `force=True` pass is worth running.
        if outcome.links and candidate.depth < frontier.max_depth:
            frontier.offer_all(outcome.links, depth=candidate.depth + 1, base=candidate.url)

    report.skipped = dict(frontier.skipped)
    report.budget_exhausted = frontier.budget_exhausted
    report.duration_seconds = time.time() - started

    # A crawl where most pages turned out to be copies of one another is not a crawl of
    # a small site — it is a single-page app serving one shell for every path. Recorded
    # on the company so the UI can say so, because "12 pages indexed" reads as coverage
    # when the truth is one page's worth of information.
    considered = report.pages_indexed + report.pages_duplicate
    report.client_rendered = bool(
        considered >= 3
        and report.pages_duplicate / considered >= settings.SPA_DUPLICATE_RATIO
        and not company.get("render")
    )
    registry.mark_client_rendered(company_id, report.client_rendered)
    if report.client_rendered:
        report.advice = (
            f"{report.pages_duplicate} of {considered} pages held identical content — "
            f"this site renders its pages in the browser. Turn on 'Render JavaScript' "
            f"for {company.get('name')} and crawl again to read them."
        )

    if report.budget_exhausted:
        logger.info(
            f"{company_id}: page cap ({frontier.max_pages}) reached with "
            f"{frontier.queued} URL(s) still queued — coverage is partial."
        )
    if report.pages_duplicate:
        logger.info(
            f"{company_id}: {report.pages_duplicate} page(s) held content already indexed "
            f"under another URL — likely a client-rendered site serving one shell."
        )
    logger.info(
        f"{company_id}: {report.pages_indexed} indexed, {report.pages_unchanged} unchanged "
        f"({report.pages_duplicate} duplicate), {report.pages_failed} failed, "
        f"{report.chunks_written} chunks, {report.duration_seconds:.1f}s."
    )
    return report


