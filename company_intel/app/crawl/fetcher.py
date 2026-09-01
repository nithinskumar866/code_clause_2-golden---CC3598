"""
Fetching one page, politely and cheaply.

Two things here matter more than the HTTP mechanics.

**Conditional GET.** Every fetch sends the `ETag` and `Last-Modified` recorded on the
last successful crawl. When the server answers `304 Not Modified` the refresh costs one
round trip: no body transferred, no HTML parsed, no text extracted, no hash computed,
no vectors written. On a corpus that is mostly static between refreshes, this is the
single largest saving in the system, and it is free because the server does the work.

**A rate gate per host.** Requests to one host are spaced by the delay robots.txt asks
for (or our configured floor, whichever is slower). Concurrency is still allowed, so a
slow-responding host does not stall the crawl — but the request *rate* stays inside
what the site asked for.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.crawl import robots

_last_request_at: Dict[str, float] = {}
_gate_lock = threading.Lock()

_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()


@dataclass
class FetchResult:
    """
    What one fetch attempt produced.

    `status` is the outcome the pipeline branches on, not the HTTP code:
      ok            fresh HTML to process
      not_modified  the server confirmed nothing changed — skip everything downstream
      blocked       robots.txt disallows this URL
      failed        transport error, 4xx/5xx, wrong content type, or too large
    """

    url: str
    status: str
    http_status: Optional[int] = None
    html: str = ""
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    content_type: str = ""
    error: str = ""
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def get_client() -> httpx.Client:
    """One connection pool for the process — reconnecting per page is most of the cost."""
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                timeout=httpx.Timeout(settings.CRAWL_TIMEOUT_SECONDS),
                follow_redirects=True,
                headers={
                    "User-Agent": settings.CRAWL_USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                    "Accept-Language": "en",
                },
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            )
        return _client


def close_client() -> None:
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


def _wait_turn(url: str) -> None:
    """Block until this host's rate gate opens."""
    from app.crawl.urls import host_of

    host = host_of(url)
    delay = robots.delay_for(url)
    while True:
        with _gate_lock:
            now = time.monotonic()
            earliest = _last_request_at.get(host, 0.0) + delay
            if now >= earliest:
                _last_request_at[host] = now
                return
            sleep_for = earliest - now
        time.sleep(min(sleep_for, delay))


def fetch(
    url: str,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
) -> FetchResult:
    """
    Fetch one URL, sending validators when we hold them.

    Retries only transport errors and 5xx. A 404 is an answer, not a hiccup, and
    retrying it three times just spends the site's budget on a page that is gone.
    """
    if not robots.allowed(url):
        logger.info(f"robots.txt disallows {url}")
        return FetchResult(url=url, status="blocked", error="Disallowed by robots.txt")

    headers = {}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    last_error = ""
    for attempt in range(1, max(1, settings.CRAWL_MAX_RETRIES) + 1):
        _wait_turn(url)
        try:
            response = get_client().get(url, headers=headers)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < settings.CRAWL_MAX_RETRIES:
                time.sleep(min(8.0, 2.0 ** attempt))
                continue
            return FetchResult(url=url, status="failed", error=last_error)

        code = response.status_code

        if code == 304:
            return FetchResult(
                url=url,
                status="not_modified",
                http_status=304,
                etag=etag,
                last_modified=last_modified,
            )

        if code >= 500 and attempt < settings.CRAWL_MAX_RETRIES:
            last_error = f"HTTP {code}"
            time.sleep(min(8.0, 2.0 ** attempt))
            continue

        if code >= 400:
            return FetchResult(
                url=url, status="failed", http_status=code, error=f"HTTP {code}"
            )

        content_type = (response.headers.get("content-type") or "").lower()
        if "html" not in content_type and "xml" not in content_type:
            return FetchResult(
                url=url,
                status="failed",
                http_status=code,
                content_type=content_type,
                error=f"Not HTML ({content_type or 'unknown'})",
            )

        body = response.content or b""
        if len(body) > settings.CRAWL_MAX_BYTES:
            return FetchResult(
                url=url,
                status="failed",
                http_status=code,
                content_type=content_type,
                error=f"Body too large ({len(body)} bytes)",
            )

        return FetchResult(
            url=str(response.url),
            status="ok",
            http_status=code,
            html=response.text,
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
            content_type=content_type,
            headers=dict(response.headers),
        )

    return FetchResult(url=url, status="failed", error=last_error or "Exhausted retries")
