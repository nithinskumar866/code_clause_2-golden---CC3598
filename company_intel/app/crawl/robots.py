"""
robots.txt: what the site says we may fetch, and how fast.

Honouring this is not politeness theatre. A crawler that ignores robots.txt gets its
IP blocked, and a blocked IP takes the whole knowledge base offline for every company
hosted behind the same CDN — which, for corporate sites, is most of them.

One parser per host, cached for the life of the process. A crawl of sixty pages should
fetch robots.txt once, not sixty times.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple
from urllib.robotparser import RobotFileParser

import httpx

from app.core.config import settings
from app.core.logging import logger

# host -> (parser, crawl_delay). A missing entry means "not fetched yet"; a parser of
# None means "fetched and unavailable", which is treated as permission to crawl.
_cache: Dict[str, Tuple[Optional[RobotFileParser], Optional[float]]] = {}
_lock = threading.Lock()


def _fetch(scheme: str, host: str) -> Tuple[Optional[RobotFileParser], Optional[float]]:
    url = f"{scheme}://{host}/robots.txt"
    try:
        response = httpx.get(
            url,
            timeout=min(10.0, settings.CRAWL_TIMEOUT_SECONDS),
            headers={"User-Agent": settings.CRAWL_USER_AGENT},
            follow_redirects=True,
        )
    except Exception as e:
        logger.info(f"robots.txt unreachable for {host} ({e}) — proceeding.")
        return None, None

    # A 4xx means no robots file, which is permission. A 5xx means the site is unwell;
    # the conservative reading is to stay out until it recovers.
    if response.status_code >= 500:
        logger.info(f"robots.txt returned {response.status_code} for {host} — backing off.")
        parser = RobotFileParser()
        parser.parse(["User-agent: *", "Disallow: /"])
        return parser, None
    if response.status_code >= 400:
        return None, None

    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    try:
        delay = parser.crawl_delay(settings.CRAWL_USER_AGENT)
    except Exception:
        delay = None
    return parser, (float(delay) if delay else None)


def _for_host(scheme: str, host: str) -> Tuple[Optional[RobotFileParser], Optional[float]]:
    with _lock:
        if host in _cache:
            return _cache[host]
    result = _fetch(scheme, host)
    with _lock:
        _cache[host] = result
    return result


def allowed(url: str) -> bool:
    """Whether robots.txt permits fetching this URL. Open when robots is absent."""
    if not settings.CRAWL_RESPECT_ROBOTS:
        return True
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    parser, _ = _for_host(parts.scheme or "https", parts.netloc.lower())
    if parser is None:
        return True
    try:
        return bool(parser.can_fetch(settings.CRAWL_USER_AGENT, url))
    except Exception:
        return True


def delay_for(url: str) -> float:
    """The site's requested crawl delay, or our configured default — whichever is slower."""
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    _, delay = _for_host(parts.scheme or "https", parts.netloc.lower())
    return max(float(settings.CRAWL_DEFAULT_DELAY_SECONDS), float(delay or 0.0))


def clear_cache() -> None:
    """Used by tests, and by an operator who has just edited a site's robots.txt."""
    with _lock:
        _cache.clear()
