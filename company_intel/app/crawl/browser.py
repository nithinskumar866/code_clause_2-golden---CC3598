"""
Fetching a page the way a browser sees it.

Most corporate sites are now single-page apps: the server returns one HTML shell for
every path and the real content is assembled by JavaScript. To a plain HTTP fetch,
twelve different URLs look like twelve copies of the homepage — which is exactly what a
crawl of mphasis.com produced, twelve pages holding one page's worth of information.

So this module renders. It is the expensive path and stays **opt-in per company**
(`render: true` in the registry), because it costs a browser process, a few hundred
megabytes of Chromium, and seconds rather than milliseconds per page. Sites that serve
real HTML should never pay that.

What it deliberately does NOT do: defeat bot protection. It sends the same identifying
User-Agent as the plain fetcher, still honours robots.txt upstream, and a site that
refuses automated traffic is still refused. Rendering is here to read JavaScript, not
to pretend to be someone else.
"""
from __future__ import annotations

import threading
from typing import Optional

from app.core.config import settings
from app.core.logging import logger
from app.crawl.fetcher import FetchResult

_playwright = None
_browser = None
_lock = threading.Lock()

# Assets that cost bandwidth and time but never contribute text. Blocking them is most
# of the difference between a render taking two seconds and taking eight.
_BLOCKED_RESOURCES = {"image", "media", "font"}


def available() -> tuple:
    """`(ok, detail)` — whether rendering can actually be used right now."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False, "playwright is not installed (pip install playwright)"
    try:
        _get_browser()
        return True, ""
    except Exception as e:
        return False, str(e)


def _get_browser():
    """
    One Chromium instance for the process, launched on first use.

    Launching per page would dominate the cost — a browser start is roughly a second,
    and a page render is less than that once it is warm.
    """
    global _playwright, _browser
    with _lock:
        if _browser is None:
            from playwright.sync_api import sync_playwright

            logger.info("Starting headless Chromium for rendered crawling...")
            _playwright = sync_playwright().start()
            _browser = _playwright.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--no-sandbox"],
            )
            logger.info("Chromium ready.")
        return _browser


def close() -> None:
    """Shut the browser down. Called on application shutdown and by tests."""
    global _playwright, _browser
    with _lock:
        if _browser is not None:
            try:
                _browser.close()
            except Exception:
                pass
            _browser = None
        if _playwright is not None:
            try:
                _playwright.stop()
            except Exception:
                pass
            _playwright = None


def _scroll_through(page) -> None:
    """
    Walk down the page a viewport at a time so lazy content is asked to load.

    Bounded by both a step count and a stable-height check: an infinite-scroll feed
    would otherwise keep growing for as long as we are willing to wait, and a company
    site has no such page worth reading to the end anyway.
    """
    import time as _time

    deadline = _time.monotonic() + settings.BROWSER_SCROLL_BUDGET_SECONDS
    try:
        previous_height = 0
        for _ in range(settings.BROWSER_SCROLL_STEPS):
            if _time.monotonic() > deadline:
                break  # a hard wall, so one scroll-jacking page cannot stall a crawl
            page.mouse.wheel(0, 1400)
            page.wait_for_timeout(settings.BROWSER_SCROLL_PAUSE_MS)
            height = page.evaluate("document.body ? document.body.scrollHeight : 0")
            if height == previous_height:
                break  # nothing new appeared; stop paying for scrolls
            previous_height = height
        # Back to the top: some sites hide the header content once scrolled, and the
        # capture should see the page as a reader first meets it.
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(150)
    except Exception:
        pass  # scrolling is an enhancement; a page that refuses it is still readable


def fetch(url: str) -> FetchResult:
    """
    Render one URL and return its HTML, in the same shape the plain fetcher returns.

    Conditional GET is not available here — a browser navigation has no ETag to send —
    so a rendered page always costs a full load. That is the other reason rendering is
    opt-in: it forfeits the cheapest of the three refresh short-circuits, leaving only
    the content hash to stop the work before chunking and embedding.
    """
    try:
        browser = _get_browser()
    except Exception as e:
        return FetchResult(url=url, status="failed", error=f"Browser unavailable: {e}")

    context = None
    try:
        context = browser.new_context(
            user_agent=settings.CRAWL_USER_AGENT,
            locale="en-US",
            viewport={"width": 1366, "height": 900},
        )
        context.set_default_timeout(settings.BROWSER_TIMEOUT_SECONDS * 1000)

        def route(handler):
            if handler.request.resource_type in _BLOCKED_RESOURCES:
                return handler.abort()
            return handler.continue_()

        context.route("**/*", route)

        page = context.new_page()
        # Navigate on `domcontentloaded`, which always fires, then ASK for network
        # quiet separately with its own budget. Waiting for `networkidle` in `goto` is
        # what made ltm.com take the full 30s timeout and return nothing: analytics and
        # chat widgets poll forever, so the network never goes idle and the navigation
        # fails rather than returning the page that was in fact ready.
        response = page.goto(url, wait_until="domcontentloaded")
        status = response.status if response else None
        try:
            page.wait_for_load_state("networkidle", timeout=settings.BROWSER_IDLE_MS)
        except Exception:
            pass  # never idle — the DOM is still there, and that is what we want

        if status and status >= 400:
            return FetchResult(
                url=url, status="failed", http_status=status, error=f"HTTP {status}"
            )

        # A short settle after the network quiets: frameworks routinely paint their
        # main content one tick after the last request finishes, and grabbing the DOM
        # before that returns the empty shell we came here to avoid.
        page.wait_for_timeout(int(settings.BROWSER_SETTLE_MS))

        # Scroll to the bottom before capturing. Modern marketing pages load most of
        # their content on intersection, so a browser that never scrolls sees only the
        # hero section — measured on ltm.com, rendering without this returned 426
        # characters where a PLAIN HTTP fetch returned 2217. Rendering that yields less
        # than not rendering is worse than useless.
        _scroll_through(page)

        html = page.content()
        landed = page.url or url

        # A render that produced nothing is a failure, not an empty page. Saying so
        # lets the caller fall back to a plain fetch instead of storing a blank.
        if not html or len(html.strip()) < 200:
            return FetchResult(
                url=landed, status="failed", http_status=status,
                error="The browser returned an empty document.",
            )
        return FetchResult(
            url=landed,
            status="ok",
            http_status=status or 200,
            html=html,
            content_type="text/html",
        )
    except Exception as e:
        return FetchResult(url=url, status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
