"""
The frontier: which URLs to visit, in which order, and where to stop.

A crawl is capped at CRAWL_MAX_PAGES, so on any site larger than the cap the frontier's
ordering *is* what the system ends up knowing. Breadth-first by discovery order spends
that budget on whatever the homepage happens to link first — usually a blog roll. This
frontier orders by page type instead: the About, Products and Leadership pages are
visited before the two hundredth press release, so a truncated crawl is still a useful
one.

Scope is decided here and nowhere else: same registrable domain, inside the depth
limit, past the deny patterns, and — when allow patterns are given — matching one of
them. Every rejection is recorded rather than dropped, so "why was that page never
indexed?" has an answer.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from app.core.config import settings
from app.crawl import robots
from app.crawl import urls as urlutil
from app.extract import classify


@dataclass
class Candidate:
    url: str
    depth: int
    page_type: str


@dataclass
class Frontier:
    """A bounded, priority-ordered queue of URLs for one company."""

    domain: str
    allow_patterns: List[str] = field(default_factory=list)
    deny_patterns: List[str] = field(default_factory=list)
    max_pages: int = 0
    max_depth: int = 0

    _heap: List[Tuple[int, int, int, str]] = field(default_factory=list, repr=False)
    _seen: Set[str] = field(default_factory=set, repr=False)
    _depths: Dict[str, int] = field(default_factory=dict, repr=False)
    _counter: int = 0
    _emitted: int = 0
    # Domains the site itself redirected us onto. Populated at crawl time, never from
    # config: the authority for "this is also our site" is the redirect, not a guess.
    _adopted: Set[str] = field(default_factory=set, repr=False)
    skipped: Dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.max_pages = self.max_pages or settings.CRAWL_MAX_PAGES
        self.max_depth = self.max_depth or settings.CRAWL_MAX_DEPTH

    # --- admission ---------------------------------------------------------
    def _reject(self, url: str, reason: str) -> None:
        # First reason wins: a URL is usually rejected for the same cause every time it
        # is linked, and the first is the one that explains the decision.
        self.skipped.setdefault(url, reason)

    def offer(self, raw_url: str, depth: int, base: Optional[str] = None) -> bool:
        """Consider one discovered URL. Returns whether it entered the queue."""
        url = urlutil.normalize(raw_url, base=base)
        if not url:
            return False
        if url in self._seen:
            return False

        if depth > self.max_depth:
            self._reject(url, f"Deeper than max_depth={self.max_depth}")
            return False
        if not self._in_scope(url):
            self._reject(url, "Off-site")
            return False
        if urlutil.matches_any(url, self.deny_patterns):
            self._reject(url, "Matched a deny pattern")
            return False
        if self.allow_patterns and not urlutil.matches_any(url, self.allow_patterns):
            self._reject(url, "Outside the allow patterns")
            return False
        if not robots.allowed(url):
            self._reject(url, "Disallowed by robots.txt")
            return False

        self._seen.add(url)
        self._depths[url] = depth
        page_type = classify.classify(url)
        self._counter += 1
        # Depth is the second key so that, among equally interesting page types, the
        # pages closer to the homepage — the canonical ones — are read first.
        heapq.heappush(
            self._heap, (classify.crawl_priority(page_type), depth, self._counter, url)
        )
        return True

    def _in_scope(self, url: str) -> bool:
        if urlutil.same_site(url, self.domain):
            return True
        return urlutil.registrable_domain(url) in self._adopted

    def adopt_domain_of(self, url: str) -> bool:
        """
        Treat the domain a redirect landed on as part of this company's site.

        Only ever called with a URL the crawler actually followed a redirect to, so the
        scope is widened by the site's own declaration rather than by inference.
        Returns whether this was new, so the caller can log it once.
        """
        domain = urlutil.registrable_domain(url)
        if not domain or domain == urlutil.registrable_domain(self.domain):
            return False
        if domain in self._adopted:
            return False
        self._adopted.add(domain)
        return True

    @property
    def adopted_domains(self) -> Set[str]:
        return set(self._adopted)

    def offer_all(self, raw_urls, depth: int, base: Optional[str] = None) -> int:
        return sum(1 for u in raw_urls if self.offer(u, depth, base=base))

    # --- consumption -------------------------------------------------------
    def next(self) -> Optional[Candidate]:
        """The next URL to fetch, or None when the queue is empty or the cap is hit."""
        if self._emitted >= self.max_pages:
            return None
        if not self._heap:
            return None
        _, depth, _, url = heapq.heappop(self._heap)
        self._emitted += 1
        return Candidate(url=url, depth=depth, page_type=classify.classify(url))

    def mark_seen(self, url: str) -> None:
        """
        Record a URL as already visited without queueing it.

        Called with the URL a fetch actually landed on, which after a redirect is not
        the one that was requested. Without this the redirect target is discovered as a
        fresh link later and fetched a second time.
        """
        normalized = urlutil.normalize(url)
        if normalized:
            self._seen.add(normalized)

    @property
    def emitted(self) -> int:
        return self._emitted

    @property
    def queued(self) -> int:
        return len(self._heap)

    @property
    def budget_exhausted(self) -> bool:
        """True when the page cap stopped the crawl before the queue ran dry.

        Reported in the crawl result rather than logged and forgotten: a crawl that hit
        its cap has silently decided what the corpus does not contain, and that is
        exactly the kind of truncation that reads as complete coverage later.
        """
        return self._emitted >= self.max_pages and bool(self._heap)
