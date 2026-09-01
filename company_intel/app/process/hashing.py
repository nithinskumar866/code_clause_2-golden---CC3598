"""
Content and URL hashing — the mechanism the whole refresh strategy rests on.

The hash is taken over a page's *normalised extracted text*, never over raw HTML.
Raw HTML changes on nearly every request: CSRF tokens, session ids, build hashes in
asset names, a rotating testimonial, a "last updated" timestamp. Hashing it would
report every page as changed on every crawl and re-embed the entire corpus daily.

The hash is also taken per PAGE, never per chunk. Chunk boundaries shift when a page
is edited, so one added sentence near the top changes the offsets of every chunk below
it and their hashes all differ — you pay the bookkeeping cost of chunk-level hashing
and still re-embed the whole page. Page-level is both cheaper and more stable.
"""
from __future__ import annotations

import hashlib
import re

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and case so cosmetic reflow is not read as a content change."""
    return _WHITESPACE.sub(" ", (text or "").strip()).lower()


def content_hash(text: str) -> str:
    """A stable digest of a page's meaningful text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def url_hash(url: str) -> str:
    """
    A short, filesystem- and payload-safe key for a URL.

    Used as the payload field that ties a page's chunks together, so all of them can be
    deleted with one filter before the page is rewritten.
    """
    return hashlib.sha256((url or "").encode("utf-8")).hexdigest()[:32]
