"""
URL normalisation and scope rules.

Every URL the crawler touches passes through `normalize` first. Without it the same
page enters the frontier several times — with a fragment, with a tracking parameter,
with and without a trailing slash — and each copy is fetched, chunked and embedded
separately, so the corpus fills with duplicates that then compete with each other at
retrieval time.
"""
from __future__ import annotations

from typing import Optional, Set
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

# Parameters that identify a campaign or a referrer rather than a page. Dropping them
# is what collapses a dozen inbound-link variants onto one canonical URL.
_TRACKING_PREFIXES = ("utm_", "mc_", "pk_", "hsa_", "_hs")
_TRACKING_EXACT = {
    "gclid", "fbclid", "msclkid", "igshid", "mkt_tok", "ref", "referrer",
    "source", "cmpid", "campaign", "yclid", "twclid", "s_kwcid",
}

# Locale switchers. Dropped for the same reason as tracking parameters: they address
# ONE page, not many. A live crawl of wipro.com spent eleven of its twelve pages on
# `careers.wipro.com?locale=` in eleven languages — the entire budget on one page,
# stored eleven times, competing with itself at retrieval.
#
# Collapsing them also keeps the corpus monolingual, which matters: an English question
# embedded against the German copy of a page matches poorly, so the translations are
# not merely redundant, they actively dilute the index.
_LOCALE_PARAMS = {
    "locale", "lang", "language", "hl", "lr", "country", "region",
    "currency", "market", "site_lang", "culture",
}

# Extensions this crawler does not read. PDFs are excluded on purpose in v1: they need
# a different extractor, and silently indexing an empty string from one is worse than
# skipping it visibly.
_SKIPPED_SUFFIXES = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar", ".gz",
    ".tar", ".7z", ".exe", ".dmg", ".pkg", ".mp3", ".mp4", ".avi", ".mov", ".wmv",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp", ".tiff",
    ".css", ".js", ".json", ".xml", ".rss", ".atom", ".woff", ".woff2", ".ttf", ".eot",
)

# Two-part public suffixes, so `bbc.co.uk` is not read as the registrable domain
# `co.uk`. This is a short pragmatic list rather than the full Public Suffix List — a
# miss only means the scope check is stricter than necessary, never looser.
_TWO_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.in", "net.in", "org.in", "ac.in",
    "co.jp", "co.kr", "co.nz", "co.za", "com.au", "com.br", "com.cn", "com.mx",
    "com.sg", "com.tr", "com.tw", "com.hk", "com.my", "com.ph", "co.id", "co.il",
}


def normalize(url: str, base: Optional[str] = None) -> Optional[str]:
    """
    Canonicalise one URL, or return None if it is not a fetchable web page.

    Query parameters are kept but sorted, because order is not meaning and sorting
    makes the same page hash to the same key however it was linked.
    """
    if not url:
        return None
    url = url.strip()
    if not url or url.startswith(("mailto:", "tel:", "javascript:", "data:", "#")):
        return None

    if base:
        url = urljoin(base, url)

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return None
    if not parts.netloc:
        return None

    host = parts.netloc.lower()
    # `www.acme.com` and `acme.com` are one site serving one page. Left alone, the apex
    # and the www host each enter the frontier, each get fetched, and each produce a
    # full duplicate of the homepage under a different key — observed on the first real
    # site this crawler was pointed at.
    if host.startswith("www."):
        host = host[4:]
    # Drop the port when it is the scheme's default, so :443 and nothing are one URL.
    if (parts.scheme == "http" and host.endswith(":80")) or (
        parts.scheme == "https" and host.endswith(":443")
    ):
        host = host.rsplit(":", 1)[0]

    path = parts.path or "/"
    while "//" in path:
        path = path.replace("//", "/")
    # Trailing slashes are dropped everywhere, the domain root included: `acme.com/`
    # and `acme.com` are one page, and letting them differ means crawling the homepage
    # twice and storing two copies of it.
    if path.endswith("/"):
        path = path[:-1]

    if path.lower().endswith(_SKIPPED_SUFFIXES):
        return None

    kept = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not _is_tracking(k)
    )
    query = urlencode(kept)

    return urlunsplit((parts.scheme, host, path, query, ""))


def _is_tracking(key: str) -> bool:
    key = key.lower()
    return (
        key in _TRACKING_EXACT
        or key in _LOCALE_PARAMS
        or key.startswith(_TRACKING_PREFIXES)
    )


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower().split(":", 1)[0]


def path_of(url: str) -> str:
    return (urlsplit(url).path or "/").lower()


def registrable_domain(url_or_host: str) -> str:
    """
    The domain that defines crawl scope: `careers.acme.co.uk` -> `acme.co.uk`.

    Subdomains stay in scope on purpose — a company's careers or product site is very
    often on one, and it holds exactly the pages this system exists to read.
    """
    host = url_or_host if "://" not in url_or_host else host_of(url_or_host)
    host = host.lower().strip(".")
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in _TWO_PART_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def same_site(url: str, domain: str) -> bool:
    """Whether a URL belongs to the company being crawled."""
    return registrable_domain(url) == registrable_domain(domain)


def matches_any(url: str, patterns: Set[str] | list) -> bool:
    """Case-insensitive substring match against the URL — the whole pattern language.

    Deliberately not regex: these patterns are typed into a registration payload by a
    person, and a bad regex there is a crash or a catastrophic backtrack rather than a
    rule that simply matches nothing.
    """
    if not patterns:
        return False
    low = url.lower()
    return any(p.lower() in low for p in patterns if p)
