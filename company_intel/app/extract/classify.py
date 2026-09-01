"""
What kind of page is this?

`page_type` earns its place twice. It sets the refresh cadence — a newsroom is re-read
every two days, a leadership page every thirty — and it lets retrieval narrow when a
question is clearly about one facet ("who runs the company" wants `leadership`, not the
pricing page).

Deliberately heuristic, in this order: URL path first, then `<title>`, then `<h1>`. The
URL is the most reliable signal a site gives about its own structure, because it is
chosen by the people who built the information architecture. Titles are marketing copy
and drift; headings are worse. An LLM classifier here would be slower, non-reproducible
and no more accurate than a path that literally reads `/about-us`.

Unmatched pages are `other`, not a guess. `other` refreshes on the slow cadence and is
still fully searchable — nothing is lost by declining to label a page.
"""
from __future__ import annotations

from typing import Optional

# Ordered: the first rule that matches wins, so the more specific patterns come first.
# `leadership` precedes `about` because /about/leadership is a leadership page.
_RULES: tuple = (
    ("leadership", (
        "/leadership", "/our-team", "/ourteam", "/team", "/management", "/executive",
        "/board", "/founders", "/who-we-are", "/people", "/directors",
    )),
    ("news", (
        "/news", "/blog", "/press", "/media", "/insights", "/articles", "/stories",
        "/newsroom", "/announcements", "/events", "/webinar",
    )),
    ("careers", (
        "/careers", "/career", "/jobs", "/join-us", "/joinus", "/life-at",
        "/work-with-us", "/opportunities", "/hiring", "/culture",
    )),
    ("products", (
        "/products", "/product", "/platform", "/features", "/pricing", "/plans",
        "/technology", "/apps", "/tools",
    )),
    ("services", (
        "/services", "/service", "/solutions", "/capabilities", "/offerings",
        "/consulting", "/expertise", "/what-we-do",
    )),
    ("clients", (
        "/clients", "/customers", "/case-stud", "/success-stor", "/testimonial",
        "/partners", "/portfolio", "/our-work", "/projects",
    )),
    ("contact", ("/contact", "/support", "/help", "/get-in-touch", "/locations", "/offices")),
    ("about", ("/about", "/company", "/who-we-are", "/our-story", "/mission", "/overview")),
)

# Title and heading fallbacks. Weaker evidence, so only consulted when the path said
# nothing — and matched as whole phrases to keep "career" out of "careering".
_TEXT_HINTS: tuple = (
    ("leadership", ("leadership team", "our team", "management team", "board of directors",
                    "meet the team", "executive team", "founders")),
    ("careers", ("careers", "join our team", "life at", "open positions", "we are hiring")),
    ("products", ("our products", "product suite", "pricing", "plans and pricing")),
    ("services", ("our services", "what we do", "solutions", "capabilities")),
    ("clients", ("our clients", "case study", "customer stories", "testimonials")),
    ("news", ("newsroom", "press release", "latest news", "blog")),
    ("about", ("about us", "who we are", "our story", "our mission", "company overview")),
    ("contact", ("contact us", "get in touch")),
)

# Subdomain prefixes that name a section as clearly as any path does. Without these
# the root of `careers.wipro.com` classifies as `home`, so a careers site is refreshed
# on the homepage cadence and can never be narrowed to by a careers question.
_HOST_RULES: tuple = (
    ("careers", ("careers", "career", "jobs", "job", "recruit", "hiring", "talent")),
    ("news", ("news", "blog", "press", "media", "insights", "newsroom")),
    ("about", ("about", "company", "corporate", "investors", "ir")),
    ("products", ("products", "product", "store", "shop", "pricing")),
    ("services", ("services", "solutions")),
    ("contact", ("support", "help", "contact")),
)

HOME = "home"
OTHER = "other"


def classify(url: str, title: str = "", h1: str = "", path: Optional[str] = None) -> str:
    """Label one page. Never raises; the worst outcome is `other`."""
    from app.crawl.urls import path_of

    page_path = (path if path is not None else path_of(url)).lower()

    # The host is checked before the path, because a section subdomain describes every
    # page under it — `careers.acme.com/x` is a careers page whatever `/x` is called.
    from app.crawl.urls import host_of

    host = host_of(url)
    first_label = host.split(".")[0] if host else ""
    if first_label and first_label not in ("www", ""):
        for label, prefixes in _HOST_RULES:
            if first_label in prefixes:
                return label

    if page_path in ("", "/"):
        return HOME

    for label, patterns in _RULES:
        if any(pattern in page_path for pattern in patterns):
            return label

    haystack = f" {(title or '').lower()} | {(h1 or '').lower()} "
    for label, phrases in _TEXT_HINTS:
        if any(phrase in haystack for phrase in phrases):
            return label

    return OTHER


def refresh_days(page_type: str) -> int:
    """
    How long this kind of page stays fresh.

    One global cadence is wrong in both directions at once: it re-reads a leadership
    page that changes yearly, and lets a newsroom go stale for a week.
    """
    from app.core.config import settings

    if page_type == "news":
        return settings.REFRESH_DAYS_NEWS
    if page_type in ("products", "services", "careers", "clients", "home"):
        return settings.REFRESH_DAYS_PRODUCTS
    if page_type in ("about", "leadership", "contact"):
        return settings.REFRESH_DAYS_ABOUT
    return settings.REFRESH_DAYS_DEFAULT


def crawl_priority(page_type: str) -> int:
    """
    Frontier ordering. Lower sorts first.

    A crawl is capped at CRAWL_MAX_PAGES, so on a large site the cap decides what the
    system knows. Spending it on `about`, `products` and `leadership` before the
    hundredth blog post is the difference between a useful corpus and an archive of
    press releases.
    """
    return {
        HOME: 0, "about": 1, "products": 2, "services": 2, "leadership": 3,
        "clients": 4, "careers": 5, "contact": 6, OTHER: 7, "news": 8,
    }.get(page_type, 7)
