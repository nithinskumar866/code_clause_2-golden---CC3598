"""
HTML in, clean sectioned text out.

Bad extraction is the quietest way a system like this fails. Nav bars, cookie banners
and footers repeat on every page of a site, so if they survive extraction the corpus
fills with near-identical chunks that then dominate retrieval — every question about
the company returns "Accept all cookies · Privacy Policy · Careers".

Two passes, deliberately:

1. `trafilatura` decides which part of the page is the article. It is good at this and
   built for exactly the boilerplate problem above.
2. `selectolax` walks the same HTML for its heading structure, so each block of text
   keeps the path of headings above it. That path is prefixed to the chunk before
   embedding — "Products > CRM > Pricing" carries meaning that the sentence "Starts at
   $12 per user" does not carry alone.

Links come from the raw HTML rather than the extracted text, because the crawler needs
the navigation that extraction is designed to throw away.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from selectolax.parser import HTMLParser

from app.core.logging import logger

_WHITESPACE = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")

# Elements that are never page content. Removed before the heading walk so a nav menu
# does not contribute headings to the section path.
_STRIP_TAGS = (
    "script", "style", "noscript", "template", "svg", "iframe", "form",
    "nav", "header", "footer", "aside",
)

# Attribute values that mark a container as chrome rather than content.
_CHROME_HINTS = (
    "cookie", "consent", "gdpr", "newsletter", "subscribe", "breadcrumb",
    "sidebar", "site-nav", "navbar", "menu", "modal", "popup", "banner",
    "social-share", "skip-link",
)


@dataclass
class Section:
    """One heading and the text beneath it."""

    heading_path: str
    text: str

    @property
    def combined(self) -> str:
        """What actually gets embedded: the heading path, then the prose."""
        return f"{self.heading_path}\n{self.text}".strip() if self.heading_path else self.text


@dataclass
class ExtractedPage:
    url: str
    title: str = ""
    h1: str = ""
    text: str = ""
    sections: List[Section] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    lang: str = ""

    @property
    def empty(self) -> bool:
        return len(self.text.strip()) < 120


def _clean(text: str) -> str:
    text = _WHITESPACE.sub(" ", (text or "").replace("\r", ""))
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def _is_hidden(node) -> bool:
    """
    Whether the page itself declares this element invisible.

    A stronger signal than any keyword list, because it is the site's own statement
    rather than our guess about its class names. Basecamp's cookie banner is
    `class="tracking tracking--hidden" aria-hidden="true"` — no "cookie" or "consent"
    anywhere in the markup, so hint-matching missed it and "We'd like to use cookies to
    help understand if our ads are working" was indexed as company information.

    Adding "tracking" to the hint list would have fixed that one page and broken others:
    a logistics company writes about shipment tracking in its real content. The
    accessibility attribute generalises where the keyword cannot.
    """
    attributes = getattr(node, "attributes", None) or {}
    if (attributes.get("aria-hidden") or "").lower() == "true":
        return True
    if "hidden" in attributes and (attributes.get("hidden") or "true").lower() != "false":
        return True
    style = (attributes.get("style") or "").lower().replace(" ", "")
    return "display:none" in style or "visibility:hidden" in style


def _is_chrome(node) -> bool:
    """
    Whether a node sits inside site chrome rather than content.

    Ancestors are walked, not just the node itself. Chrome is marked on the container —
    `<div class="cookie-banner"><p>Accept all cookies</p></div>` — and the paragraph
    inside carries no class of its own. Checking only the node let a live crawl of
    basecamp.com put "We'd like to use cookies to help understand if our ads are
    working" into the corpus, where it then answered a question about the company.
    """
    current = node
    for _ in range(10):  # bounded, but deep enough for heavily-wrapped page shells
        if current is None:
            break
        if _is_hidden(current):
            return True
        attributes = getattr(current, "attributes", None) or {}
        for attr in ("class", "id", "role", "aria-label", "data-testid"):
            value = (attributes.get(attr) or "").lower()
            if value and any(hint in value for hint in _CHROME_HINTS):
                return True
        current = current.parent
    return False


def _main_text(html: str, url: str) -> str:
    """trafilatura's read of the page body, with a plain-text fallback."""
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
            no_fallback=False,
        )
        if extracted and len(extracted.strip()) >= 120:
            return _clean(extracted)
    except Exception as e:
        logger.warning(f"trafilatura failed on {url}: {e}")

    # Fallback: strip chrome ourselves and take what is left. Coarser, but a short
    # marketing page with little prose is exactly where trafilatura returns nothing,
    # and those pages are often the ones naming the products.
    tree = HTMLParser(html)
    tree.strip_tags(list(_STRIP_TAGS))
    body = tree.body or tree.root
    return _clean(body.text(separator="\n")) if body else ""


_TEXT_TAGS = ("p", "li", "td", "dd", "blockquote", "figcaption")
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def _has_text_ancestor(node) -> bool:
    """
    Whether a text node sits inside another one we already collected.

    `<li><p>...</p></li>` would otherwise contribute its sentence twice — once for the
    li and once for the p — and a duplicated sentence is a duplicated chunk competing
    with itself at retrieval time.
    """
    parent = node.parent
    while parent is not None:
        if parent.tag in _TEXT_TAGS:
            return True
        parent = parent.parent
    return False


def _sections(html: str) -> List[Section]:
    """
    Split the page at its headings, keeping the h1>h2>h3 path for each block.

    Depth-first over the whole subtree, not just the body's direct children: real pages
    nest their content several wrappers deep, and a shallow walk finds no headings at
    all on most of them.
    """
    tree = HTMLParser(html)
    tree.strip_tags(list(_STRIP_TAGS))
    body = tree.body or tree.root
    if body is None:
        return []

    path: List[str] = ["", "", "", "", "", ""]
    buffer: List[str] = []
    current = ""
    out: List[Section] = []

    def flush() -> None:
        nonlocal buffer
        text = _clean("\n".join(buffer))
        buffer = []
        if len(text) >= 40:
            out.append(Section(heading_path=current, text=text))

    for node in body.traverse(include_text=False):
        tag = node.tag
        if tag in _HEADING_TAGS:
            if _is_chrome(node):
                continue
            heading = _clean(node.text(separator=" "))
            if not heading or len(heading) > 200:
                continue
            flush()
            level = int(tag[1])
            path[level - 1] = heading
            for deeper in range(level, 6):
                path[deeper] = ""
            current = " > ".join(p for p in path if p)
        elif tag in _TEXT_TAGS:
            if _is_chrome(node) or _has_text_ancestor(node):
                continue
            text = _clean(node.text(separator=" "))
            if text:
                buffer.append(text)

    flush()
    return out


def _links(html: str, base_url: str) -> List[str]:
    """Every in-page link, normalised and de-duplicated, order preserved."""
    from app.crawl.urls import normalize

    tree = HTMLParser(html)
    seen: set = set()
    out: List[str] = []
    for node in tree.css("a[href]"):
        href = node.attributes.get("href")
        resolved = normalize(href or "", base=base_url)
        if resolved and resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def extract(html: str, url: str) -> ExtractedPage:
    """The whole extraction step for one page."""
    if not html or not html.strip():
        return ExtractedPage(url=url)

    tree = HTMLParser(html)
    title_node = tree.css_first("title")
    h1_node = tree.css_first("h1")
    html_node = tree.css_first("html")

    page = ExtractedPage(
        url=url,
        title=_clean(title_node.text()) if title_node else "",
        h1=_clean(h1_node.text()) if h1_node else "",
        lang=(html_node.attributes.get("lang") or "").lower()[:5] if html_node else "",
        links=_links(html, url),
    )
    page.text = _main_text(html, url)

    sections = _sections(html)
    # When the heading walk finds nothing usable — a page with no headings at all —
    # the whole body becomes one section so the text is still indexed rather than lost.
    if not sections and page.text:
        sections = [Section(heading_path=page.h1 or page.title, text=page.text)]
    page.sections = sections
    return page
