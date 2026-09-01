"""
Detecting a page that is not the page.

A crawler's most damaging failure is not an error — it is a 200 OK whose body says
"Please enable JavaScript" or "verify that you're not a robot". Nothing raises, the
text extracts cleanly, it chunks, it embeds, and the corpus now holds a bot wall that
the assistant will happily cite as company information.

Observed live: a crawl of techmahindra.com indexed 21 chunks reading "JavaScript is
disabled. In order to continue, we need to verify that you're not a robot." Every one
of them was a real, searchable passage attributed to the company.

The test is deliberately narrow — a short page whose text is dominated by one of these
phrases. A long article that happens to discuss bot detection is not a bot wall, and
throwing it away would be its own kind of data loss.
"""
from __future__ import annotations

import re

_SIGNALS = (
    r"enable\s+javascript",
    r"javascript\s+is\s+(?:disabled|required|not\s+available)",
    r"you\s+need\s+to\s+enable\s+javascript",
    r"verify\s+(?:that\s+)?you(?:'re|\s+are)\s+not\s+a\s+robot",
    r"checking\s+your\s+browser\s+before",
    r"please\s+(?:enable\s+cookies|complete\s+the\s+security\s+check)",
    r"access\s+denied",
    r"request\s+(?:blocked|unsuccessful)",
    r"ddos\s+protection\s+by",
    r"attention\s+required",
    r"are\s+you\s+a\s+(?:human|robot)",
    r"unusual\s+traffic\s+from\s+your",
    r"cloudflare\s+ray\s+id",
    r"this\s+site\s+requires\s+javascript",
)
_PATTERN = re.compile("|".join(_SIGNALS), re.IGNORECASE)

# Above this length, an interstitial is not plausible: real bot walls are short, and a
# long page mentioning one of these phrases is a page ABOUT the subject.
_MAX_INTERSTITIAL_CHARS = 1200


def looks_blocked(text: str) -> bool:
    """Whether this page's text is an interstitial rather than the content we asked for."""
    body = (text or "").strip()
    if not body or len(body) > _MAX_INTERSTITIAL_CHARS:
        return False
    return bool(_PATTERN.search(body))


def reason(text: str) -> str:
    """The phrase that triggered the verdict, for the crawl report."""
    match = _PATTERN.search((text or "")[:_MAX_INTERSTITIAL_CHARS])
    phrase = match.group(0).strip() if match else "an interstitial"
    return (
        f"Served an interstitial, not content (\"{phrase}\"). The site needs JavaScript "
        f"or is challenging automated traffic, so this page cannot be indexed."
    )
