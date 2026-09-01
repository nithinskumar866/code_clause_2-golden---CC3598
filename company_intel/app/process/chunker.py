"""
Turning a page's sections into the units that get embedded.

The unit of retrieval is not the page. A company's "Solutions" page can cover six
unrelated offerings, and embedding it whole averages all six into one vector — so a
question about any single one of them matches weakly, while a page that mentions
everything shallowly matches everything. Sections keep each topic's meaning intact.

Three rules do the work:

* **Split at headings first, characters second.** A heading is an author's own
  statement about where one topic ends, and it is better than any window we could pick.
* **Carry the heading path into the text.** The chunk that gets embedded begins with
  "Products > CRM", so the vector encodes what the prose is *about* and not only what
  it says. Retrieval against a short question improves noticeably for it.
* **Merge the runts.** A two-line section under its own heading embeds badly — too
  little signal, and it competes as an equal against a full paragraph. Short sections
  are combined with their neighbours until they carry enough text to mean something.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app.core.config import settings
from app.extract.html_text import ExtractedPage, Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """One embeddable unit of a page."""

    text: str          # what gets embedded — heading path included
    body: str          # the prose alone, for display and citation
    section: str       # the heading path, for the UI and for filtering
    index: int


def _split_long(text: str, target: int, overlap: int) -> List[str]:
    """
    Break one over-long block on paragraph, then sentence, boundaries.

    Overlap exists so a fact stated across a boundary is not lost to both sides. It is
    a fixed character budget rather than a fixed sentence count because sentence
    lengths vary far more than the model's context does.
    """
    pieces: List[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= target:
            pieces.append(paragraph)
            continue
        sentences = _SENTENCE_END.split(paragraph)
        buffer = ""
        for sentence in sentences:
            if buffer and len(buffer) + len(sentence) + 1 > target:
                pieces.append(buffer.strip())
                tail = buffer[-overlap:] if overlap else ""
                buffer = (tail + " " + sentence).strip()
            else:
                buffer = (buffer + " " + sentence).strip()
        if buffer.strip():
            pieces.append(buffer.strip())

    # Re-pack: paragraph splitting can leave several small pieces where one chunk fits.
    packed: List[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > target:
            packed.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        packed.append(current)
    return packed


def chunk_sections(sections: List[Section], reserve_first: int = 0) -> List[Chunk]:
    """
    Sections in, chunks out. Order is preserved so `index` is stable across a page.

    `reserve_first` is headroom the caller intends to prepend to the first chunk (the
    page title). The budget below is enforced on the text that is actually EMBEDDED —
    heading path included — because that is what the model truncates. Budgeting the
    prose alone lets a long heading push the chunk past the model's window, which is
    the same silent-truncation bug in a smaller costume.
    """
    target = settings.CHUNK_TARGET_CHARS
    overlap = settings.CHUNK_OVERLAP_CHARS
    minimum = settings.CHUNK_MIN_CHARS

    # Pass one: merge sections that are too short to stand alone. The merged chunk
    # keeps the first section's heading path, which is the more general of the two.
    merged: List[Section] = []
    for section in sections:
        body = section.text.strip()
        if not body:
            continue
        if merged and len(merged[-1].text) < minimum:
            previous = merged[-1]
            joined_heading = previous.heading_path or section.heading_path
            merged[-1] = Section(
                heading_path=joined_heading,
                text=f"{previous.text}\n\n{body}".strip(),
            )
        else:
            merged.append(Section(heading_path=section.heading_path, text=body))

    # Pass two: split whatever is still too long, budgeting for the prefix each chunk
    # will carry.
    chunks: List[Chunk] = []
    for section in merged:
        prefix_chars = len(section.heading_path) + 1 if section.heading_path else 0
        if not chunks:
            prefix_chars += reserve_first
        effective = max(minimum, target - prefix_chars)
        for body in _split_long(section.text, effective, overlap):
            if len(body.strip()) < 40:  # a stray fragment carries no retrievable meaning
                continue
            prefixed = f"{section.heading_path}\n{body}".strip() if section.heading_path else body
            chunks.append(
                Chunk(
                    text=prefixed,
                    body=body.strip(),
                    section=section.heading_path,
                    index=len(chunks),
                )
            )
    return chunks


def chunk_page(page: ExtractedPage) -> List[Chunk]:
    """
    Chunk a whole extracted page.

    The page title is prepended to the first chunk's embedded text. On a short page the
    title is often the only place the company or product is named at all, and without it
    that chunk is a paragraph of pronouns with nothing to match a question against.
    """
    chunks = chunk_sections(
        page.sections, reserve_first=len(page.title) + 1 if page.title else 0
    )
    if chunks and page.title:
        first = chunks[0]
        chunks[0] = Chunk(
            text=f"{page.title}\n{first.text}".strip(),
            body=first.body,
            section=first.section,
            index=0,
        )
    return chunks
