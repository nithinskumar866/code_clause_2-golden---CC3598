import re
from typing import List, Dict, Any, Tuple
from llama_index.core.schema import TextNode
from app.core.config import settings
from app.core.logging import logger

# Section Heading Regex maps
SECTION_HEADERS = {
    "Skills": r"^(skills|technical skills|key skills|core competencies|technologies|skills & tools|expertise|specialties|tools)$",
    "Experience": r"^(experience|professional experience|work experience|employment history|work history|career history|employment|background)$",
    "Projects": r"^(projects|personal projects|selected projects|academic projects|key projects|development projects)$",
    "Education": r"^(education|academic background|academic history|degrees|academic credentials|university)$",
    "Certifications": r"^(certifications|certificates|courses|licenses|training|professional development)$"
}

def parse_sections(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parses raw resume text line-by-line, tracking page boundaries and grouping text
    under logical resume categories (Skills, Experience, Projects, Education, Certifications, Summary).
    """
    lines = text.split("\n")
    
    current_section = "Summary"
    current_page = 1
    
    # Structure: {section_name: [{"text": line, "page": page_num}]}
    sections_map: Dict[str, List[Dict[str, Any]]] = {
        "Summary": [],
        "Skills": [],
        "Experience": [],
        "Projects": [],
        "Education": [],
        "Certifications": []
    }
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # 1. Track page number
        page_match = re.match(r"^---\s*PAGE\s+(\d+)\s*---$", line_stripped, re.IGNORECASE)
        if page_match:
            current_page = int(page_match.group(1))
            continue
            
        # 2. Check for section header transitions
        # We check if the line is relatively short (less than 40 chars) and matches our headings
        matched_new_section = False
        if len(line_stripped) < 40:
            clean_header = re.sub(r"[^\w\s&]", "", line_stripped).lower().strip()
            for sec_name, sec_regex in SECTION_HEADERS.items():
                if re.match(sec_regex, clean_header):
                    current_section = sec_name
                    matched_new_section = True
                    break
                    
        if matched_new_section:
            continue
            
        # 3. Add content to active section
        sections_map[current_section].append({
            "text": line_stripped,
            "page": current_page
        })
        
    return sections_map

# Sentence boundary: end punctuation (. ! ?) then whitespace then a capital/digit.
# (Fixed-width lookbehind of one char — Python re requires that.) A post-pass then
# re-joins fragments that were split right after a known abbreviation.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_ABBREV = {
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "ms.", "mrs.", "dr.", "sr.", "jr.",
    "ph.", "st.", "no.", "inc.", "ltd.", "co.", "u.s.", "u.k.",
}


def _split_into_sentences(line: str) -> List[str]:
    """Split a line into sentences on strong boundaries. A bullet or a fragment with no
    terminal punctuation stays whole — resume bullets are the atomic unit we want to keep.
    Fragments split right after an abbreviation (e.g. "e.g.") are re-joined."""
    line = line.strip()
    if not line:
        return []
    out: List[str] = []
    for frag in _SENTENCE_BOUNDARY.split(line):
        frag = frag.strip()
        if not frag:
            continue
        if out:
            last_word = out[-1].split()[-1].lower() if out[-1].split() else ""
            # Merge when the previous fragment ended on an abbreviation or a single
            # capital initial (e.g. "B." in "B.S."), which is not a real sentence end.
            if last_word in _ABBREV or re.fullmatch(r"[a-z]\.", last_word):
                out[-1] = f"{out[-1]} {frag}"
                continue
        out.append(frag)
    return out


def _chunk_sentences(sentences: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
    """
    Group (sentence, page) units into sentence-boundary-respecting chunks that target
    CHUNK_TARGET_CHARS and never exceed CHUNK_MAX_CHARS, with CHUNK_SENTENCE_OVERLAP
    sentences carried into the next chunk for retrieval-recall context. Each chunk's
    page is the page of its first sentence. Deterministic — no randomness.
    """
    target = settings.CHUNK_TARGET_CHARS
    hard_max = settings.CHUNK_MAX_CHARS
    overlap = max(0, settings.CHUNK_SENTENCE_OVERLAP)

    chunks: List[Dict[str, Any]] = []
    current: List[Tuple[str, int]] = []

    def _flush():
        if not current:
            return
        text = " ".join(s for s, _ in current).strip()
        if text:
            chunks.append({"text": text, "page": current[0][1]})

    cur_len = 0
    for sent, page in sentences:
        add_len = len(sent) + (1 if current else 0)
        # Flush before adding when the current chunk is already substantial and this
        # sentence would push it past the hard max.
        if current and cur_len >= target and cur_len + add_len > hard_max:
            _flush()
            tail = current[-overlap:] if overlap else []
            current = list(tail)
            cur_len = sum(len(s) + 1 for s, _ in current)
        current.append((sent, page))
        cur_len += add_len
        # A single oversized sentence (e.g. a long unpunctuated bullet) stands alone.
        if cur_len >= hard_max:
            _flush()
            current = []
            cur_len = 0

    _flush()
    return chunks


def structure_resume_to_nodes(
    text: str,
    candidate_id: int,
    resume_id: int,
    filename: str
) -> List[TextNode]:
    """
    Structure raw resume text into retrieval-ready TextNodes: section-aware first
    (Skills/Experience/Projects/…), then sentence-aware chunking within each section
    so chunks fall on natural boundaries rather than an arbitrary line/char count.
    Page numbers propagate from the source lines; metadata schema is unchanged.
    """
    logger.info(f"Structuring resume text into LlamaIndex Nodes for Resume ID: {resume_id}")
    sections_map = parse_sections(text)

    nodes: List[TextNode] = []
    global_chunk_count = 0

    for section_name, items in sections_map.items():
        if not items:
            continue

        # Flatten the section's lines into (sentence, page) units, preserving the page
        # each sentence came from (so evidence still cites the right page).
        sentence_units: List[Tuple[str, int]] = []
        for item in items:
            for sent in _split_into_sentences(item["text"]):
                sentence_units.append((sent, item["page"]))

        for block in _chunk_sentences(sentence_units):
            block_text = block["text"].strip()
            if not block_text:
                continue
            node = TextNode(
                text=block_text,
                metadata={
                    "candidate_id": candidate_id,
                    "resume_id": resume_id,
                    "filename": filename,
                    "section": section_name,
                    "chunk_id": global_chunk_count,
                    "page": block["page"],
                    "source_type": "resume"
                }
            )
            nodes.append(node)
            global_chunk_count += 1

    logger.info(f"Generated {len(nodes)} TextNodes for Resume ID: {resume_id}")
    return nodes
