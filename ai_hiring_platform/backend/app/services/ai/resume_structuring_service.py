import re
from typing import List, Dict, Any
from llama_index.core.schema import TextNode
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

def structure_resume_to_nodes(
    text: str, 
    candidate_id: int, 
    resume_id: int, 
    filename: str
) -> List[TextNode]:
    """
    Groups raw text into semantic sections, slices sections into reasonable paragraph chunks,
    and returns a list of configured LlamaIndex TextNodes with rich metadata.
    """
    logger.info(f"Structuring resume text into LlamaIndex Nodes for Resume ID: {resume_id}")
    sections_map = parse_sections(text)
    
    nodes: List[TextNode] = []
    global_chunk_count = 0
    
    for section_name, items in sections_map.items():
        if not items:
            continue
            
        # Group lines by paragraphs (lines are grouped if they were adjacent or block-based)
        # For simplicity, we join contiguous lines and split them on larger logical blocks.
        # Alternatively, we can group items in blocks of 3-5 lines to preserve sentences,
        # or combine paragraphs until they reach 400-800 characters.
        
        # Let's rebuild text from lines, keeping track of the starting page for the block.
        blocks: List[Dict[str, Any]] = []
        current_block_lines = []
        block_start_page = items[0]["page"]
        
        for item in items:
            current_block_lines.append(item["text"])
            # If line ends with sentence punctuation or block is getting large, we split
            if len(current_block_lines) >= 4 or len(" ".join(current_block_lines)) > 600:
                blocks.append({
                    "text": "\n".join(current_block_lines),
                    "page": block_start_page
                })
                current_block_lines = []
                block_start_page = item["page"]
                
        if current_block_lines:
            blocks.append({
                "text": "\n".join(current_block_lines),
                "page": block_start_page
            })
            
        # Create a TextNode for each chunk block
        for idx, block in enumerate(blocks):
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
