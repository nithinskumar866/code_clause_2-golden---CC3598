import re
from typing import List
from app.core.constants import TECH_TAXONOMY
from app.core.logging import logger

def extract_requirements(jd_text: str) -> List[str]:
    """
    Extracts key software engineering requirements from a Job Description text
    using a predefined taxonomy and boundary-safe keyword matching.
    """
    logger.info("Extracting requirements from Job Description text...")
    extracted_skills = set()
    
    # Process taxonomy keywords
    for category, skills in TECH_TAXONOMY.items():
        for skill in skills:
            # Safe boundary check supporting special characters like C++, C#, .NET
            # (?<!\w) ensures no letters/numbers directly precede the keyword
            # (?!\w) ensures no letters/numbers directly follow the keyword
            pattern = re.compile(rf"(?<!\w){re.escape(skill)}(?!\w)", re.IGNORECASE)
            
            if pattern.search(jd_text):
                # Save original capitalized skill name from taxonomy
                extracted_skills.add(skill)
                
    # Normalize list (e.g. capitalize nicely)
    # We map them to their standard naming convention matching the taxonomy casing
    result = list(extracted_skills)
    
    # Sort for deterministic outcomes
    result.sort()
    
    logger.info(f"Extracted {len(result)} requirements: {result}")
    return result
