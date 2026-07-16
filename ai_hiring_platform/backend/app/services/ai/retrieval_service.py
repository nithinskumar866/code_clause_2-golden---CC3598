import re
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex
from app.core.logging import logger
from app.core.config import settings

def retrieve_evidence_for_requirements(
    index: VectorStoreIndex, 
    requirements: List[str], 
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Queries the FAISS index for each extracted requirement keyword.
    Ranks matches dynamically using similarity scores, section weights, and detail density signals,
    and returns a ranked, deduplicated, consolidated list of matches per requirement.
    """
    logger.info(f"Initiating requirement-wise semantic retrieval for {len(requirements)} items (top_k={top_k})...")
    
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieval_results = []
    
    # Common tech keyword patterns to identify technical specificity
    tech_patterns = re.compile(
        r"\b(python|javascript|typescript|c\+\+|java|go|rust|ruby|php|sql|nosql|docker|kubernetes|aws|gcp|azure|"
        r"tensorflow|pytorch|keras|scikit-learn|fastapi|django|flask|react|vue|angular|redis|postgres|mysql|"
        r"mongodb|kafka|rabbitmq|git|ci/cd|jenkins|terraform|ansible|nlp|cnn|rnn|bert|gpt|llm|ocr|cv|api)\b",
        re.IGNORECASE
    )
    
    for req in requirements:
        logger.info(f"Querying vector store for requirement: '{req}'")
        try:
            nodes = retriever.retrieve(req)
            
            raw_matches = []
            seen_chunks = set()
            
            for node_with_score in nodes:
                node = node_with_score.node
                score = node_with_score.score or 0.0
                
                # Check for duplicates
                chunk_txt = node.text.strip()
                if chunk_txt in seen_chunks:
                    continue
                seen_chunks.add(chunk_txt)
                
                score_rounded = round(float(score), 3)
                section = node.metadata.get("section", "Summary")
                
                # --- Dynamic Ranking Signals (Module 1 & 3) ---
                
                # Signal 1: Normalized Semantic similarity (0-100)
                sim_score = min(max(score_rounded * 100, 0), 100)
                
                # Signal 2: Normalized Section Importance (0-100)
                sec_score = 0
                if section == "Experience":
                    sec_score = 100
                elif section == "Projects":
                    sec_score = 80
                elif section == "Skills":
                    sec_score = 30
                
                # Signal 3: Normalized Detail density (0-100)
                text_len = len(chunk_txt)
                density_score = 0
                if text_len > 150:
                    density_score = 100
                elif text_len > 70:
                    density_score = 70
                elif text_len > 30:
                    density_score = 30
                    
                # Signal 4: Normalized Technical Specificity (0-100)
                tech_matches = tech_patterns.findall(chunk_txt)
                tech_score = min(len(tech_matches) * 20, 100)
                
                # Calculate combined confidence score (0-100) using configurable weights
                confidence_score = int(
                    (sim_score * settings.RETRIEVAL_WEIGHT_SIMILARITY) +
                    (sec_score * settings.RETRIEVAL_WEIGHT_SECTION) +
                    (density_score * settings.RETRIEVAL_WEIGHT_DENSITY) +
                    (tech_score * settings.RETRIEVAL_WEIGHT_TECH_SPECIFICITY)
                )
                confidence_score = min(max(confidence_score, 0), 100)
                
                raw_matches.append({
                    "chunk": chunk_txt,
                    "section": section,
                    "score": score_rounded,
                    "confidence": confidence_score,
                    "page": node.metadata.get("page", 1),
                    "filename": node.metadata.get("filename", ""),
                    "chunk_id": node.metadata.get("chunk_id", 0)
                })
            
            # Sort matches based on confidence score descending (pushing stronger evidence to the top)
            ranked_matches = sorted(raw_matches, key=lambda x: x["confidence"], reverse=True)
            
            retrieval_results.append({
                "requirement": req,
                "matches": ranked_matches
            })
            
        except Exception as e:
            logger.error(f"Error retrieving evidence for keyword '{req}': {e}", exc_info=True)
            retrieval_results.append({
                "requirement": req,
                "matches": [],
                "error": str(e)
            })
            
    logger.info("Semantic retrieval pipeline complete.")
    return retrieval_results
