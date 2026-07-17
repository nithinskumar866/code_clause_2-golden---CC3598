import re
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from app.core.logging import logger
from app.core.config import settings
from app.core.constants import TECH_TAXONOMY

# Single source of known technologies (no parallel hardcoded keyword list).
_KNOWN_TECH = {s.lower() for skills in TECH_TAXONOMY.values() for s in skills}
# A token is "technical" if it is CamelCase (TensorFlow), an acronym (AWS/SQL),
# carries tech punctuation (C++, CI/CD, node.js), or embeds a digit (S3, k8s).
_MORPH_TECH = re.compile(r"[A-Za-z]+[A-Z0-9]|[A-Z]{2,}|[A-Za-z]*[0-9][A-Za-z0-9]*|[A-Za-z][A-Za-z0-9]*[+#/.][A-Za-z0-9+#/.]*")


def _technical_specificity(text: str) -> int:
    """Generic 0-100 signal for how technically specific a chunk is — rewards concrete
    technology mentions without hardcoding an example list. Counts distinct known-tech
    terms plus morphologically-technical tokens (CamelCase / acronyms / symbols / digits)."""
    lowered = text.lower()
    hits = {t for t in _KNOWN_TECH if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", lowered)}
    for m in _MORPH_TECH.finditer(text):
        tok = m.group(0).lower().strip("+#/.")
        if len(tok) >= 2:
            hits.add(tok)
    return min(len(hits) * 20, 100)

def retrieve_evidence_for_requirements(
    index: VectorStoreIndex,
    requirements: List[str],
    top_k: int = 4,
    min_similarity: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Queries the FAISS index for each extracted requirement keyword.
    Ranks matches dynamically using similarity scores, section weights, and detail density signals,
    and returns a ranked, deduplicated, consolidated list of matches per requirement.

    Chunks whose raw cosine similarity falls below ``min_similarity`` (defaults to
    ``settings.RETRIEVAL_MIN_SIMILARITY``) are discarded as noise, so a requirement
    with no genuinely-relevant evidence resolves to an empty match list — and is
    reported downstream as "Missing" rather than a weak, misleading "Partial".
    """
    threshold = settings.RETRIEVAL_MIN_SIMILARITY if min_similarity is None else min_similarity
    logger.info(
        f"Initiating requirement-wise semantic retrieval for {len(requirements)} items "
        f"(top_k={top_k}, min_similarity={threshold})..."
    )

    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieval_results = []

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

                # Drop weak matches below the similarity floor so that a genuinely
                # absent skill produces no evidence (→ "Missing") instead of being
                # backed by an unrelated nearest-neighbour chunk.
                if score_rounded < threshold:
                    logger.debug(
                        f"Dropping match for '{req}' (score={score_rounded} < {threshold})"
                    )
                    continue

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
                    
                # Signal 4: Normalized Technical Specificity (0-100), computed
                # generically from token morphology + the known-tech taxonomy.
                tech_score = _technical_specificity(chunk_txt)
                
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
