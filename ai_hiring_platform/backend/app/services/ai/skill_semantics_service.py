"""
Skill Semantics Service.

Provides generic, embedding-driven reasoning primitives used by the Hiring
Decision Agent's deterministic engine:

  * classify_category(requirement)   -> nearest-category label
  * estimate_transfer(missing, have) -> how transferable existing skills are

Both operate as *algorithms*, not lookup tables: any requirement (including
skills never seen before) is reasoned about via semantic similarity to the
category prototypes derived from TECH_TAXONOMY. There is no per-skill branching.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.core.constants import (
    CATEGORY_DISPLAY_NAMES,
    FALLBACK_CATEGORY,
    TECH_TAXONOMY,
)
from app.core.logging import logger
from app.services.ai.embedding_service import get_embedding_model

# Module-level caches (built lazily on first use).
_embedding_cache: Dict[str, np.ndarray] = {}
_category_centroids: Optional[Dict[str, np.ndarray]] = None


def _embed(text: str) -> np.ndarray:
    """Return the L2-normalized embedding vector for a piece of text (cached)."""
    key = text.strip().lower()
    cached = _embedding_cache.get(key)
    if cached is not None:
        return cached
    raw = np.asarray(get_embedding_model().get_text_embedding(key), dtype=float)
    vec = raw / (np.linalg.norm(raw) + 1e-9)
    _embedding_cache[key] = vec
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are already normalized, so the dot product is the cosine similarity.
    return float(np.dot(a, b))


def _get_category_centroids() -> Dict[str, np.ndarray]:
    """
    Build one prototype (centroid) embedding per taxonomy category by averaging
    the embeddings of that category's known skills. Cached after first build.
    Unseen requirements are later classified by nearest centroid, so the model
    generalizes beyond the taxonomy rather than memorizing it.
    """
    global _category_centroids
    if _category_centroids is not None:
        return _category_centroids

    logger.info("Building skill category centroids from TECH_TAXONOMY...")
    centroids: Dict[str, np.ndarray] = {}
    for category, skills in TECH_TAXONOMY.items():
        vectors = [_embed(skill) for skill in skills]
        if not vectors:
            continue
        mean = np.mean(vectors, axis=0)
        centroids[category] = mean / (np.linalg.norm(mean) + 1e-9)

    _category_centroids = centroids
    logger.info(f"Category centroids ready for {len(centroids)} categories.")
    return centroids


def classify_category(requirement: str) -> str:
    """
    Classify any requirement into the nearest category via prototype similarity.
    Returns the human-friendly display label, or FALLBACK_CATEGORY when the
    requirement is not confidently close to any known category.
    """
    if not requirement or not requirement.strip():
        return FALLBACK_CATEGORY

    try:
        query = _embed(requirement)
        centroids = _get_category_centroids()
        best_key, best_sim = None, -1.0
        for key, centroid in centroids.items():
            sim = _cosine(query, centroid)
            if sim > best_sim:
                best_key, best_sim = key, sim

        if best_key is None or best_sim < settings.CATEGORY_MIN_SIMILARITY:
            return FALLBACK_CATEGORY
        return CATEGORY_DISPLAY_NAMES.get(best_key, FALLBACK_CATEGORY)
    except Exception as e:
        logger.error(f"Category classification failed for '{requirement}': {e}", exc_info=True)
        return FALLBACK_CATEGORY


def estimate_transfer(
    missing_requirement: str,
    candidate_skills: List[str],
) -> Tuple[float, str, bool]:
    """
    Estimate how transferable a candidate's demonstrated skills are toward a
    missing requirement.

    Returns (transfer_score, best_related_skill, same_category):
      * transfer_score   - 0..1, higher means the gap is easier to close
      * best_related_skill - the candidate skill most related to the gap ("" if none)
      * same_category    - whether that related skill shares the missing skill's category

    The score is the maximum cosine similarity between the missing skill and any
    demonstrated skill, plus a bonus when they share a category. This works for
    any skill pair, with no hardcoded relationships.
    """
    if not missing_requirement or not candidate_skills:
        return 0.0, "", False

    try:
        gap_vec = _embed(missing_requirement)
        gap_category = classify_category(missing_requirement)

        best_score, best_skill, best_same_cat = 0.0, "", False
        for skill in candidate_skills:
            if not skill:
                continue
            sim = _cosine(gap_vec, _embed(skill))
            same_cat = classify_category(skill) == gap_category
            score = sim + (settings.TRANSFER_SAME_CATEGORY_BOOST if same_cat else 0.0)
            if score > best_score:
                best_score, best_skill, best_same_cat = score, skill, same_cat

        return min(best_score, 1.0), best_skill, best_same_cat
    except Exception as e:
        logger.error(f"Transfer estimation failed for '{missing_requirement}': {e}", exc_info=True)
        return 0.0, "", False
