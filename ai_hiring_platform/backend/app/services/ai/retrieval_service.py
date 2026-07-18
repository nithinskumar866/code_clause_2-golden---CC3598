import re
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from app.core.logging import logger
from app.core.config import settings
from app.core.constants import TECH_TAXONOMY
from app.services.ai import keyword_retrieval

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

def _build_match(node, cosine: float) -> Dict[str, Any]:
    """Compute the ranking signals + confidence blend for one retrieved chunk."""
    chunk_txt = node.text.strip()
    section = node.metadata.get("section", "Summary")

    # Signal 1: Normalized Semantic similarity (0-100)
    sim_score = min(max(cosine * 100, 0), 100)

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
    tech_score = _technical_specificity(chunk_txt)

    confidence_score = int(
        (sim_score * settings.RETRIEVAL_WEIGHT_SIMILARITY) +
        (sec_score * settings.RETRIEVAL_WEIGHT_SECTION) +
        (density_score * settings.RETRIEVAL_WEIGHT_DENSITY) +
        (tech_score * settings.RETRIEVAL_WEIGHT_TECH_SPECIFICITY)
    )
    confidence_score = min(max(confidence_score, 0), 100)

    return {
        "chunk": chunk_txt,
        "section": section,
        "score": round(float(cosine), 3),
        "confidence": confidence_score,
        "page": node.metadata.get("page", 1),
        "filename": node.metadata.get("filename", ""),
        "chunk_id": node.metadata.get("chunk_id", 0),
        "_node_id": node.node_id,
    }


def _fuse_ranks(matches: List[Dict[str, Any]], bm25_by_id: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion of the dense ranking (by the confidence blend, which is
    cosine-dominated) and the sparse BM25 keyword ranking. RRF score for a chunk is
    ``w_dense/(k+rank_dense) + w_sparse/(k+rank_sparse)``; higher is better. Fusion
    only re-orders chunks that already cleared the cosine floor, so it sharpens which
    evidence surfaces first without ever admitting semantically-irrelevant chunks.
    """
    k = settings.HYBRID_RRF_K
    w_dense = settings.HYBRID_WEIGHT_DENSE
    w_sparse = settings.HYBRID_WEIGHT_SPARSE

    dense_order = sorted(matches, key=lambda m: m["confidence"], reverse=True)
    dense_rank = {m["_node_id"]: i for i, m in enumerate(dense_order)}
    sparse_order = sorted(matches, key=lambda m: bm25_by_id.get(m["_node_id"], 0.0), reverse=True)
    sparse_rank = {m["_node_id"]: i for i, m in enumerate(sparse_order)}

    def rrf(m: Dict[str, Any]) -> float:
        nid = m["_node_id"]
        return w_dense / (k + dense_rank[nid]) + w_sparse / (k + sparse_rank[nid])

    return sorted(matches, key=rrf, reverse=True)


def _corpus_from_index(index: VectorStoreIndex):
    """Return (node_ids, texts) for every chunk stored in the index, for BM25 idf."""
    try:
        nodes = list(index.docstore.docs.values())
        return [n.node_id for n in nodes], [n.get_content() for n in nodes]
    except Exception:
        return [], []


def retrieve_evidence_for_requirements(
    index: VectorStoreIndex,
    requirements: List[str],
    top_k: int = 4,
    min_similarity: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval per requirement: dense (FAISS cosine) candidate generation with a
    similarity floor, fused with sparse BM25 keyword ranking via Reciprocal Rank Fusion.

    Chunks whose raw cosine similarity falls below ``min_similarity`` (defaults to
    ``settings.RETRIEVAL_MIN_SIMILARITY``) are discarded as noise, so a requirement
    with no genuinely-relevant evidence resolves to an empty match list — reported
    downstream as "Missing" rather than a weak "Partial". Fusion only re-ranks chunks
    that pass the floor. When ``HYBRID_RETRIEVAL_ENABLED`` is off, ranking falls back
    to the pure dense confidence order.
    """
    threshold = settings.RETRIEVAL_MIN_SIMILARITY if min_similarity is None else min_similarity
    hybrid = settings.HYBRID_RETRIEVAL_ENABLED
    logger.info(
        f"Initiating requirement-wise hybrid retrieval for {len(requirements)} items "
        f"(top_k={top_k}, min_similarity={threshold}, hybrid={hybrid})..."
    )

    # A wider dense candidate pool than the final top_k gives fusion room to re-rank.
    candidate_k = max(top_k * 4, 12)
    retriever = index.as_retriever(similarity_top_k=candidate_k)

    corpus_ids, corpus_texts = _corpus_from_index(index) if hybrid else ([], [])
    retrieval_results = []

    for req in requirements:
        logger.info(f"Querying store for requirement: '{req}'")
        try:
            nodes = retriever.retrieve(req)

            matches: List[Dict[str, Any]] = []
            seen_chunks = set()
            for node_with_score in nodes:
                node = node_with_score.node
                cosine = round(float(node_with_score.score or 0.0), 3)
                chunk_txt = node.text.strip()
                if chunk_txt in seen_chunks:
                    continue
                seen_chunks.add(chunk_txt)
                # Similarity floor: drop unrelated nearest-neighbours so an absent skill
                # yields no evidence (→ "Missing").
                if cosine < threshold:
                    continue
                matches.append(_build_match(node, cosine))

            # Rank: hybrid RRF (dense ⊕ BM25) when enabled and a corpus is available,
            # else the pure dense confidence order.
            if hybrid and matches and corpus_texts:
                bm25 = keyword_retrieval.bm25_scores(req, corpus_texts)
                bm25_by_id = dict(zip(corpus_ids, bm25))
                ranked = _fuse_ranks(matches, bm25_by_id)
            else:
                ranked = sorted(matches, key=lambda m: m["confidence"], reverse=True)

            # Trim internal fields and cap to the requested top_k.
            final = []
            for m in ranked[:top_k]:
                m.pop("_node_id", None)
                final.append(m)

            retrieval_results.append({"requirement": req, "matches": final})

        except Exception as e:
            logger.error(f"Error retrieving evidence for keyword '{req}': {e}", exc_info=True)
            retrieval_results.append({"requirement": req, "matches": [], "error": str(e)})

    logger.info("Hybrid retrieval pipeline complete.")
    return retrieval_results
