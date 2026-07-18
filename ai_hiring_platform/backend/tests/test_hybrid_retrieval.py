"""Hybrid retrieval building blocks: BM25 scoring and Reciprocal Rank Fusion."""
from app.services.ai import keyword_retrieval
from app.services.ai.retrieval_service import _fuse_ranks


def test_bm25_rewards_exact_term_matches():
    docs = [
        "Built ETL pipelines on Snowflake and Airflow.",
        "Collaborated with the team to deliver features on time.",
        "Deep experience with Kubernetes and Docker orchestration.",
    ]
    scores = keyword_retrieval.bm25_scores("Kubernetes", docs)
    assert len(scores) == 3
    # The doc literally containing "Kubernetes" scores highest.
    assert scores[2] == max(scores)
    assert scores[2] > 0
    # Docs without the term score zero.
    assert scores[0] == 0.0 and scores[1] == 0.0


def test_bm25_absent_term_is_all_zero():
    docs = ["Python and TensorFlow work.", "AWS and Docker deployments."]
    assert keyword_retrieval.bm25_scores("golang", docs) == [0.0, 0.0]


def test_bm25_empty_corpus():
    assert keyword_retrieval.bm25_scores("anything", []) == []


def test_fuse_ranks_promotes_keyword_and_semantic_agreement():
    # Two chunks pass the floor; A has higher dense confidence, B has the keyword hit.
    matches = [
        {"_node_id": "A", "confidence": 90, "chunk": "a"},
        {"_node_id": "B", "confidence": 70, "chunk": "b"},
    ]
    # BM25 strongly favours B — fusion should still keep both, ordering by combined rank.
    fused = _fuse_ranks(matches, {"A": 0.1, "B": 5.0})
    assert {m["_node_id"] for m in fused} == {"A", "B"}
    # With dense#1=A and sparse#1=B, the top depends on weights; assert it is deterministic
    # and returns a full ordering without dropping anything.
    assert len(fused) == 2


def test_fuse_ranks_agreement_wins():
    # A is best on BOTH signals → must rank first.
    matches = [
        {"_node_id": "A", "confidence": 95, "chunk": "a"},
        {"_node_id": "B", "confidence": 60, "chunk": "b"},
    ]
    fused = _fuse_ranks(matches, {"A": 9.0, "B": 1.0})
    assert fused[0]["_node_id"] == "A"
