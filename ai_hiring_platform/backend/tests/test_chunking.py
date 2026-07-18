"""Sentence-aware chunking: boundaries, size caps, page propagation, overlap."""
from app.core.config import settings
from app.services.ai.resume_structuring_service import (
    _split_into_sentences,
    _chunk_sentences,
    structure_resume_to_nodes,
)


def test_split_keeps_bullets_and_splits_sentences():
    # A bullet fragment with no terminal punctuation stays whole.
    assert _split_into_sentences("- Built ETL pipelines on Snowflake") == ["- Built ETL pipelines on Snowflake"]
    # Two real sentences split.
    parts = _split_into_sentences("Led the backend team. Shipped a payments service.")
    assert parts == ["Led the backend team.", "Shipped a payments service."]
    # Abbreviations do not cause a false split.
    assert _split_into_sentences("Worked with e.g. Docker and AWS daily.") == [
        "Worked with e.g. Docker and AWS daily."
    ]


def test_chunks_respect_max_chars_and_page():
    sentences = [(f"Sentence number {i} about backend systems.", (i // 3) + 1) for i in range(12)]
    chunks = _chunk_sentences(sentences)
    assert len(chunks) >= 1
    # No chunk exceeds the hard cap.
    for c in chunks:
        assert len(c["text"]) <= settings.CHUNK_MAX_CHARS
        assert c["page"] >= 1
    # First chunk starts on page 1.
    assert chunks[0]["page"] == 1


def test_overlap_carries_context():
    # Long enough to force multiple chunks; overlap should repeat a boundary sentence.
    sentences = [(f"This is sentence {i} with enough length to matter here.", 1) for i in range(20)]
    chunks = _chunk_sentences(sentences)
    if settings.CHUNK_SENTENCE_OVERLAP > 0 and len(chunks) > 1:
        # Some sentence text appears in two consecutive chunks (the overlap).
        joined = [c["text"] for c in chunks]
        assert any(
            any(frag and frag in joined[i + 1] for frag in joined[i].split(". ")[-1:])
            for i in range(len(joined) - 1)
        )


def test_structure_preserves_metadata_schema():
    text = (
        "--- PAGE 1 ---\n"
        "Jane Doe\n"
        "Skills\n"
        "Python, TensorFlow, Docker, AWS\n"
        "Experience\n"
        "Built NLP systems using PyTorch. Deployed models on AWS with Docker.\n"
    )
    nodes = structure_resume_to_nodes(text, candidate_id=1, resume_id=1, filename="jane.pdf")
    assert len(nodes) > 0
    for n in nodes:
        assert set(["candidate_id", "resume_id", "filename", "section", "chunk_id", "page", "source_type"]).issubset(
            n.metadata.keys()
        )
    # chunk_ids are unique and contiguous from 0.
    ids = sorted(n.metadata["chunk_id"] for n in nodes)
    assert ids == list(range(len(nodes)))
