"""Vector index is content-addressed: a resume_id must never load a different file's index."""
import os
from app.services.ai import vector_store_service as vs


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_fingerprint_differs_by_content(tmp_path):
    a = _write(tmp_path, "a.pdf", b"resume A content")
    b = _write(tmp_path, "b.pdf", b"resume B content")
    assert vs.compute_fingerprint(a) != vs.compute_fingerprint(b)
    assert vs.compute_fingerprint(a) == vs.compute_fingerprint(a)  # stable
    assert vs.compute_fingerprint("nonexistent.pdf") == ""


def test_has_valid_index_requires_matching_fingerprint(tmp_path, monkeypatch):
    # Isolate the vector store dir to a temp location.
    store = tmp_path / "vectors"
    monkeypatch.setattr(vs, "VECTOR_STORE_DIR", str(store))

    rid = 4242
    # No index yet.
    assert vs.has_valid_index(rid, "fp_original") is False

    # Simulate a persisted index (the file has_existing_index checks for) + fingerprint.
    persist = vs.get_persist_dir(rid)
    open(os.path.join(persist, "index_store.json"), "w").close()
    with open(vs._fingerprint_path(rid), "w") as f:
        f.write("fp_original")

    # Existence-only check passes; fingerprint check is content-sensitive.
    assert vs.has_existing_index(rid) is True
    assert vs.has_valid_index(rid) is True                    # no fingerprint requested
    assert vs.has_valid_index(rid, "fp_original") is True     # matches
    assert vs.has_valid_index(rid, "fp_DIFFERENT") is False   # a different resume → rebuild


def test_legacy_index_without_fingerprint_is_rebuilt(tmp_path, monkeypatch):
    store = tmp_path / "vectors"
    monkeypatch.setattr(vs, "VECTOR_STORE_DIR", str(store))
    rid = 4243
    persist = vs.get_persist_dir(rid)
    open(os.path.join(persist, "index_store.json"), "w").close()  # legacy: no fingerprint sidecar
    # A fingerprint is requested but none was recorded → invalid → forces one-time rebuild.
    assert vs.has_valid_index(rid, "some_fp") is False
