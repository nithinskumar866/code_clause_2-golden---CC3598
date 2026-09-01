"""
Fakes for the three things the pipeline cannot own in a test: Qdrant, the network, and
the embedding model.

The fake store is a real implementation of the small slice of Qdrant this project uses —
payload filters, scroll, delete-by-filter, cosine search — rather than a mock that
records calls. That distinction matters here: the invariant most worth testing is that
deleting a page's chunks by filter actually removes the ones that no longer exist, and
a call-recording mock would happily assert the delete happened while proving nothing
about what survived it.
"""
from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List, Optional, Sequence

import pytest


# --- fake Qdrant ------------------------------------------------------------
class FakeStore:
    """An in-memory stand-in for the parts of `app.store.qdrant` the pipeline uses."""

    def __init__(self) -> None:
        self.points: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # -- filters: a plain description, interpreted by `_matches` --
    @staticmethod
    def match_filter(**equals: Any) -> Optional[Dict[str, Any]]:
        clean = {k: v for k, v in equals.items() if v is not None}
        return {"kind": "match", "equals": clean} if clean else None

    @staticmethod
    def any_filter(field: str, values: Sequence[Any]) -> Optional[Dict[str, Any]]:
        return {"kind": "any", "field": field, "values": list(values)} if values else None

    @staticmethod
    def due_filter(now_ts: float) -> Dict[str, Any]:
        return {"kind": "due", "now": now_ts}

    @staticmethod
    def _matches(payload: Dict[str, Any], flt: Optional[Dict[str, Any]]) -> bool:
        if not flt:
            return True
        if flt["kind"] == "match":
            return all(payload.get(k) == v for k, v in flt["equals"].items())
        if flt["kind"] == "any":
            return payload.get(flt["field"]) in flt["values"]
        if flt["kind"] == "due":
            return payload.get("status") == "live" and float(
                payload.get("next_due_at") or 0.0
            ) <= flt["now"]
        return True

    # -- collection names --
    def COMPANIES(self) -> str:
        return "ci_companies"

    def SOURCES(self) -> str:
        return "ci_sources"

    def CONTENT(self) -> str:
        return "ci_content"

    # -- ids --
    @staticmethod
    def point_id(*parts: str) -> str:
        return "|".join(parts)

    # -- writes --
    def upsert(self, collection: str, points, batch: int = 64) -> int:
        bucket = self.points.setdefault(collection, {})
        for p in points:
            bucket[str(p["id"])] = {"vector": p.get("vector") or [1.0], "payload": dict(p["payload"])}
        return len(points)

    def upsert_meta(self, collection: str, key: str, payload: Dict[str, Any]) -> str:
        pid = self.point_id(collection, key)
        self.upsert(collection, [{"id": pid, "vector": [1.0], "payload": payload}])
        return pid

    def delete_by_filter(self, collection: str, flt) -> None:
        if flt is None:
            raise ValueError("Refusing an unfiltered delete.")
        bucket = self.points.get(collection, {})
        for pid in [k for k, v in bucket.items() if self._matches(v["payload"], flt)]:
            del bucket[pid]

    # -- reads --
    def scroll(self, collection: str, flt=None, limit: int = 1000, with_vectors: bool = False):
        for pid, entry in list(self.points.get(collection, {}).items()):
            if self._matches(entry["payload"], flt):
                yield {"id": pid, "payload": dict(entry["payload"])}

    def count(self, collection: str, flt=None) -> int:
        return sum(1 for _ in self.scroll(collection, flt))

    def search(self, vector, limit: int = 24, flt=None, min_similarity=None) -> List[Dict[str, Any]]:
        floor = 0.0 if min_similarity is None else min_similarity
        hits = []
        for pid, entry in self.points.get(self.CONTENT(), {}).items():
            if not self._matches(entry["payload"], flt):
                continue
            score = _cosine(vector, entry["vector"])
            if score >= floor:
                hits.append({"id": pid, "score": score, "payload": dict(entry["payload"])})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:limit]

    def configured(self) -> bool:
        return True

    def ensure_collections(self) -> None:
        pass


def _cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# --- fake embedder ----------------------------------------------------------
def _bag_of_words_vector(text: str, dim: int = 32) -> List[float]:
    """
    A deterministic hashed bag of words.

    Not a good embedding — it has no semantics at all — but it is stable, needs no
    model download, and gives related texts overlapping dimensions, which is enough to
    test that retrieval wiring, filtering and ordering behave.

    CRC32 rather than the builtin `hash()`: Python salts string hashing per process, so
    `hash()` here made every ranking assertion pass or fail by luck. A retrieval test
    that depends on PYTHONHASHSEED is worse than no test, because a red run tells you
    nothing and a green one tells you less.
    """
    vector = [0.0] * dim
    for token in (text or "").lower().split():
        vector[zlib.crc32(token.encode("utf-8")) % dim] += 1.0
    return vector or [0.0] * dim


# --- fixtures ---------------------------------------------------------------
@pytest.fixture
def store(monkeypatch) -> FakeStore:
    """Install the fake store everywhere the real one is imported."""
    fake = FakeStore()
    for module in (
        "app.store.qdrant",
        "app.sources.registry",
        "app.sources.state",
        "app.ingest.pipeline",
        "app.chat.retrieval",
    ):
        try:
            mod = __import__(module, fromlist=["qdrant"])
        except ImportError:  # pragma: no cover
            continue
        if hasattr(mod, "qdrant"):
            monkeypatch.setattr(mod, "qdrant", fake, raising=False)
    return fake


@pytest.fixture
def embedder(monkeypatch):
    """Replace the real model with the hashed bag of words."""
    monkeypatch.setattr(
        "app.embed.engine.embed_documents", lambda texts: [_bag_of_words_vector(t) for t in texts]
    )
    monkeypatch.setattr(
        "app.embed.engine.embed_query", lambda text: _bag_of_words_vector(text)
    )


@pytest.fixture
def no_robots(monkeypatch):
    """Every URL allowed, no delay — robots.txt itself is tested separately."""
    monkeypatch.setattr("app.crawl.robots.allowed", lambda url: True)
    monkeypatch.setattr("app.crawl.robots.delay_for", lambda url: 0.0)
    monkeypatch.setattr("app.crawl.frontier.robots.allowed", lambda url: True)


class FakeSite:
    """
    A tiny website the fetcher can be pointed at.

    Pages can be edited mid-test, and the site tracks how many times each URL was
    requested — which is how the refresh short-circuits are proved: the assertion is not
    that the code took a branch, but that the work downstream of it never happened.
    """

    def __init__(self, pages: Dict[str, str]) -> None:
        self.pages = dict(pages)
        self.etags: Dict[str, str] = {}
        self.requests: Dict[str, int] = {}

    def edit(self, url: str, html: str) -> None:
        self.pages[url] = html
        self.etags.pop(url, None)

    def install(self, monkeypatch, etags: bool = False) -> None:
        from app.crawl.fetcher import FetchResult

        def fake_fetch(url: str, etag=None, last_modified=None) -> FetchResult:
            self.requests[url] = self.requests.get(url, 0) + 1
            if url not in self.pages:
                return FetchResult(url=url, status="failed", http_status=404, error="HTTP 404")
            current = self.etags.setdefault(url, f"etag-{len(self.pages[url])}-{hash(self.pages[url]) & 0xFFFF}")
            if etags and etag and etag == current:
                return FetchResult(url=url, status="not_modified", http_status=304, etag=etag)
            return FetchResult(
                url=url,
                status="ok",
                http_status=200,
                html=self.pages[url],
                etag=current if etags else None,
                content_type="text/html",
            )

        monkeypatch.setattr("app.crawl.fetcher.fetch", fake_fetch)
        monkeypatch.setattr("app.ingest.pipeline.fetcher.fetch", fake_fetch)


@pytest.fixture
def site_factory(monkeypatch):
    def build(pages: Dict[str, str], etags: bool = False) -> FakeSite:
        site = FakeSite(pages)
        site.install(monkeypatch, etags=etags)
        return site

    return build
