"""
The company knowledge base: a Qdrant Cloud collection, one point per company FIELD.

Why field-level points rather than one point per company. A company record is seven
unrelated paragraphs — what it does, who it sells to, what it makes, how it feels to
work there, who runs it. Embedding them as one blob averages five topics into a single
vector, so "which companies have a training-heavy culture" competes against product
names and client lists inside the same 768 numbers and loses. One vector per field
keeps each topic's meaning intact, and the company is reassembled at answer time by
grouping hits on `company_id`.

Every point carries the WHOLE row in its payload, not just the field it embeds. That is
a deliberate cost (60 companies x 7 copies is nothing) bought for a specific property:
during a build phase the set of interesting fields keeps changing, and a payload that
already holds everything means changing your mind is a config edit rather than a
re-ingest. People are payload too — a `people` array on the company record, with no
vectors of their own.

The collection's vector size is pinned to the GPU engine's dimension at creation. This
module never accepts a fallback engine; see `embedding_engines.gpu_engine_strict`.
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings
from app.core.logging import logger

# --- What gets embedded -----------------------------------------------------
# The narrative columns, in the order they read best. Everything else in the sheet
# still lands in the payload; this list only decides what earns a vector of its own.
# It is data, not logic, precisely so it can be changed while the schema is in flux.
VECTOR_FIELDS: tuple[str, ...] = (
    "about",
    "clients",
    "services",
    "products",
    "culture",
    "ceo_details",
    "industries",
)

# How each field is spoken about in an answer, so a citation reads like a sentence
# rather than a column header.
FIELD_LABELS: Dict[str, str] = {
    "about": "company overview",
    "clients": "clients",
    "services": "services",
    "products": "products",
    "culture": "culture",
    "ceo_details": "leadership",
    "industries": "industries served",
    "people": "people",
    "person": "person",
}

# The pseudo-field a person's own point is stored under. People are not a column of the
# company sheet, so they are not in VECTOR_FIELDS; they get a vector of their own so a
# question that DESCRIBES someone ("who leads engineering in fintech?") is answerable by
# meaning, not only by typing their name exactly. The full people table still lives in
# every company payload — this adds a searchable copy, it does not move anything.
PERSON_FIELD = "person"

_client_singleton: Optional[Any] = None
_client_lock = threading.Lock()


# --- Connection -------------------------------------------------------------
def configured() -> bool:
    """Whether a Qdrant target exists at all. Checked before every public call."""
    return bool(settings.QDRANT_URL and settings.QDRANT_API_KEY)


def get_client():
    """Cached Qdrant client. Raises if the module was never configured."""
    global _client_singleton
    if not configured():
        raise RuntimeError(
            "The company knowledge base is not configured. Set QDRANT_URL and "
            "QDRANT_API_KEY in backend/.env."
        )
    with _client_lock:
        if _client_singleton is None:
            from qdrant_client import QdrantClient

            _client_singleton = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=int(settings.QDRANT_TIMEOUT_SECONDS),
            )
            logger.info(f"Company store connected: {settings.QDRANT_URL}")
        return _client_singleton


def reset_client() -> None:
    """Drop the cached client so the next call re-reads config. Used by tests."""
    global _client_singleton
    with _client_lock:
        _client_singleton = None


def vector_size() -> int:
    """The dimension the collection is (or will be) built at."""
    return int(settings.EMBEDDING_GPU_DIM)


# --- Collection lifecycle ---------------------------------------------------
def collection_name() -> str:
    return settings.QDRANT_COMPANIES_COLLECTION


def collection_exists() -> bool:
    try:
        return bool(get_client().collection_exists(collection_name()))
    except Exception as e:
        logger.warning(f"Could not check the company collection: {e}")
        return False


def ensure_collection(recreate: bool = False) -> None:
    """
    Create the collection if absent, optionally wiping it first.

    `recreate=True` is the loader's normal path: the sheet is the single source of
    truth, so rebuilding from scratch is what guarantees no rows survive from an older
    column layout. Upsert-in-place would leave fields nobody writes any more sitting in
    payloads, silently answering questions with data the sheet no longer contains.
    """
    from qdrant_client import models

    client = get_client()
    name = collection_name()

    if recreate and client.collection_exists(name):
        logger.info(f"Company store: dropping collection '{name}' for a clean rebuild.")
        client.delete_collection(name)

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size(), distance=models.Distance.COSINE
            ),
        )
        logger.info(f"Company store: created '{name}' at {vector_size()} dims (cosine).")

    # Payload indexes make the filters below cheap and, more importantly, exact. A
    # keyword index on company_id is what lets "everything about TCS" be a lookup
    # rather than a similarity search that might miss its own company.
    for field, schema in (
        ("company_id", "keyword"),
        ("company_name", "keyword"),
        ("field", "keyword"),
        ("industries_list", "keyword"),
        ("person_name", "keyword"),
    ):
        try:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema
            )
        except Exception:
            pass  # already indexed — Qdrant has no create-if-absent for indexes


def drop_collection() -> None:
    """Remove the collection entirely. Only the loader and tests should call this."""
    client = get_client()
    if client.collection_exists(collection_name()):
        client.delete_collection(collection_name())
        logger.info(f"Company store: dropped '{collection_name()}'.")


def count_points() -> int:
    try:
        return int(get_client().count(collection_name(), exact=True).count)
    except Exception as e:
        logger.warning(f"Could not count company points: {e}")
        return 0


def status() -> Dict[str, Any]:
    """A health summary for the UI — never raises, so a dead store still renders."""
    if not configured():
        return {
            "configured": False,
            "reachable": False,
            "collection": collection_name(),
            "points": 0,
            "companies": 0,
            "vector_size": vector_size(),
            "detail": "QDRANT_URL / QDRANT_API_KEY are not set.",
        }
    try:
        exists = collection_exists()
        points = count_points() if exists else 0
        return {
            "configured": True,
            "reachable": True,
            "collection": collection_name(),
            "points": points,
            "companies": len(list_companies()) if exists else 0,
            "vector_size": vector_size(),
            "detail": "" if exists else "Collection not built yet — run the loader.",
        }
    except Exception as e:
        return {
            "configured": True,
            "reachable": False,
            "collection": collection_name(),
            "points": 0,
            "companies": 0,
            "vector_size": vector_size(),
            "detail": str(e),
        }


# --- Writing ----------------------------------------------------------------
def point_id(company_id: str, field: str) -> str:
    """
    A deterministic id per (company, field).

    Deterministic rather than random so re-running the loader over an unchanged sheet
    overwrites the same points instead of duplicating the corpus — which matters even
    with wipe-and-reload, because a run that fails half way must be re-runnable.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"company:{company_id}:{field}"))


def person_point_id(company_id: str, person_name: str) -> str:
    """
    A deterministic id per (company, person).

    Keyed on the person's name rather than their row number so re-ordering the sheet,
    or adding someone above them, does not silently create a duplicate of a person who
    is already stored.
    """
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"person:{company_id}:{person_name.strip().lower()}")
    )


def upsert(points: Sequence[Dict[str, Any]], batch: int = 64) -> int:
    """
    Write points. Each dict is `{id, vector, payload}`.

    Batched because a single request carrying every vector is the one that times out on
    a slow link, and a partial failure there loses the whole run rather than one batch.
    """
    from qdrant_client import models

    if not points:
        return 0
    client = get_client()
    written = 0
    for start in range(0, len(points), batch):
        chunk = points[start:start + batch]
        client.upsert(
            collection_name=collection_name(),
            points=[
                models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in chunk
            ],
            wait=True,
        )
        written += len(chunk)
        logger.info(f"Company store: wrote {written}/{len(points)} points.")
    return written


# --- Reading ----------------------------------------------------------------
def search(
    vector: Sequence[float],
    limit: int = 24,
    company_ids: Optional[Sequence[str]] = None,
    fields: Optional[Sequence[str]] = None,
    min_similarity: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Vector search, optionally narrowed by company or field.

    The filters only ever NARROW a similarity search; they never replace it. This is a
    RAG store, so relevance is decided by meaning and the filter is there to answer
    "…but only for TCS", not to become a keyword lookup wearing a vector's clothes.
    """
    from qdrant_client import models

    conditions = []
    if company_ids:
        conditions.append(
            models.FieldCondition(
                key="company_id", match=models.MatchAny(any=list(company_ids))
            )
        )
    if fields:
        conditions.append(
            models.FieldCondition(key="field", match=models.MatchAny(any=list(fields)))
        )
    query_filter = models.Filter(must=conditions) if conditions else None

    floor = settings.COMPANY_MIN_SIMILARITY if min_similarity is None else min_similarity
    response = get_client().query_points(
        collection_name=collection_name(),
        query=list(vector),
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
        score_threshold=floor,
    )
    return [
        {"id": str(p.id), "score": float(p.score), "payload": dict(p.payload or {})}
        for p in response.points
    ]


def fetch_company(company_id: str) -> List[Dict[str, Any]]:
    """Every stored point for one company, no similarity involved."""
    from qdrant_client import models

    records, _ = get_client().scroll(
        collection_name=collection_name(),
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="company_id", match=models.MatchValue(value=company_id)
                )
            ]
        ),
        limit=len(VECTOR_FIELDS) + 8,
        with_payload=True,
        with_vectors=False,
    )
    return [{"id": str(r.id), "score": 1.0, "payload": dict(r.payload or {})} for r in records]


def scroll_all(with_text: bool = True) -> List[Dict[str, Any]]:
    """
    Every point's payload in one pass.

    The name index needs three things at once — the company list, the people arrays and
    the corpus text that decides which name words are distinctive. Fetching them per
    company was 60 round trips to answer one question; this is one scroll of 420 rows.
    """
    fields = ["company_id", "company_name", "field", "people"]
    if with_text:
        fields.append("text")

    out: List[Dict[str, Any]] = []
    offset = None
    while True:
        records, offset = get_client().scroll(
            collection_name=collection_name(),
            limit=256,
            offset=offset,
            with_payload=fields,
            with_vectors=False,
        )
        out.extend(dict(r.payload or {}) for r in records)
        if offset is None:
            break
    return out


def list_companies() -> List[Dict[str, str]]:
    """
    Every company in the store as `{company_id, company_name}`.

    Read from the `about` points only — one per company — so this is a 60-row scroll
    rather than a 420-row one that then has to be de-duplicated.
    """
    from qdrant_client import models

    out: Dict[str, str] = {}
    offset = None
    while True:
        records, offset = get_client().scroll(
            collection_name=collection_name(),
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="field", match=models.MatchValue(value="about"))]
            ),
            limit=256,
            offset=offset,
            with_payload=["company_id", "company_name"],
            with_vectors=False,
        )
        for r in records:
            payload = r.payload or {}
            cid = str(payload.get("company_id") or "")
            if cid:
                out[cid] = str(payload.get("company_name") or cid)
        if offset is None:
            break
    return [{"company_id": k, "company_name": v} for k, v in sorted(out.items(), key=lambda kv: kv[1])]
