"""
Excel -> company records -> Qdrant points.

The sheet is the source of truth. One row is one company, one column is one field, and
the loader is written to survive the columns changing underneath it: nothing here names
a column literally except to give it a nicer label. A header it has never seen still
lands in the payload, and still earns a vector if it reads like prose.

That last rule is what keeps the module generic. "Does this column deserve to be
embedded" is answered by measuring the column — narrative text is long, categorical
codes are short — rather than by a maintained list of blessed headers. Add a
`Tech Stack` column tomorrow and it will be searchable without a code change; add a
`Founded Year` column and it will be payload-only, correctly, because embedding "1968"
puts a number into a semantic space where it means nothing.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import logger
from app.services.ai import embedding_engines
from app.services.company import qdrant_store

# A column is embedded when its typical value is at least this long. Measured on the
# median, not the mean, so one unusually chatty cell cannot drag a categorical column
# into the vector space.
NARRATIVE_MIN_MEDIAN_CHARS = 40

# Headers that identify the row rather than describe the company.
_IDENTITY_KEYS = {"company_name", "company", "name", "id", "company_id", "sno", "s_no"}

# Sheets that hold people rather than companies, if one is ever added.
_PEOPLE_SHEET_NAMES = {"people", "employees", "employee", "contacts", "staff"}

# Sheets that are documentation, not data.
_IGNORED_SHEET_NAMES = {"notes", "readme", "info", "instructions"}

_HEADER_ALIASES = {
    "company_about": "about",
    "about_company": "about",
    "description": "about",
    "overview": "about",
    "ceo": "ceo_details",
    "ceo_detail": "ceo_details",
    "leadership": "ceo_details",
    "client": "clients",
    "service": "services",
    "product": "products",
    "company_culture": "culture",
    "industry": "industries",
    "industries_present": "industries",
}


# --- Normalisation ----------------------------------------------------------
def normalise_header(header: Any) -> str:
    """'CEO Details' -> 'ceo_details'. Unknown headers keep their normalised form."""
    key = re.sub(r"[^a-z0-9]+", "_", str(header or "").strip().lower()).strip("_")
    return _HEADER_ALIASES.get(key, key)


def slugify(name: str) -> str:
    """'Tata Consultancy Services (TCS)' -> 'tata-consultancy-services-tcs'."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return slug or "unknown"


def split_list(value: str) -> List[str]:
    """Split a prose list ('BFSI, Retail & Consumer, Telecom.') into clean items."""
    if not value:
        return []
    parts = re.split(r"[,;/]|\band\b|&", str(value))
    out: List[str] = []
    for part in parts:
        item = re.sub(r"\s+", " ", part).strip(" .;:-")
        if len(item) >= 2:
            out.append(item)
    return out


def _median(numbers: List[int]) -> float:
    if not numbers:
        return 0.0
    ordered = sorted(numbers)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


# --- Reading the workbook ---------------------------------------------------
def _read_sheet(worksheet) -> Tuple[List[str], List[List[Any]]]:
    rows = worksheet.iter_rows(values_only=True)
    header_row = next(rows, None) or ()
    headers = [normalise_header(h) for h in header_row]
    body = [list(r) for r in rows if any(c is not None and str(c).strip() for c in r)]
    return headers, body


def _people_from_sheet(worksheet) -> Dict[str, List[Dict[str, str]]]:
    """
    Optional people sheet -> `{company_slug: [person, ...]}`.

    People are payload only: they are attached to their company's record and never get
    a vector of their own. A person is therefore found by finding their company first,
    or by the loader-built name index in `company_retrieval`.
    """
    headers, body = _read_sheet(worksheet)
    if not headers:
        return {}
    try:
        company_col = next(
            i for i, h in enumerate(headers) if h in {"company", "company_name", "employer"}
        )
    except StopIteration:
        logger.warning(
            f"People sheet '{worksheet.title}' has no company column; skipping it."
        )
        return {}

    out: Dict[str, List[Dict[str, str]]] = {}
    for row in body:
        company = str(row[company_col] or "").strip()
        if not company:
            continue
        person = {
            headers[i]: str(row[i]).strip()
            for i in range(min(len(headers), len(row)))
            if i != company_col and row[i] is not None and str(row[i]).strip()
        }
        if person:
            out.setdefault(slugify(company), []).append(person)
    return out


def read_workbook(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parse the workbook into company records plus the list of fields worth embedding.

    Returns `(records, vector_fields)`. A record is
    `{company_id, company_name, fields: {key: text}, people: [...], extra…}`.
    """
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)

    people_by_company: Dict[str, List[Dict[str, str]]] = {}
    company_sheet = None
    for worksheet in workbook.worksheets:
        title = re.sub(r"[^a-z]+", "", worksheet.title.lower())
        if title in _PEOPLE_SHEET_NAMES:
            people_by_company.update(_people_from_sheet(worksheet))
        elif title in _IGNORED_SHEET_NAMES:
            continue
        elif company_sheet is None:
            company_sheet = worksheet

    if company_sheet is None:
        raise ValueError(f"No company sheet found in {path}.")

    headers, body = _read_sheet(company_sheet)
    if not headers:
        raise ValueError(f"Sheet '{company_sheet.title}' has no header row.")

    try:
        name_col = next(i for i, h in enumerate(headers) if h in _IDENTITY_KEYS)
    except StopIteration:
        name_col = 0
        logger.warning(
            f"No company-name column recognised in '{company_sheet.title}'; "
            f"using the first column ('{headers[0]}')."
        )

    # Which columns read like prose? Decided from the data, not from their names.
    lengths: Dict[str, List[int]] = {}
    for row in body:
        for i in range(min(len(headers), len(row))):
            if i == name_col or not headers[i]:
                continue
            text = str(row[i] or "").strip()
            if text:
                lengths.setdefault(headers[i], []).append(len(text))

    candidate_keys = list(
        dict.fromkeys(h for i, h in enumerate(headers) if h and i != name_col)
    )
    vector_fields = [
        key
        for key in candidate_keys
        if key in qdrant_store.VECTOR_FIELDS
        or _median(lengths.get(key, [])) >= NARRATIVE_MIN_MEDIAN_CHARS
    ]

    records: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for row in body:
        company_name = str(row[name_col] or "").strip() if name_col < len(row) else ""
        if not company_name:
            continue
        company_id = slugify(company_name)
        # Two rows may legitimately share a name (a subsidiary, a duplicate entry). The
        # slug must still be unique or the second row would silently overwrite the first.
        if company_id in seen:
            seen[company_id] += 1
            company_id = f"{company_id}-{seen[company_id]}"
            logger.warning(f"Duplicate company name '{company_name}'; stored as {company_id}.")
        else:
            seen[company_id] = 1

        fields: Dict[str, str] = {}
        for i in range(min(len(headers), len(row))):
            if i == name_col or not headers[i]:
                continue
            text = str(row[i]).strip() if row[i] is not None else ""
            if text:
                fields[headers[i]] = re.sub(r"\s+", " ", text)

        records.append(
            {
                "company_id": company_id,
                "company_name": company_name,
                "fields": fields,
                "people": people_by_company.get(company_id, []),
            }
        )

    logger.info(
        f"Read {len(records)} companies from '{company_sheet.title}'. "
        f"Embedding fields: {', '.join(vector_fields) or '(none)'}."
    )
    return records, vector_fields


# --- Building points --------------------------------------------------------
def embedding_text(company_name: str, field: str, text: str) -> str:
    """
    What actually gets embedded.

    The company name and the field's plain-English label are prefixed onto the value so
    the vector carries its own context. Without it, a culture paragraph and a services
    paragraph from the same firm sit in the space as anonymous prose, and a question of
    the form "what is the culture like at X" has nothing in the vector to match "at X"
    or "culture" against.
    """
    label = qdrant_store.FIELD_LABELS.get(field, field.replace("_", " "))
    return f"{company_name} — {label}: {text}"


def build_payload(record: Dict[str, Any], field: str, vector_fields: List[str]) -> Dict[str, Any]:
    """
    The payload attached to every point: the whole company, not just this field.

    Carrying the full row on each point means one search hit is enough to answer a
    follow-up about any other field without a second round trip, and means a column
    that is not embedded today is still available to the answer.
    """
    fields = record["fields"]
    payload: Dict[str, Any] = {
        "company_id": record["company_id"],
        "company_name": record["company_name"],
        "field": field,
        "text": fields.get(field, ""),
        "fields": fields,
        "vector_fields": vector_fields,
        "people": record.get("people", []),
    }
    if fields.get("industries"):
        payload["industries_list"] = split_list(fields["industries"])
    return payload


# Whichever of these the sheet used as its name column.
_PERSON_NAME_KEYS = ("name", "person", "person_name", "full_name", "employee", "employee_name")


def person_name(person: Dict[str, Any]) -> str:
    """The person's name, whatever the sheet called that column."""
    for key in _PERSON_NAME_KEYS:
        value = str(person.get(key) or "").strip()
        if value:
            return value
    return ""


def person_text(company_name: str, person: Dict[str, Any]) -> str:
    """
    What a person's vector is built from.

    Every attribute the sheet gave, plus the employer, written as one phrase — so a
    question that describes someone rather than naming them ("who leads engineering at
    a fintech company") has something to match on. Without this a person is reachable
    only by typing their name exactly.
    """
    name = person_name(person)
    details = ", ".join(
        f"{key.replace('_', ' ')}: {value}"
        for key, value in person.items()
        if key not in _PERSON_NAME_KEYS and str(value).strip()
    )
    return f"{name} at {company_name}" + (f" — {details}" if details else "")


def build_person_payload(record: Dict[str, Any], person: Dict[str, Any]) -> Dict[str, Any]:
    """
    A person's point carries the person AND their company's record.

    Same reasoning as the field points: one hit should be enough to answer a follow-up
    about the employer without a second round trip.
    """
    payload: Dict[str, Any] = {
        "company_id": record["company_id"],
        "company_name": record["company_name"],
        "field": qdrant_store.PERSON_FIELD,
        "person_name": person_name(person),
        "person": person,
        "text": person_text(record["company_name"], person),
        "fields": record["fields"],
        "people": record.get("people", []),
    }
    if record["fields"].get("industries"):
        payload["industries_list"] = split_list(record["fields"]["industries"])
    return payload


def build_points(
    records: List[Dict[str, Any]], vector_fields: List[str]
) -> List[Dict[str, Any]]:
    """
    Embed every (company, field) pair and every person, and shape them into points.

    Both go into ONE collection and one embedding pass. People are not a column of the
    company sheet, so they cannot ride on `vector_fields`; they are appended as their
    own pseudo-field, which is what makes a person searchable by description instead of
    only by exact name.
    """
    engine = embedding_engines.gpu_engine_strict()

    pending: List[Tuple[Dict[str, Any], str]] = [
        (record, field)
        for record in records
        for field in vector_fields
        if record["fields"].get(field)
    ]
    texts = [embedding_text(r["company_name"], f, r["fields"][f]) for r, f in pending]

    # People, appended after the fields so one embedding call covers both.
    pending_people: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
        (record, person)
        for record in records
        for person in record.get("people", [])
        if person_name(person)
    ]
    texts += [person_text(r["company_name"], p) for r, p in pending_people]

    if not texts:
        return []

    logger.info(
        f"Embedding {len(pending)} company fields and {len(pending_people)} people "
        f"with '{engine.model}'…"
    )
    vectors = engine.embed_documents(texts)
    if vectors.shape[1] != qdrant_store.vector_size():
        raise RuntimeError(
            f"'{engine.model}' returned {vectors.shape[1]} dims but the collection is "
            f"{qdrant_store.vector_size()}. Fix EMBEDDING_GPU_DIM before loading."
        )

    points = [
        {
            "id": qdrant_store.point_id(record["company_id"], field),
            "vector": vectors[i].tolist(),
            "payload": build_payload(record, field, vector_fields),
        }
        for i, (record, field) in enumerate(pending)
    ]
    points += [
        {
            "id": qdrant_store.person_point_id(record["company_id"], person_name(person)),
            "vector": vectors[len(pending) + i].tolist(),
            "payload": build_person_payload(record, person),
        }
        for i, (record, person) in enumerate(pending_people)
    ]
    return points


def load(path: str, recreate: bool = True) -> Dict[str, Any]:
    """
    Full ingest: read the sheet, rebuild the collection, embed, write.

    `recreate=True` by default because the sheet is authoritative — see
    `qdrant_store.ensure_collection` for why in-place updates are the wrong default
    while the column set is still moving.
    """
    records, vector_fields = read_workbook(path)
    if not records:
        raise ValueError(f"No company rows found in {path}.")

    points = build_points(records, vector_fields)
    qdrant_store.ensure_collection(recreate=recreate)
    written = qdrant_store.upsert(points)

    result = {
        "file": path,
        "companies": len(records),
        "points": written,
        "vector_fields": vector_fields,
        "people": sum(len(r.get("people", [])) for r in records),
        "collection": qdrant_store.collection_name(),
        "vector_size": qdrant_store.vector_size(),
    }
    logger.info(f"Company store loaded: {result}")
    return result
