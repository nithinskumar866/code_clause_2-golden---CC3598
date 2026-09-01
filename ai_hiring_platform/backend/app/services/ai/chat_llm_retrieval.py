"""
LLM-assisted retrieval — the third stage on top of dense + sparse.

WHY THIS EXISTS
---------------
The deterministic funnel is good at *recall* and bad at *judgement*. Two places in
`chat_retrieval_service` were doing judgement with string operations:

  `_confirms()`   decides whether a passage EVIDENCES a skill by checking whether the
                  skill's own words appear in it. That rule exists for a real reason —
                  dense retrieval alone answered "communication skills for an HR role"
                  with a Mulesoft architect at 100% — but it is blunt in both
                  directions. "Led the sprint ceremonies and unblocked the team" does
                  evidence Communication without containing the word; "communication"
                  in a Skills list is a claim, not proof.

  ranking        The final order is a weighted sum of coverage, depth, place and
                  years. Nothing in it reads what the passage actually SAYS, so two
                  candidates with identical coverage are indistinguishable even when
                  one wrote a paragraph about the work and the other listed a keyword.

Both are the same class of problem the .NET portal hit with skill transfer: an
algorithm was measured, found incapable, and the judgement moved to a model under
tight constraints. Same pattern here.

WHAT THE MODEL IS AND IS NOT ALLOWED TO DO
------------------------------------------
It only ever judges text that deterministic retrieval already returned. It cannot
introduce a candidate, a passage, a skill or a score of its own; every id it is shown
came out of FAISS, and every id it returns is checked back against the set it was
given. A wrong verdict can reorder or demote a REAL candidate — it can never invent
one, and it can never resurrect one the funnel excluded.

That containment is what lets this be optional. Both stages default OFF; with the
model absent, unreachable, slow or malformed, the deterministic result stands exactly
as it did before. Every failure path returns "no opinion", never an exception.

COST DISCIPLINE
---------------
The .NET portal shipped a sequential per-item LLM loop and it cost ~13 serial round
trips per turn — the single worst latency bug in the project. Three rules here:

  1. BATCH     several items per prompt, so 50 candidates is ~7 calls not 50.
  2. CONCURRENT   batches run in a thread pool, so those 7 calls overlap.
  3. DEADLINE  one wall-clock ceiling for the whole stage. When it expires, whatever
               came back is used and the rest is left deterministic.

Verdicts are cached on (model, skill, passage) — the same resume re-queried for the
same skill costs nothing the second time.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.logging import logger
from app.services.ai import llm_service

# --- Cache -------------------------------------------------------------------
# Keyed by a hash of (model, task, payload). A verdict is a property of the text and
# the model, not of the turn, so it survives across conversations and sessions.
_cache: "OrderedDict[str, Any]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(task: str, model: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join([task, model, *parts]).encode("utf-8", "ignore")).hexdigest()
    return digest[:32]


def _cache_get(key: str) -> Any:
    with _cache_lock:
        if key not in _cache:
            return None
        _cache.move_to_end(key)
        return _cache[key]


def _cache_put(key: str, value: Any) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > max(64, settings.RETRIEVAL_LLM_CACHE_SIZE):
            _cache.popitem(last=False)


def clear_cache() -> None:
    """Drop every cached verdict. Used by tests and after a model change."""
    with _cache_lock:
        _cache.clear()


# --- Model access ------------------------------------------------------------
def _model_id(llm: Any) -> str:
    """A stable name for cache keying — verdicts must not survive a model swap."""
    return str(getattr(llm, "model", None) or getattr(llm, "model_name", None) or type(llm).__name__)


def verify_enabled() -> bool:
    return bool(settings.RETRIEVAL_LLM_VERIFY_ENABLED)


def rerank_enabled() -> bool:
    return bool(settings.RETRIEVAL_LLM_RERANK_ENABLED)


def available() -> bool:
    """Whether either assisted stage is switched on AND a model is reachable."""
    if not (verify_enabled() or rerank_enabled()):
        return False
    return llm_service.get_llm() is not None


def _complete(llm: Any, system: str, prompt: str) -> Optional[str]:
    """One completion, returning None rather than raising on any failure."""
    try:
        try:
            response = llm.complete(prompt, system=system, temperature=0.0)
        except TypeError:
            # Hosted llama-index LLMs take a single prompt string.
            response = llm.complete(f"{system}\n\n{prompt}")
        text = getattr(response, "text", None) or str(response)
        return text.strip() or None
    except Exception as e:
        logger.warning(f"LLM-assisted retrieval call failed: {e}")
        return None


_JSON_BLOCK = re.compile(r"\[.*\]|\{.*\}", re.DOTALL)


def _parse_json(text: str) -> Optional[Any]:
    """
    Read the JSON a small model produced, tolerating the prose it wraps around it.

    Local 7B-class models routinely answer with ```json fences or a sentence of
    preamble. Failing the whole batch over that would make the feature look far less
    reliable than the model actually is.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _run_batches(
    items: Sequence[Any],
    batch_size: int,
    worker: Callable[[List[Any]], Dict[Any, Any]],
    deadline: float,
) -> Dict[Any, Any]:
    """
    Run `worker` over batches of `items` concurrently, stopping at `deadline`.

    Partial results are a valid outcome: anything not judged in time simply keeps its
    deterministic treatment, which is why the caller merges rather than replaces.
    """
    batches = [list(items[i:i + batch_size]) for i in range(0, len(items), batch_size)]
    if not batches:
        return {}

    merged: Dict[Any, Any] = {}
    workers = max(1, min(settings.RETRIEVAL_LLM_CONCURRENCY, len(batches)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, batch): batch for batch in batches}
        for future in as_completed(futures, timeout=None):
            if time.monotonic() > deadline:
                # Do not cancel in flight work — just stop waiting for it. The answer
                # has to go out; a late verdict is worth nothing.
                logger.warning("LLM-assisted retrieval hit its deadline; using partial results.")
                break
            try:
                merged.update(future.result() or {})
            except Exception as e:
                logger.warning(f"LLM-assisted retrieval batch failed: {e}")
    return merged


def _clip(text: str, limit: int = 700) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


# --- Stage A: evidence verification ------------------------------------------
_VERIFY_SYSTEM = """You check whether a passage from a CV is real evidence that the person has a skill.

For each item decide:
  "yes"  the passage shows the person USED or APPLIED the skill, or clearly states it as theirs
  "weak" the skill is only named in a list, or is implied but not demonstrated
  "no"   the passage does not support the skill at all

Judge ONLY the passage given. Do not use outside knowledge about the person.
A passage about a DIFFERENT technology is "no", even if the two are related.

Reply with a JSON array and nothing else:
[{"id": "<id>", "verdict": "yes|weak|no"}]"""


def verify_evidence(pairs: Sequence[Tuple[str, str, str]]) -> Dict[str, str]:
    """
    Judge whether each (id, skill, passage) is real evidence of that skill.

    Returns {id -> "yes"|"weak"|"no"} for the items that were judged in time. Ids that
    are absent were NOT judged and must keep their deterministic treatment — an
    absent verdict is "no opinion", never "no".
    """
    if not pairs or not verify_enabled():
        return {}
    llm = llm_service.get_llm()
    if llm is None:
        return {}

    model = _model_id(llm)
    verdicts: Dict[str, str] = {}
    pending: List[Tuple[str, str, str]] = []

    for item_id, skill, passage in pairs:
        key = _cache_key("verify", model, skill, passage)
        cached = _cache_get(key)
        if cached is not None:
            verdicts[item_id] = cached
        else:
            pending.append((item_id, skill, passage))

    if not pending:
        return verdicts

    def worker(batch: List[Tuple[str, str, str]]) -> Dict[str, str]:
        listing = "\n".join(
            f'- id: {i}\n  skill: {skill}\n  passage: "{_clip(passage)}"'
            for i, skill, passage in batch
        )
        raw = _complete(llm, _VERIFY_SYSTEM, listing)
        parsed = _parse_json(raw or "")
        if not isinstance(parsed, list):
            return {}

        allowed = {i for i, _, _ in batch}
        out: Dict[str, str] = {}
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            item_id = str(entry.get("id", ""))
            verdict = str(entry.get("verdict", "")).strip().lower()
            # An id the model invented is discarded. It can only answer about what it
            # was asked about.
            if item_id in allowed and verdict in {"yes", "weak", "no"}:
                out[item_id] = verdict
        for item_id, skill, passage in batch:
            if item_id in out:
                _cache_put(_cache_key("verify", model, skill, passage), out[item_id])
        return out

    deadline = time.monotonic() + settings.RETRIEVAL_LLM_DEADLINE_SECONDS
    verdicts.update(_run_batches(pending, max(1, settings.RETRIEVAL_LLM_BATCH), worker, deadline))
    logger.info(f"LLM evidence verification: {len(verdicts)}/{len(pairs)} judged (model={model}).")
    return verdicts


# --- Stage B: re-ranking ------------------------------------------------------
_RERANK_SYSTEM = """You rate how well a candidate's CV evidence answers a recruiter's request.

Score 0-100:
  90-100  directly and repeatedly evidences everything asked for
  70-89   evidences the main requirement with real work behind it
  40-69   partial: some requirements evidenced, others absent
  10-39   only loosely related
  0-9     does not address the request

Judge ONLY the evidence shown. Missing evidence means a LOW score, never a guess.
Do not reward a candidate for a skill that is merely plausible for their job title.

Reply with a JSON array and nothing else:
[{"id": <id>, "score": <0-100>}]"""


def rerank(request: str, candidates: Sequence[Dict[str, Any]]) -> Dict[int, float]:
    """
    Score how well each candidate's evidence answers `request`, 0..100.

    `candidates` are the already-scored, already-ordered deterministic results; only
    the top `RETRIEVAL_LLM_RERANK_TOP_K` are judged, because recall is the retriever's
    job and precision is the model's — paying for a verdict on candidate 300 buys
    nothing.

    Returns {resume_id -> relevance}. A resume_id absent from the result was not
    judged; the caller must leave its score untouched rather than assume zero.
    """
    if not candidates or not request.strip() or not rerank_enabled():
        return {}
    llm = llm_service.get_llm()
    if llm is None:
        return {}

    window = list(candidates)[: max(1, settings.RETRIEVAL_LLM_RERANK_TOP_K)]
    model = _model_id(llm)

    def digest(candidate: Dict[str, Any]) -> str:
        """The evidence the model is allowed to see — nothing else about the person."""
        quotes = []
        for chunk in (candidate.get("evidence") or [])[:4]:
            section = chunk.get("section") or "CV"
            quotes.append(f"[{section}] {_clip(chunk.get('text', ''), 400)}")
        return "\n    ".join(quotes) or "(no evidence retrieved)"

    scores: Dict[int, float] = {}
    pending: List[Dict[str, Any]] = []
    for candidate in window:
        key = _cache_key("rerank", model, request, digest(candidate))
        cached = _cache_get(key)
        if cached is not None:
            scores[candidate["resume_id"]] = cached
        else:
            pending.append(candidate)

    if not pending:
        return scores

    def worker(batch: List[Dict[str, Any]]) -> Dict[int, float]:
        listing = "\n".join(
            f"- id: {c['resume_id']}\n  evidence:\n    {digest(c)}" for c in batch
        )
        raw = _complete(llm, _RERANK_SYSTEM, f"Recruiter request: {request}\n\nCandidates:\n{listing}")
        parsed = _parse_json(raw or "")
        if not isinstance(parsed, list):
            return {}

        allowed = {c["resume_id"] for c in batch}
        out: Dict[int, float] = {}
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            try:
                rid = int(entry.get("id"))
                score = float(entry.get("score"))
            except (TypeError, ValueError):
                continue
            # Only ids we asked about, only scores in range. A model that returns a
            # resume_id it was never shown is hallucinating a candidate.
            if rid in allowed:
                out[rid] = max(0.0, min(100.0, score))
        for candidate in batch:
            if candidate["resume_id"] in out:
                _cache_put(_cache_key("rerank", model, request, digest(candidate)),
                           out[candidate["resume_id"]])
        return out

    deadline = time.monotonic() + settings.RETRIEVAL_LLM_DEADLINE_SECONDS
    scores.update(_run_batches(pending, max(1, settings.RETRIEVAL_LLM_BATCH), worker, deadline))
    logger.info(f"LLM re-rank: {len(scores)}/{len(window)} candidates judged (model={model}).")
    return scores
