"""
Answering: retrieval first, phrasing second.

The LLM is optional here, and that is a design position rather than a convenience. With
no key configured the service returns an extractive answer — the retrieved passages
themselves, attributed and ordered — which is less pleasant to read and exactly as
factually sound. Anything that only works with an LLM key is a feature that cannot be
tested deterministically and cannot be trusted to be grounded.

The floor is enforced before either path runs: when nothing clears the similarity
threshold, the service says it does not know. That sentence is the most important one
it can produce, because a plausible answer assembled from weak evidence is worse than no
answer at all — the reader cannot tell the difference, and the citation looks real.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.chat import guardrails, retrieval


def _extractive(
    result: retrieval.RetrievalResult, citations: List[dict], used: List
) -> str:
    """
    The no-LLM answer: what the sources say, quoted and attributed.

    Grouped by company so a pool-wide question reads as a comparison rather than a list
    of disconnected paragraphs. Quotes only the passages that were actually used, so
    this path shows exactly what the LLM path would have been given.
    """
    lines: List[str] = []
    by_company: Dict[str, List] = {}
    for item in used:
        by_company.setdefault(item.company_name or item.company_id, []).append(item)

    for company_name, items in by_company.items():
        lines.append(f"**{company_name}**")
        for item in items[:3]:
            snippet = item.text.strip()
            # Shorter than the card's own clamp: this text IS the answer here, and a
            # 400-character quote per passage is what made the no-LLM path a wall.
            if len(snippet) > 260:
                snippet = snippet[:260].rsplit(" ", 1)[0] + "…"
            number = next(
                (c["n"] for c in citations if c["page_url"] == item.page_url), None
            )
            marker = f" [{number}]" if number else ""
            lines.append(f"- {snippet}{marker}")
        lines.append("")

    lines.append("_This answer quotes the sources directly._")
    return "\n".join(lines).strip()


def _llm_openai(prompt: str) -> Optional[str]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=settings.ANSWER_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": guardrails.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip() or None


def _llm_ollama(prompt: str) -> Optional[str]:
    """
    A local or tunnelled Ollama server (`/api/chat`).

    `stream: false` because this call sits behind our own SSE stream: the stages the UI
    shows are pipeline steps, not tokens, and re-streaming tokens through a second layer
    would buy nothing but a second place for the connection to break.

    Temperature is near zero. The model's job here is to phrase retrieved passages, and
    every degree of creativity is a degree of drift away from what the sources say.
    """
    import httpx

    url = settings.ANSWER_OLLAMA_URL.rstrip("/") + "/api/chat"
    response = httpx.post(
        url,
        timeout=httpx.Timeout(settings.ANSWER_TIMEOUT_SECONDS),
        json={
            "model": settings.ANSWER_MODEL,
            "stream": False,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": guardrails.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
    )
    response.raise_for_status()
    return ((response.json().get("message") or {}).get("content") or "").strip() or None


def _llm(question: str, context: str) -> Optional[str]:
    """
    Ask the configured model to phrase an answer over the fenced sources.

    Returns None on any failure, and that is deliberate: the caller then falls back to
    quoting the sources. An unreachable model costs readability, never accuracy — the
    facts were fixed by retrieval before this function was called.
    """
    prompt = "\n\n".join(["Sources:", context, f"Question: {question}"])
    provider = (settings.ANSWER_PROVIDER or "none").strip().lower()
    try:
        if provider == "ollama":
            return _llm_ollama(prompt)
        if provider == "openai":
            return _llm_openai(prompt)
        return None
    except Exception as e:
        logger.warning(f"LLM answer failed ({provider}), quoting sources instead: {e}")
        return None


def ask(
    question: str,
    company_id: Optional[str] = None,
    limit: int = 8,
    use_llm: bool = True,
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """
    Answer one question. Always returns a payload; never raises for a bad question.

    `on_progress(stage, detail)` is called as each step actually begins, so a UI can
    show what is happening rather than a spinner that means nothing. It is called from
    whichever thread runs the answer; the SSE route hands it a queue.
    """
    started = time.time()

    def progress(stage: str, detail: str) -> None:
        if on_progress is not None:
            try:
                on_progress(stage, detail)
            except Exception:  # a broken listener must never fail the answer
                pass

    progress("guardrails", "Checking the question")
    ok, reason = guardrails.check_question(question)
    if not ok:
        return {
            "answer": reason,
            "refused": True,
            "scope": "none",
            "company": None,
            "companies": [],
            "citations": [],
            "evidence_count": 0,
            "llm_used": False,
            "duration_seconds": round(time.time() - started, 3),
        }

    progress("retrieval", "Searching the indexed pages")
    result = retrieval.retrieve(question, company_id=company_id, limit=limit)

    if result.empty:
        known = "that company" if result.company else "the tracked companies"
        return {
            "answer": (
                f"Nothing in the indexed pages for {known} answers that. The corpus only "
                f"contains what has been crawled from the registered websites — if the "
                f"page exists but has not been indexed, run a crawl for that company."
            ),
            "refused": False,
            "scope": result.scope,
            "company": result.company,
            "companies": [],
            "citations": [],
            "evidence_count": 0,
            "llm_used": False,
            "duration_seconds": round(time.time() - started, 3),
        }

    progress(
        "grounding",
        f"Grounding {len(result.evidence)} passage(s) from "
        f"{len({e.company_id for e in result.evidence})} company(ies)",
    )
    # The passages that actually enter the prompt. Everything the caller is shown is
    # built from THIS list and nothing wider: retrieval routinely returns more than the
    # model is given, and presenting the surplus as the reasoning behind an answer is a
    # quiet lie in a system whose whole claim is that answers are auditable.
    used = result.evidence[: settings.ANSWER_MAX_CONTEXT_CHUNKS]
    context, citations, tampered = guardrails.build_context(
        used, max_chunks=settings.ANSWER_MAX_CONTEXT_CHUNKS
    )

    text = None
    if use_llm and settings.llm_configured:
        progress("answering", "Writing the answer")
        text = _llm(question, context)
    else:
        progress("answering", "Assembling the answer from the sources")
    if text is None:
        text = _extractive(result, citations, used)
        llm_used = False
    else:
        llm_used = True

    if tampered:
        text += (
            "\n\n_Note: instruction-like text was found in one of the source pages and "
            "was neutralised before this answer was written._"
        )

    return {
        "answer": text,
        "refused": False,
        "scope": result.scope,
        "company": result.company,
        "companies": retrieval.group_by_company(used),
        "citations": citations,
        # The number of passages the answer was WRITTEN from, not the number retrieved.
        # The UI reports this to the reader, so it has to mean what they assume it does.
        "evidence_count": len(used),
        "retrieved_count": len(result.evidence),
        "intent": result.intent,
        "llm_used": llm_used,
        "duration_seconds": round(time.time() - started, 3),
    }
