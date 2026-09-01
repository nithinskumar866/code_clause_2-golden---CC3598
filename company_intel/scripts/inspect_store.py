"""
Read what is actually in Qdrant. Nothing is written or changed.

The index is the part of a RAG system you cannot see from the outside — an answer that
looks wrong and an answer that is right about the wrong text are indistinguishable
until you read the passages themselves. This prints them.

    python scripts/inspect_store.py                       # what each collection holds
    python scripts/inspect_store.py --company mphasis-com # its pages, and how they scored
    python scripts/inspect_store.py --company mphasis-com --chunks
    python scripts/inspect_store.py --search "cloud migration"
    python scripts/inspect_store.py --url https://mphasis.com/home.html
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.core.config import settings  # noqa: E402
from app.embed import engine  # noqa: E402
from app.sources import registry, state  # noqa: E402
from app.store import qdrant  # noqa: E402

RULE = "─" * 78


def when(ts) -> str:
    if not ts:
        return "never"
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def overview() -> None:
    print(f"\n{RULE}\nQDRANT  {settings.QDRANT_URL}\n{RULE}")
    client = qdrant.get_client()
    for name in (qdrant.COMPANIES(), qdrant.SOURCES(), qdrant.CONTENT()):
        if not client.collection_exists(name):
            print(f"  {name:<16} (does not exist)")
            continue
        info = client.get_collection(name)
        size = info.config.params.vectors.size
        distance = info.config.params.vectors.distance
        note = "the corpus — the only collection ever searched" if name == qdrant.CONTENT() else (
            "registry — placeholder vector, read by filter only"
            if name == qdrant.COMPANIES()
            else "crawl state — placeholder vector, read by filter only"
        )
        print(f"  {name:<16} {qdrant.count(name):>6} points   {size:>4}d {str(distance):<22} {note}")

    print(f"\n  embeddings: {engine.describe()}   floor {settings.min_similarity}")

    print(f"\n{RULE}\nCOMPANIES\n{RULE}")
    for company in registry.list_all():
        cid = company["company_id"]
        summary = state.summarize(cid)
        chunks = qdrant.count(qdrant.CONTENT(), qdrant.match_filter(company_id=cid))
        print(f"  {company['name']}")
        print(f"    id      {cid}   domain {company['domain']}")
        print(f"    pages   {summary['pages_known']}  {summary['pages_by_type']}")
        print(f"    chunks  {chunks}   crawled {when(summary['last_crawled_at'])}")
        if company.get("linkedin_urls"):
            print(f"    linkedin {company['linkedin_urls']}  (stored as links, never fetched)")
        if summary["failures"]:
            print(f"    failed  {len(summary['failures'])} page(s):")
            for f in summary["failures"][:3]:
                print(f"       {f['url']}\n         {str(f['error'])[:110]}")
    print()


def company_detail(company_id: str, show_chunks: bool) -> None:
    company = registry.get(company_id)
    if not company:
        print(f"No company with id {company_id!r}. Run without --company to list them.")
        return

    print(f"\n{RULE}\n{company['name']}  ({company_id})\n{RULE}")
    for record in sorted(state.list_for_company(company_id), key=lambda r: r["url"]):
        chunks = qdrant.count(
            qdrant.CONTENT(),
            qdrant.match_filter(company_id=company_id, url_hash=record["url_hash"]),
        )
        flag = "" if record["status"] == "live" else f"  [{record['status']}]"
        print(f"\n  [{record['page_type']:<10}] {record['url']}{flag}")
        print(f"     {chunks} chunk(s) · hash {str(record.get('page_hash'))[:12]}… "
              f"· crawled {when(record.get('last_crawled_at'))} "
              f"· due {when(record.get('next_due_at'))}")
        if record.get("last_error"):
            print(f"     error: {str(record['last_error'])[:150]}")

    if not show_chunks:
        print(f"\n  (add --chunks to print the passage text)\n")
        return

    print(f"\n{RULE}\nCHUNKS\n{RULE}")
    rows = [
        p["payload"]
        for p in qdrant.scroll(qdrant.CONTENT(), flt=qdrant.match_filter(company_id=company_id))
    ]
    rows.sort(key=lambda r: (r.get("page_url", ""), r.get("chunk_index", 0)))
    current = None
    for row in rows:
        if row.get("page_url") != current:
            current = row.get("page_url")
            print(f"\n  {current}")
        print(f"\n    #{row.get('chunk_index')}  [{row.get('section') or 'no heading'}]")
        print("      " + (row.get("text") or "").replace("\n", "\n      ")[:700])
    print()


def by_url(url: str) -> None:
    from app.crawl.urls import normalize
    from app.process.hashing import url_hash

    canonical = normalize(url) or url
    key = url_hash(canonical)
    print(f"\n{RULE}\n{canonical}\n  url_hash {key}\n{RULE}")
    rows = [p["payload"] for p in qdrant.scroll(qdrant.CONTENT(), flt=qdrant.match_filter(url_hash=key))]
    if not rows:
        print("  Nothing indexed for that URL. Note it is matched on the CANONICAL form,")
        print("  so tracking parameters and a trailing slash are stripped first.\n")
        return
    for row in sorted(rows, key=lambda r: r.get("chunk_index", 0)):
        print(f"\n  #{row.get('chunk_index')}  [{row.get('section') or 'no heading'}]  "
              f"{row.get('company_name')}")
        print("    " + (row.get("text") or "").replace("\n", "\n    ")[:700])
    print()


def search(query: str, limit: int, company_id) -> None:
    """
    The same search the assistant runs, with the floor removed.

    Showing the rejected hits is the point: when an answer says "nothing matches", this
    is how you see whether the corpus really lacks the answer or the floor is simply set
    too high for this question.
    """
    flt = qdrant.match_filter(company_id=company_id) if company_id else None
    hits = qdrant.search(engine.embed_query(query), limit=limit, flt=flt, min_similarity=0.0)
    floor = settings.min_similarity
    print(f"\n{RULE}\n\"{query}\"   floor {floor} ({engine.model_name()})\n{RULE}")
    for hit in hits:
        p = hit["payload"]
        verdict = "KEPT   " if hit["score"] >= floor else "below  "
        print(f"\n  {verdict} {hit['score']:.3f}  {p.get('company_name')}  [{p.get('page_type')}]")
        print(f"     {p.get('page_url')}")
        print(f"     {(p.get('text') or '')[:260]}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read the Qdrant store. Read-only.")
    parser.add_argument("--company", help="Show one company's pages")
    parser.add_argument("--chunks", action="store_true", help="Print passage text too")
    parser.add_argument("--search", help="Run a similarity search, showing rejected hits")
    parser.add_argument("--url", help="Show the chunks stored for one page URL")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if not qdrant.configured():
        print("QDRANT_URL is not set in .env.")
        return 2

    if args.url:
        by_url(args.url)
    elif args.search:
        search(args.search, args.limit, args.company)
    elif args.company:
        company_detail(args.company, args.chunks)
    else:
        overview()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
