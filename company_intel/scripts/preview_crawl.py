"""
Crawl a site and print what WOULD be indexed. Nothing is embedded or stored.

This exists because bad extraction is the quietest failure mode in the whole system.
Once text is inside a vector, a corpus full of cookie banners and nav menus looks
exactly like a corpus full of company information — retrieval just gets mysteriously
worse. Reading the extracted chunks first, with your own eyes, is the cheapest quality
gate there is, and it needs neither Qdrant nor an embedding model.

    python scripts/preview_crawl.py acme.com
    python scripts/preview_crawl.py acme.com --max-pages 10 --show-chunks
    python scripts/preview_crawl.py acme.com --deny /blog --deny /tag
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Extracted web text is full of typographic characters that the default Windows console
# codepage cannot encode, and a preview tool that crashes on an em dash is useless.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.crawl import fetcher, urls as urlutil  # noqa: E402
from app.crawl.frontier import Frontier  # noqa: E402
from app.extract import classify, html_text  # noqa: E402
from app.process import chunker  # noqa: E402
from app.process.hashing import content_hash  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview what a crawl would index.")
    parser.add_argument("domain", help="Company domain, e.g. acme.com")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--allow", action="append", default=[])
    parser.add_argument("--deny", action="append", default=[])
    parser.add_argument("--show-chunks", action="store_true", help="Print every chunk in full.")
    args = parser.parse_args()

    domain = urlutil.registrable_domain(args.domain)
    seed = urlutil.normalize(args.domain if "://" in args.domain else f"https://{domain}")
    if not seed:
        print(f"Not a fetchable address: {args.domain}")
        return 2

    frontier = Frontier(
        domain=domain,
        allow_patterns=args.allow,
        deny_patterns=args.deny,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
    )
    frontier.offer(seed, depth=0)

    print(f"\nPreviewing {domain} — up to {args.max_pages} pages, depth {args.max_depth}\n")
    total_chunks = 0
    total_chars = 0

    while True:
        candidate = frontier.next()
        if candidate is None:
            break

        result = fetcher.fetch(candidate.url)
        if not result.ok:
            print(f"  [{candidate.page_type:<10}] {candidate.url}\n      FAILED — {result.error}\n")
            continue

        frontier.mark_seen(result.url)
        page = html_text.extract(result.html, result.url)
        page_type = classify.classify(result.url, title=page.title, h1=page.h1)
        chunks = chunker.chunk_page(page)
        total_chunks += len(chunks)
        total_chars += len(page.text)

        print(f"  [{page_type:<10}] {result.url}")
        print(f"      title    : {page.title[:90]}")
        print(f"      text     : {len(page.text)} chars · {len(page.sections)} sections "
              f"· {len(chunks)} chunks · refresh every {classify.refresh_days(page_type)}d")
        print(f"      hash     : {content_hash(page.text)[:16]}…")

        if chunks:
            if args.show_chunks:
                for chunk in chunks:
                    print(f"\n      --- chunk {chunk.index} [{chunk.section or 'no heading'}] ---")
                    print("      " + chunk.body.replace("\n", "\n      ")[:1200])
            else:
                preview = chunks[0].body.replace("\n", " ")[:180]
                print(f"      first    : {preview}…")
        else:
            print("      (nothing survived extraction — inspect this page by hand)")
        print()

        if candidate.depth < frontier.max_depth:
            frontier.offer_all(page.links, depth=candidate.depth + 1, base=result.url)

    print(f"\n{frontier.emitted} page(s) · {total_chunks} chunk(s) · {total_chars} chars extracted")
    if frontier.budget_exhausted:
        print(f"Page cap reached with {frontier.queued} URL(s) still queued — coverage is partial.")
    if frontier.skipped:
        print(f"\nSkipped {len(frontier.skipped)} URL(s). First few:")
        for url, reason in list(frontier.skipped.items())[:8]:
            print(f"  {reason:<32} {url}")

    fetcher.close_client()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
