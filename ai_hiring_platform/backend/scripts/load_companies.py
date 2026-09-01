"""
Load the company Excel sheet into the Qdrant knowledge base.

The sheet is authoritative: by default the collection is dropped and rebuilt, so no row
survives from an older column layout. Pass --keep to upsert in place instead, which is
only correct once the schema has stopped moving.

Usage:
    python -m scripts.load_companies ../../IT_Companies_Database.xlsx
    python -m scripts.load_companies <file.xlsx> --keep
    python -m scripts.load_companies --status
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.company import excel_loader, qdrant_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help="Path to the company .xlsx")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Upsert into the existing collection instead of rebuilding it.",
    )
    parser.add_argument("--status", action="store_true", help="Report store status and exit.")
    args = parser.parse_args()

    if args.status:
        for key, value in qdrant_store.status().items():
            print(f"  {key:<12} {value}")
        return 0

    if not args.file:
        parser.error("a .xlsx path is required unless --status is given")
    if not os.path.isfile(args.file):
        print(f"No such file: {args.file}")
        return 1
    if not qdrant_store.configured():
        print("QDRANT_URL / QDRANT_API_KEY are not set in backend/.env.")
        return 1

    try:
        result = excel_loader.load(args.file, recreate=not args.keep)
    except Exception as e:
        print(f"Load failed: {e}")
        return 1

    print("\nCompany store loaded:")
    for key, value in result.items():
        print(f"  {key:<14} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
