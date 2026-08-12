"""
Bulk-register a folder of resumes into the platform, then index the talent pool.

Mirrors exactly what `POST /api/v1/resume/upload` does per file (DB row first, then
the file saved as `<id>_<filename>`), so bulk-loaded resumes are indistinguishable
from hand-uploaded ones everywhere downstream. Re-running is safe: a resume whose
filename is already registered is skipped.

Usage:
    python -m scripts.bulk_load_resumes <folder> [--no-index]
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.constants import ALLOWED_EXTENSIONS, RESUME_UPLOAD_DIR, STATUS_UPLOADED  # noqa: E402
from app.core.database import SessionLocal, init_db  # noqa: E402
from app.models.database import Resume  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Directory containing .pdf/.docx resumes")
    parser.add_argument("--no-index", action="store_true", help="Register only; skip corpus indexing")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Not a directory: {args.folder}")
        return 1

    init_db()
    db = SessionLocal()
    added = skipped = 0
    try:
        existing = {r.filename for r in db.query(Resume.filename).all()}
        for name in sorted(os.listdir(args.folder)):
            src = os.path.join(args.folder, name)
            if not os.path.isfile(src):
                continue
            if os.path.splitext(name.lower())[1] not in ALLOWED_EXTENSIONS:
                continue
            if name in existing:
                skipped += 1
                continue

            row = Resume(filename=name, status=STATUS_UPLOADED)
            db.add(row)
            db.commit()
            db.refresh(row)
            shutil.copyfile(src, os.path.join(RESUME_UPLOAD_DIR, f"{row.id}_{name}"))
            added += 1

        print(f"Registered {added} resume(s); skipped {skipped} already present.")

        if not args.no_index:
            from app.services.ai import corpus_index_service

            print("Building the unified corpus index (first run downloads the BGE model)...")
            print(corpus_index_service.sync_corpus(db))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
