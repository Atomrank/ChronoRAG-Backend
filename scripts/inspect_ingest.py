"""Inspect a stored document's ingestion structure.

  python scripts/inspect_ingest.py DOC_ID
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import docstore  # noqa: E402
from app.db import ensure_schema, pool  # noqa: E402


def main(doc_id: str) -> int:
    pool()
    ensure_schema()
    doc = docstore.load_document(doc_id)

    print(f"doc_id: {doc_id}")
    print(f"structure_source: {doc.structure_source}")
    print(f"stats: {doc.stats}")
    print(f"text_chars: {len(doc.text)}")
    print(f"pages: {len(doc.pages)}  sections: {len(doc.sections)}  paragraphs: {len(doc.paragraphs)}")
    print()

    print("--- first 40 section titles ---")
    for s in doc.sections[:40]:
        print(f"  L{s.level} [{s.id}] {s.title!r}  chars={s.start}-{s.end}")
    if len(doc.sections) > 40:
        print(f"  ... ({len(doc.sections) - 40} more)")
    print()

    body = [p for p in doc.paragraphs if not p.is_heading]
    sample = body if len(body) <= 10 else random.Random(0).sample(body, 10)
    print("--- 10 random paragraphs ---")
    for p in sorted(sample, key=lambda x: x.start):
        snippet = doc.text[p.start:p.end][:200].replace("\n", " ")
        print(f"  [{p.id}] {p.end - p.start} chars  p.{p.page_start}-{p.page_end}: {snippet!r}")
    print()

    long = [p for p in doc.paragraphs if (p.end - p.start) > 5000]
    print(f"--- paragraphs longer than 5,000 chars: {len(long)} ---")
    for p in long[:20]:
        print(f"  [{p.id}] {p.end - p.start} chars")
    print()

    empty_pages = [pg.no for pg in doc.pages if not doc.text[pg.start:pg.end].strip()]
    print(f"--- pages with zero text: {len(empty_pages)} ---")
    if empty_pages:
        print(f"  {empty_pages[:50]}{'...' if len(empty_pages) > 50 else ''}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/inspect_ingest.py DOC_ID", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
