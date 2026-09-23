"""Save / load the v2 Document (clean text + page/section/paragraph offset maps)."""
import hashlib
import json

from .config import settings
from .db import pg
from .ingest_v2 import Document


def _cache_file(doc_id: str):
    return settings.cache_path / f"{doc_id}_document.json"


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def save_document(doc_id: str, doc: Document) -> None:
    _cache_file(doc_id).write_text(json.dumps(doc.to_dict(), ensure_ascii=False),
                                   encoding="utf-8")
    with pg() as cur:
        cur.execute(
            """INSERT INTO doc_text (doc_id, text, text_sha1, structure_source, stats)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (doc_id) DO UPDATE SET text = EXCLUDED.text,
                 text_sha1 = EXCLUDED.text_sha1,
                 structure_source = EXCLUDED.structure_source, stats = EXCLUDED.stats""",
            (doc_id, doc.text, text_sha1(doc.text), doc.structure_source,
             json.dumps(doc.stats)),
        )
        for table in ("doc_pages", "doc_sections", "doc_paragraphs"):
            cur.execute(f"DELETE FROM {table} WHERE doc_id = %s", (doc_id,))
        cur.executemany(
            "INSERT INTO doc_pages (doc_id, page_no, char_start, char_end) VALUES (%s,%s,%s,%s)",
            [(doc_id, p.no, p.start, p.end) for p in doc.pages],
        )
        cur.executemany(
            """INSERT INTO doc_sections (doc_id, section_id, level, title, char_start,
                                         char_end, parent_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            [(doc_id, s.id, s.level, s.title, s.start, s.end, s.parent) for s in doc.sections],
        )
        cur.executemany(
            """INSERT INTO doc_paragraphs (doc_id, para_id, char_start, char_end, page_start,
                                           page_end, section_id, is_heading)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(doc_id, p.id, p.start, p.end, p.page_start, p.page_end, p.section_id,
              p.is_heading) for p in doc.paragraphs],
        )


def load_document(doc_id: str) -> Document:
    f = _cache_file(doc_id)
    if f.exists():
        return Document.from_dict(json.loads(f.read_text(encoding="utf-8")))
    with pg() as cur:
        cur.execute("SELECT text, structure_source, stats FROM doc_text WHERE doc_id = %s",
                    (doc_id,))
        row = cur.fetchone()
        if not row:
            raise KeyError(f"{doc_id} has no v2 text — re-upload the PDF")
        cur.execute("SELECT page_no, char_start, char_end FROM doc_pages "
                    "WHERE doc_id = %s ORDER BY page_no", (doc_id,))
        pages = [{"no": r["page_no"], "start": r["char_start"], "end": r["char_end"]}
                 for r in cur.fetchall()]
        cur.execute("SELECT * FROM doc_sections WHERE doc_id = %s ORDER BY char_start",
                    (doc_id,))
        sections = [{"id": r["section_id"], "level": r["level"], "title": r["title"],
                     "start": r["char_start"], "end": r["char_end"], "parent": r["parent_id"]}
                    for r in cur.fetchall()]
        cur.execute("SELECT * FROM doc_paragraphs WHERE doc_id = %s ORDER BY char_start",
                    (doc_id,))
        paragraphs = [{"id": r["para_id"], "start": r["char_start"], "end": r["char_end"],
                       "page_start": r["page_start"], "page_end": r["page_end"],
                       "section_id": r["section_id"], "is_heading": r["is_heading"]}
                      for r in cur.fetchall()]
    doc = Document.from_dict({"text": row["text"], "pages": pages, "sections": sections,
                              "paragraphs": paragraphs,
                              "structure_source": row["structure_source"],
                              "stats": row["stats"]})
    _cache_file(doc_id).write_text(json.dumps(doc.to_dict(), ensure_ascii=False),
                                   encoding="utf-8")
    return doc


def page_spans(doc_id: str) -> dict[int, tuple[int, int]]:
    """page_no -> (char_start, char_end); lets v1 events be scored on spans."""
    doc = load_document(doc_id)
    return {p.no: (p.start, p.end) for p in doc.pages}
