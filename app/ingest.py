"""
Ingest helpers: PDF page text + sliding windows.

Extract pipelines (kaalkram_v1 pass1 and kaalkram_v2) share paragraph-aligned
windows from ``ingest_v2.windows`` — see ``extract_windows_for_doc``.
"""
import hashlib
import re
from pathlib import Path

import fitz  # pymupdf

from .config import settings


def extract_pages(pdf_path: Path) -> list[str]:
    """Return page text, index 0 = page 1."""
    doc = fitz.open(pdf_path)
    try:
        return [page.get_text("text") or "" for page in doc]
    finally:
        doc.close()


def doc_id_for(pdf_path: Path) -> str:
    h = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:16]
    return f"doc_{h}"


# ------------------------------------------------------------
# PART A chunking: character windows, page-boundary agnostic.
# This is the honest naive baseline — it does NOT know about pages.
# ------------------------------------------------------------
def naive_chunks(pages: list[str]) -> list[dict]:
    size = settings.naive_chunk_chars
    overlap = settings.naive_chunk_overlap

    # Build one long string, remembering where each page starts
    offsets: list[tuple[int, int]] = []   # (char_offset, page_no)
    buf: list[str] = []
    cursor = 0
    for i, text in enumerate(pages, start=1):
        offsets.append((cursor, i))
        buf.append(text)
        cursor += len(text) + 1
    full = "\n".join(buf)

    def page_at(pos: int) -> int:
        lo, hi, ans = 0, len(offsets) - 1, 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if offsets[mid][0] <= pos:
                ans = offsets[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return ans

    chunks: list[dict] = []
    start = 0
    idx = 0
    step = max(1, size - overlap)
    while start < len(full):
        end = min(start + size, len(full))
        content = full[start:end].strip()
        if content:
            chunks.append({
                "chunk_index": idx,
                "page_start": page_at(start),
                "page_end": page_at(max(start, end - 1)),
                "content": content,
            })
            idx += 1
        if end == len(full):
            break
        start += step
    return chunks


# ------------------------------------------------------------
# PART B windowing: page-preserving sliding windows with
# explicit [PAGE N] markers injected into the text stream.
# (Legacy; kaalkram extract uses extract_windows_for_doc instead.)
# ------------------------------------------------------------
def sliding_windows(pages: list[str]) -> list[dict]:
    size = settings.window_size
    overlap = settings.window_overlap
    windows: list[dict] = []
    start = 0
    n = len(pages)
    while start < n:
        end = min(start + size, n)
        body = "\n".join(
            f"[PAGE {start + i + 1}]\n{pages[start + i]}"
            for i in range(end - start)
        )
        windows.append({
            "id": f"win_{start + 1}_{end}",
            "start": start + 1,
            "end": end,
            "text": body.strip(),
        })
        if end == n:
            break
        start += max(1, size - overlap)
    return windows


def extract_windows_for_doc(doc) -> list[dict]:
    """
    Shared extract windows for v1 pass1 and v2: paragraph-aligned, capped by
    ``v2_window_chars`` / ``v2_window_max_paras``, with ``v2_window_overlap_paras``
    of read-only context.

    Returns windows in the pass1 shape (id, start, end as page numbers, text with
    [PAGE N] markers) plus char_start/char_end/para_ids for telemetry and dedupe.
    """
    from .ingest_v2 import windows as make_windows

    raw = make_windows(
        doc, settings.v2_window_chars, settings.v2_window_overlap_paras,
        break_level=0, max_paras=settings.v2_window_max_paras,
    )
    return [pass1_window_from_v2(doc, w) for w in raw]


def pass1_window_from_v2(doc, win: dict) -> dict:
    """Convert an ingest_v2 window into a pass1 window with [PAGE N] markers."""
    lines: list[str] = []
    page_nos: list[int] = []
    owned = set(win.get("para_ids") or [])
    context = set(win.get("context_para_ids") or [])

    for block in (win.get("text") or "").split("\n\n"):
        m = re.match(r"^\[([^\]]+)\] (.*)$", block, re.S)
        if not m:
            continue
        pid, body = m.group(1), m.group(2)
        is_ctx = body.startswith("(context")
        if is_ctx:
            body = re.sub(r"^\(context[^)]*\)\s*", "", body)
            body = body.lstrip("…")
        p = doc.para(pid) or doc.para(pid.split(".")[0])
        if p is not None:
            pages = doc.pages_for_span(p.start, p.end) or [1]
            page = pages[0]
            text = doc.text[p.start:p.end]
        else:
            pages = win.get("pages") or [1]
            page = pages[0]
            text = body
        if pid in owned or pid in context or is_ctx:
            page_nos.append(page)
            prefix = f"[PAGE {page}]"
            if is_ctx or pid in context:
                lines.append(f"{prefix} (context, already processed)\n{text}")
            else:
                lines.append(f"{prefix}\n{text}")

    pages_all = list(win.get("pages") or page_nos or [1])
    return {
        "id": win["id"],
        "start": min(pages_all),
        "end": max(pages_all),
        "text": "\n\n".join(lines).strip(),
        "char_start": win.get("start"),
        "char_end": win.get("end"),
        "para_ids": list(win.get("para_ids") or []),
        "context_para_ids": list(win.get("context_para_ids") or []),
    }


PAGE_RE = re.compile(r"\[PAGE\s+([0-9,\s]+)\]")


def parse_pages_from_bullet(line: str) -> list[int]:
    """Extract page numbers from a '[PAGE 8, 9] text' observation bullet."""
    m = PAGE_RE.search(line)
    if not m:
        return []
    return sorted({int(x) for x in re.findall(r"\d+", m.group(1))})
