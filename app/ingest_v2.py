"""
Structure-aware ingestion (v2).

Turns a PDF into ONE clean text string plus maps that locate every page,
section (heading hierarchy) and paragraph inside it by character offset.

Everything downstream cites by character offset, so retrieval metrics can
compare naive chunks, v1 events (pages) and v2 events on the same yardstick.

Nothing here knows which book it is reading. Structure comes from, in order:
  1. the PDF outline (get_toc), if the file has one;
  2. font-size clustering: lines noticeably larger than body text are headings;
  3. a typographic fallback: short standalone lines that are mostly upper-case.
Running headers/footers are detected by repetition across pages, not by a list.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz  # pymupdf

# Tunables (hyperparameters, not book knowledge). Report them with results.
MARGIN_BAND = 0.08          # top/bottom fraction of a page scanned for headers/footers
REPEAT_MIN_FRAC = 0.30      # a margin line on >=30% of pages is a running header/footer
HEADING_SIZE_RATIO = 1.12   # >=12% larger than body text counts as a heading
HEADING_MAX_CHARS = 120
CAPS_HEADING_MAX_CHARS = 60
CAPS_HEADING_MIN_UPPER = 0.8
PARA_SEP = "\n\n"

_TERMINAL = tuple('.!?"\'”’)]:;')


@dataclass
class Page:
    no: int          # 1-based
    start: int
    end: int


@dataclass
class Section:
    id: str
    level: int       # 1 = top level
    title: str
    start: int       # offset of the heading text
    end: int         # offset where the next heading of same/higher level starts
    parent: str | None = None


@dataclass
class Paragraph:
    id: str          # "p{n}" — stable, used as the citation marker in windows
    start: int
    end: int
    page_start: int
    page_end: int
    section_id: str | None
    is_heading: bool = False


@dataclass
class Document:
    text: str
    pages: list[Page]
    sections: list[Section]
    paragraphs: list[Paragraph]
    structure_source: str                    # "toc" | "font" | "caps" | "none"
    stats: dict = field(default_factory=dict)

    # ---------- lookups ----------
    def page_span(self, page_no: int) -> tuple[int, int] | None:
        for p in self.pages:
            if p.no == page_no:
                return (p.start, p.end)
        return None

    def pages_for_span(self, start: int, end: int) -> list[int]:
        return [p.no for p in self.pages if p.start < end and start < p.end]

    def para(self, pid: str) -> Paragraph | None:
        return self._pindex().get(pid)

    def _pindex(self) -> dict[str, Paragraph]:
        if not hasattr(self, "_pi"):
            self._pi = {p.id: p for p in self.paragraphs}
        return self._pi

    def section_path(self, offset: int) -> list[str]:
        """Titles of all sections containing offset, outermost first."""
        path = [s for s in self.sections if s.start <= offset < s.end]
        path.sort(key=lambda s: s.level)
        return [s.title for s in path]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "pages": [asdict(p) for p in self.pages],
            "sections": [asdict(s) for s in self.sections],
            "paragraphs": [asdict(p) for p in self.paragraphs],
            "structure_source": self.structure_source,
            "stats": self.stats,
        }

    @staticmethod
    def from_dict(d: dict) -> "Document":
        return Document(
            text=d["text"],
            pages=[Page(**p) for p in d["pages"]],
            sections=[Section(**s) for s in d["sections"]],
            paragraphs=[Paragraph(**p) for p in d["paragraphs"]],
            structure_source=d.get("structure_source", "none"),
            stats=d.get("stats", {}),
        )


def doc_id_for(pdf_path: Path) -> str:
    h = hashlib.sha1(Path(pdf_path).read_bytes()).hexdigest()[:16]
    return f"doc_{h}"


# ============================================================
# Raw line extraction
# ============================================================
@dataclass
class _Line:
    page: int
    block: int
    y0: float
    y1: float
    page_h: float
    text: str
    size: float
    bold: bool


def _raw_lines(pdf: fitz.Document) -> list[_Line]:
    out: list[_Line] = []
    for pno, page in enumerate(pdf, start=1):
        h = page.rect.height or 1.0
        d = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for bno, block in enumerate(d.get("blocks", [])):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in line["spans"]).strip()
                chars = sum(len(s["text"]) for s in spans)
                size = sum(s["size"] * len(s["text"]) for s in spans) / max(1, chars)
                bold_chars = sum(len(s["text"]) for s in spans if s.get("flags", 0) & 16)
                y0, y1 = line["bbox"][1], line["bbox"][3]
                out.append(_Line(pno, bno, y0, y1, h, text, round(size, 1),
                                 bold_chars >= 0.6 * chars))
    return out


def _norm_margin(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _strip_running_heads(lines: list[_Line], n_pages: int) -> tuple[list[_Line], int]:
    """Remove lines in the top/bottom band that repeat across many pages, and bare page numbers."""
    def in_band(l: _Line) -> bool:
        return l.y0 < l.page_h * MARGIN_BAND or l.y1 > l.page_h * (1 - MARGIN_BAND)

    counts = Counter(_norm_margin(l.text) for l in lines if in_band(l))
    pages_with = {k: set() for k in counts}
    for l in lines:
        if in_band(l):
            pages_with[_norm_margin(l.text)].add(l.page)
    threshold = max(3, int(REPEAT_MIN_FRAC * n_pages))
    repeated = {k for k, pg in pages_with.items() if len(pg) >= threshold}

    kept, removed = [], 0
    for l in lines:
        if in_band(l):
            key = _norm_margin(l.text)
            if key in repeated or re.fullmatch(r"[\divxlcdmIVXLCDM\-–— .]{1,8}", l.text.strip()):
                removed += 1
                continue
        kept.append(l)
    return kept, removed


def _body_size(lines: list[_Line]) -> float:
    c: Counter = Counter()
    for l in lines:
        c[l.size] += len(l.text)
    return c.most_common(1)[0][0] if c else 10.0


# ============================================================
# Blocks -> paragraphs
# ============================================================
@dataclass
class _Para:
    lines: list[_Line]
    heading_level: int = 0     # 0 = body

    @property
    def page_start(self) -> int:
        return self.lines[0].page

    @property
    def page_end(self) -> int:
        return self.lines[-1].page


PARA_GAP_RATIO = 0.9        # vertical gap > 0.9 line-heights starts a new paragraph
SIZE_CHANGE_RATIO = 0.05    # font size change > 5% starts a new paragraph


def _group_lines(lines: list[_Line]) -> list[list[_Line]]:
    """Split the line stream into paragraphs. A new paragraph starts when:
    the page changes; the font size changes (heading <-> body); there is a
    blank-line-sized vertical gap; or the PDF block changes right after a
    sentence end. Mid-sentence block/page changes are joined later."""
    groups: list[list[_Line]] = []
    prev: _Line | None = None
    for l in lines:
        new = prev is None or l.page != prev.page
        if not new:
            height = max(1.0, prev.y1 - prev.y0)
            gap = l.y0 - prev.y1
            size_change = abs(l.size - prev.size) > SIZE_CHANGE_RATIO * max(l.size, prev.size)
            ended = prev.text.rstrip().endswith(_TERMINAL)
            new = (size_change or gap > PARA_GAP_RATIO * height
                   or (l.block != prev.block and ended))
        if new:
            groups.append([])
        groups[-1].append(l)
        prev = l
    return groups


def _is_caps_heading(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3 or len(text) > CAPS_HEADING_MAX_CHARS:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= CAPS_HEADING_MIN_UPPER


def _classify_headings(groups: list[list[_Line]], body: float) -> tuple[list[_Para], str]:
    paras = [_Para(g) for g in groups]

    # Font-size headings
    sizes = sorted({max(l.size for l in p.lines) for p in paras
                    if len(p.lines) <= 3
                    and max(l.size for l in p.lines) >= body * HEADING_SIZE_RATIO
                    and len(" ".join(l.text for l in p.lines)) <= HEADING_MAX_CHARS},
                   reverse=True)
    level_of = {s: i + 1 for i, s in enumerate(sizes)}
    found = False
    for p in paras:
        s = max(l.size for l in p.lines)
        txt = " ".join(l.text for l in p.lines)
        if s in level_of and len(p.lines) <= 3 and len(txt) <= HEADING_MAX_CHARS:
            p.heading_level = level_of[s]
            found = True
    if found:
        return paras, "font"

    # Typographic fallback: standalone, short, mostly upper-case
    for p in paras:
        if len(p.lines) == 1 and _is_caps_heading(p.lines[0].text):
            p.heading_level = 1
            found = True
    return paras, ("caps" if found else "none")


def _merge_across_breaks(paras: list[_Para]) -> list[_Para]:
    """Join a body paragraph that was split by a page/column break mid-sentence."""
    out: list[_Para] = []
    for p in paras:
        if (out and p.heading_level == 0 and out[-1].heading_level == 0):
            prev_last = out[-1].lines[-1].text.rstrip()
            first = p.lines[0].text.lstrip()
            if (prev_last and not prev_last.endswith(_TERMINAL)
                    and first[:1].islower()):
                out[-1].lines.extend(p.lines)
                continue
        out.append(p)
    return out


def _join_lines(lines: list[_Line]) -> tuple[str, list[tuple[int, int]]]:
    """Join lines into one paragraph string with de-hyphenation.
    Returns (text, [(char_offset_within_para, page_no) at each line start])."""
    buf = ""
    marks: list[tuple[int, int]] = []
    for i, l in enumerate(lines):
        t = re.sub(r"\s+", " ", l.text).strip()
        if not t:
            continue
        if buf:
            if buf.endswith("-") and t[:1].islower() and len(buf) > 1 and buf[-2].isalpha():
                buf = buf[:-1]              # exam-\nple -> example
            else:
                buf += " "
        marks.append((len(buf), l.page))
        buf += t
    return buf, marks


# ============================================================
# TOC-driven structure
# ============================================================
def _toc_headings(pdf: fitz.Document) -> list[tuple[int, str, int]]:
    try:
        toc = pdf.get_toc(simple=True)
    except Exception:
        return []
    return [(lvl, title.strip(), page) for lvl, title, page in toc if title.strip()]


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ============================================================
# Public entry point
# ============================================================
def extract_document(pdf_path: Path) -> Document:
    pdf = fitz.open(pdf_path)
    try:
        n_pages = len(pdf)
        lines = _raw_lines(pdf)
        lines, removed = _strip_running_heads(lines, n_pages)
        body = _body_size(lines)
        paras, source = _classify_headings(_group_lines(lines), body)
        toc = _toc_headings(pdf)
    finally:
        pdf.close()

    if toc:
        # The outline is authoritative: clear font guesses, mark matching paragraphs.
        for p in paras:
            p.heading_level = 0
        wanted = [(lvl, _norm_title(t), pg) for lvl, t, pg in toc]
        used = set()
        for lvl, nt, pg in wanted:
            for i, p in enumerate(paras):
                if i in used or not (pg - 1 <= p.page_start <= pg + 1):
                    continue
                pt = _norm_title(" ".join(l.text for l in p.lines))
                if pt and (pt == nt or (len(pt) <= len(nt) + 20 and pt.startswith(nt))):
                    p.heading_level = lvl
                    used.add(i)
                    break
        source = "toc" if used else source

    paras = _merge_across_breaks(paras)

    # ---- assemble text + maps
    text_parts: list[str] = []
    cursor = 0
    paragraphs: list[Paragraph] = []
    line_marks: list[tuple[int, int]] = []           # (global offset, page)
    heads: list[tuple[int, str, int]] = []           # (level, title, para index)
    for p in paras:
        body_text, marks = _join_lines(p.lines)
        if not body_text:
            continue
        if text_parts:
            text_parts.append(PARA_SEP)
            cursor += len(PARA_SEP)
        start = cursor
        text_parts.append(body_text)
        cursor += len(body_text)
        line_marks.extend((start + off, pg) for off, pg in marks)
        idx = len(paragraphs)
        paragraphs.append(Paragraph(
            id=f"p{idx + 1}", start=start, end=cursor,
            page_start=p.page_start, page_end=p.page_end,
            section_id=None, is_heading=p.heading_level > 0,
        ))
        if p.heading_level:
            heads.append((p.heading_level, body_text, idx))
    text = "".join(text_parts)

    # ---- page spans from line marks (exact even when a paragraph crosses a page)
    pages: list[Page] = []
    for i, (off, pg) in enumerate(line_marks):
        if pages and pages[-1].no == pg:
            continue
        if pages:
            pages[-1].end = off
        pages.append(Page(no=pg, start=off, end=len(text)))
    if pages:
        pages[0].start = 0

    # ---- sections
    sections: list[Section] = []
    for j, (lvl, title, pidx) in enumerate(heads):
        start = paragraphs[pidx].start
        end = len(text)
        for lvl2, _, pidx2 in heads[j + 1:]:
            if lvl2 <= lvl:
                end = paragraphs[pidx2].start
                break
        parent = None
        for s in reversed(sections):
            if s.level < lvl and s.start <= start < s.end:
                parent = s.id
                break
        sections.append(Section(id=f"s{j + 1}", level=lvl, title=title,
                                start=start, end=end, parent=parent))
    for para in paragraphs:
        inner = [s for s in sections if s.start <= para.start < s.end]
        if inner:
            para.section_id = max(inner, key=lambda s: s.level).id

    return Document(
        text=text, pages=pages, sections=sections, paragraphs=paragraphs,
        structure_source=source,
        stats={
            "pages": n_pages,
            "chars": len(text),
            "paragraphs": len(paragraphs),
            "sections": len(sections),
            "running_head_lines_removed": removed,
            "body_font_size": body,
        },
    )


# ============================================================
# Windows for LLM extraction (paragraph-aligned, never truncated)
# ============================================================
def _split_long(text: str, start: int, max_chars: int) -> list[tuple[int, int]]:
    """Split an over-long paragraph at sentence ends; returns absolute spans."""
    spans, s = [], 0
    while s < len(text):
        e = min(len(text), s + max_chars)
        if e < len(text):
            cut = max(text.rfind(". ", s, e), text.rfind("? ", s, e), text.rfind("! ", s, e))
            if cut > s:
                e = cut + 1
        spans.append((start + s, start + e))
        s = e
        while s < len(text) and text[s] == " ":
            s += 1
    return spans


def windows(doc: Document, max_chars: int = 12000, overlap_paras: int = 2,
            break_level: int = 1) -> list[dict]:
    """
    Paragraph-aligned windows sized to fit the model WITHOUT truncation.

    Each window lists the paragraphs it owns ("new") and a few preceding
    paragraphs as read-only context. Windows never cross a heading of level
    <= break_level, so each top-level unit can be processed on its own.
    Text is marked "[p17] ..." so extracted items cite paragraph ids, which map
    back to exact character offsets.
    """
    units: list[tuple[str, int, int, bool]] = []    # (label, start, end, is_heading)
    unit_max = max_chars - (max_chars // 4 if overlap_paras else 0) - 16
    for p in doc.paragraphs:
        if p.end - p.start <= unit_max:
            units.append((p.id, p.start, p.end, p.is_heading))
        else:
            for k, (a, b) in enumerate(_split_long(doc.text[p.start:p.end], p.start, unit_max)):
                units.append((f"{p.id}.{k + 1}", a, b, False))

    top_heading_ids = {p.id for p in doc.paragraphs if p.is_heading and any(
        s.level <= break_level and s.start == p.start for s in doc.sections)}

    out: list[dict] = []
    cur: list[tuple[str, int, int, bool]] = []
    size = 0
    # Context from the previous window is capped at a quarter of the budget, so
    # a window's total size stays within max_chars (plus marker overhead).
    ctx_budget = max_chars // 4 if overlap_paras else 0
    own_budget = max_chars - ctx_budget

    def flush():
        nonlocal cur, size
        if not cur:
            return
        prev = out[-1]["_units"] if out else []
        # context only within the same top-level unit
        cand = prev[-overlap_paras:] if (prev and overlap_paras
                                        and cur[0][0].split(".")[0] not in top_heading_ids) else []
        ctx, used = [], 0
        for u in reversed(cand):
            room = ctx_budget - used
            if room <= 40:
                break
            a = max(u[1], u[2] - room)               # keep the tail nearest the new text
            ctx.insert(0, (u[0], a, u[2], u[1] != a))
            used += u[2] - a
        lines = [f"[{u[0]}] (context{', truncated' if u[3] else ''}, already processed) "
                 f"{'…' if u[3] else ''}{doc.text[u[1]:u[2]]}" for u in ctx]
        lines += [f"[{u[0]}] {doc.text[u[1]:u[2]]}" for u in cur]
        start, end = cur[0][1], cur[-1][2]
        out.append({
            "id": f"win_{len(out) + 1}",
            "start": start,
            "end": end,
            "para_ids": [u[0] for u in cur],
            "context_para_ids": [u[0] for u in ctx],
            "section_path": doc.section_path(start),
            "pages": doc.pages_for_span(start, end),
            "text": "\n\n".join(lines),
            "_units": cur,
        })
        cur, size = [], 0

    for u in units:
        ulen = u[2] - u[1] + 8
        if cur and (size + ulen > own_budget or u[0].split(".")[0] in top_heading_ids):
            flush()
        cur.append(u)
        size += ulen
    flush()
    for w in out:
        w.pop("_units", None)
    return out


# ============================================================
# Naive chunks (unchanged behaviour, now with character offsets)
# ============================================================
def naive_chunks(doc: Document, size: int, overlap: int) -> list[dict]:
    """Fixed-size character windows over the cleaned text — the honest baseline.
    Same text the v2 pipeline reads, so neither side gets cleaner input."""
    text = doc.text
    chunks: list[dict] = []
    step = max(1, size - overlap)
    start = idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        content = text[start:end]
        if content.strip():
            pages = doc.pages_for_span(start, end) or [0]
            chunks.append({
                "chunk_index": idx, "char_start": start, "char_end": end,
                "page_start": pages[0], "page_end": pages[-1], "content": content,
            })
            idx += 1
        if end == len(text):
            break
        start += step
    return chunks


# ============================================================
# Quote -> offset verification (used to build the gold set)
# ============================================================
_QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})


def _normalise_with_map(s: str) -> tuple[str, list[int]]:
    """Lower-case, unify quotes/dashes, collapse whitespace; keep index map to original."""
    out, idx = [], []
    prev_space = False
    for i, ch in enumerate(s.translate(_QUOTE_MAP)):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch.lower())
            prev_space = False
        idx.append(i)
    return "".join(out), idx


def locate_quote(doc_text: str, quote: str) -> tuple[int, int] | None:
    """Exact (normalised) location of a quote in the document, or None.
    Used to reject gold evidence the proposing model misquoted."""
    hay, hmap = _normalise_with_map(doc_text)
    needle, _ = _normalise_with_map(quote.strip())
    needle = needle.strip()
    if len(needle) < 12:
        return None
    pos = hay.find(needle)
    if pos < 0 or hay.find(needle, pos + 1) >= 0:   # missing or ambiguous
        return None
    return hmap[pos], hmap[pos + len(needle) - 1] + 1
