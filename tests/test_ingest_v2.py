import fitz
import pytest

from app.ingest_v2 import extract_document, locate_quote, naive_chunks, windows

BODY = ("The messenger reached the hall at dusk and waited by the gate while the "
        "councillors argued about the tribute that had been promised long ago. ")


def _make_pdf(path, with_toc=False):
    doc = fitz.open()
    def page(lines):
        p = doc.new_page(width=420, height=600)
        p.insert_text((150, 20), "THE TEST BOOK", fontsize=8)          # running header
        y = 60
        for text, size in lines:
            p.insert_text((40, y), text, fontsize=size)
            y += size * 1.6
        p.insert_text((200, 585), str(doc.page_count), fontsize=8)     # page number
    page([("BOOK ONE", 18), ("", 1), ("SECTION I", 14), ("", 1),
          ("The king sat in the hall. He called for the old minister and", 10),
          ("asked him about the promise made to the river-folk.", 10), ("", 1),
          ("The minister spoke at length about the treaty, and it was a hard", 10),
          ("bar-", 10), ("gain that nobody had wanted to remember in", 10)])
    page([("those days of famine.", 10), ("", 1),
          ("SECTION II", 14), ("", 1),
          ("Years earlier, the river had flooded the western fields.", 10)])
    page([("BOOK TWO", 18), ("", 1), ("SECTION I", 14), ("", 1),
          ("A new season began and the envoys set out for the northern hills.", 10)])
    if with_toc:
        doc.set_toc([[1, "BOOK ONE", 1], [2, "SECTION I", 1], [2, "SECTION II", 2],
                     [1, "BOOK TWO", 3], [2, "SECTION I", 3]])
    doc.save(path)


@pytest.fixture(params=[False, True], ids=["font", "toc"])
def doc(tmp_path, request):
    p = tmp_path / "t.pdf"
    _make_pdf(p, with_toc=request.param)
    return extract_document(p)


def test_running_heads_and_page_numbers_removed(doc):
    assert "THE TEST BOOK" not in doc.text
    assert doc.stats["running_head_lines_removed"] >= 3


def test_dehyphenation_and_cross_page_join(doc):
    assert "bargain that nobody" in doc.text                       # bar-\ngain
    assert "remember in those days of famine." in doc.text          # joined across page 1->2


def test_structure(doc):
    titles = [(s.level, s.title) for s in doc.sections]
    assert (1, "BOOK ONE") in titles and (1, "BOOK TWO") in titles
    s2 = [s for s in doc.sections if s.title == "SECTION II"][0]
    parent = [s for s in doc.sections if s.id == s2.parent][0]
    assert parent.title == "BOOK ONE"
    off = doc.text.index("Years earlier")
    assert doc.section_path(off) == ["BOOK ONE", "SECTION II"]


def test_offsets_are_exact(doc):
    for p in doc.paragraphs:
        assert doc.text[p.start:p.end].strip() == doc.text[p.start:p.end]
    off = doc.text.index("Years earlier")
    assert doc.pages_for_span(off, off + 10) == [2]
    assert doc.pages_for_span(doc.text.index("A new season"), doc.text.index("A new season") + 5) == [3]
    # a paragraph crossing the page break maps to both pages
    s = doc.text.index("bargain"); e = doc.text.index("famine.")
    assert doc.pages_for_span(s, e) == [1, 2]


def test_windows_max_paras_and_char_caps(doc):
    ws = windows(doc, max_chars=1_500, overlap_paras=1, max_paras=12)
    assert all(len(w["para_ids"]) <= 12 for w in ws)
    assert all(len(w["text"]) <= 1_500 + 400 for w in ws)  # markers + 1-para context
    covered = [pid for w in ws for pid in w["para_ids"]]
    assert covered == [p.id for p in doc.paragraphs]


def test_windows_max_paras_cap(doc):
    ws = windows(doc, max_chars=50_000, overlap_paras=0, max_paras=2)
    assert all(len(w["para_ids"]) <= 2 for w in ws)
    covered = [pid for w in ws for pid in w["para_ids"]]
    assert covered == [p.id for p in doc.paragraphs]


def test_windows_never_cross_top_level_and_fit(doc):
    ws = windows(doc, max_chars=300, overlap_paras=1)
    assert all(len(w["text"]) <= 300 + 60 for w in ws)             # markers add a little
    book2 = doc.text.index("BOOK TWO")
    for w in ws:
        assert not (w["start"] < book2 < w["end"])
    first_b2 = [w for w in ws if w["start"] >= book2][0]
    assert first_b2["context_para_ids"] == []                        # no context from Book One
    covered = [pid for w in ws for pid in w["para_ids"]]
    assert covered == [p.id for p in doc.paragraphs]                  # every paragraph exactly once


def test_long_paragraph_split(tmp_path):
    d = fitz.open(); p = d.new_page(width=420, height=2000); y = 40
    for _ in range(60):
        p.insert_text((20, y), BODY[:70], fontsize=9); y += 13
    d.save(tmp_path / "long.pdf")
    doc = extract_document(tmp_path / "long.pdf")
    ws = windows(doc, max_chars=1000)
    assert len(ws) >= 3 and all(len(w["text"]) <= 1000 + 120 for w in ws)


def test_naive_chunks_offsets(doc):
    ch = naive_chunks(doc, 120, 20)
    for c in ch:
        assert doc.text[c["char_start"]:c["char_end"]] == c["content"]
    assert ch[0]["char_start"] == 0 and ch[-1]["char_end"] == len(doc.text)


def test_locate_quote(doc):
    q = "it was a hard   bargain that nobody had wanted"
    s, e = locate_quote(doc.text, q)
    assert doc.text[s:e] == "it was a hard bargain that nobody had wanted"
    assert locate_quote(doc.text, "this sentence is not in the book at all") is None
    assert locate_quote(doc.text, "SECTION I") is None               # too short / ambiguous
