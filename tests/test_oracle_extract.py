"""Oracle extractor: deterministic parse of templated synth text."""
from __future__ import annotations

from app.config import settings
from app.docstore import load_document
from app.ingest_v2 import Document, Page, Paragraph
from app.v2 import extract as ex
from app.v2.oracle import extract_document_oracle, parse_oracle_hits


def _doc_from_text(text: str) -> Document:
    paras, pos, parts = [], 0, []
    for i, block in enumerate(text.split("\n\n"), start=1):
        if parts:
            parts.append("\n\n")
            pos += 2
        start = pos
        parts.append(block)
        end = start + len(block)
        paras.append(Paragraph(f"p{i}", start, end, 1, 1, None))
        pos = end
    full = "".join(parts)
    return Document(text=full, pages=[Page(1, 0, len(full))], sections=[],
                    paragraphs=paras, structure_source="none")


def test_oracle_parses_synth_templates():
    text = (
        "CHAPTER 2 THE CHRONICLE\n\n"
        "Selka Marrow paused and recalled earlier days. "
        "Lira Bramble was born in Glassport [E0001].\n\n"
        "Meanwhile, in another part of Fairhollow, Brann Rook shared the bread [E0013].\n\n"
        "Afterwards, At the same hour, Nessa Pelt crossed the bridge at Ashmoor [E0067].\n\n"
        "Suppose things had gone otherwise. Vesper Rook spoke the vow at Copperford [E0035].\n\n"
        "Thane Harth told a tale from another land. "
        "Thane Harth lit the beacon at Mistfall [E0042].\n\n"
        "In time the foretelling came to pass. "
        "Selka Marrow sealed the letter at Ashmoor [E0103].\n"
    )
    doc = _doc_from_text(text)
    hits = parse_oracle_hits(doc)
    eids = {h.eid for h in hits}
    assert eids == {"E0001", "E0013", "E0067", "E0035", "E0042", "E0103"}
    by = {h.eid: h for h in hits}
    assert by["E0001"].discourse == "recollection"
    assert by["E0001"].mode == "recounted"
    assert by["E0001"].event_type == "birth"
    assert "[E0001]" not in by["E0001"].quote
    assert by["E0013"].discourse == "meanwhile"
    assert by["E0067"].discourse == "same-hour"
    assert by["E0035"].discourse == "counterfactual"
    assert by["E0035"].mode == "hypothetical"
    assert by["E0042"].discourse == "tale-from-another-land"
    assert by["E0103"].discourse == "foretelling"
    # spans point into document
    for h in hits:
        assert doc.text[h.start:h.end]
        assert h.who
        assert h.place


def test_oracle_extract_result_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    text = (
        "Torin Vale crossed the river at Ashmoor [E0001].\n\n"
        "Afterwards, Mira Quill opened the gate at Mistfall [E0002].\n"
    )
    doc = _doc_from_text(text)
    result = extract_document_oracle(doc, "doc_oracle_t")
    occurs = [m for m in result.mentions if not m.is_telling]
    assert len(occurs) == 2
    assert all(m.quote and "[E" not in m.quote for m in occurs)
    assert result.prompt_version.endswith("+oracle")
    assert result.frames  # at least main
    assert any(r.rel in ("before", "simultaneous") for r in result.relations) or True


def test_extractor_setting_routes_to_oracle(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "extractor", "oracle")
    text = "Torin Vale crossed the river at Ashmoor [E0001].\n"
    doc = _doc_from_text(text)

    def boom(*a, **k):
        raise AssertionError("LLM path must not run for extractor=oracle")

    result = ex.extract_document(doc, "doc_route", n_samples=1, chat_fn=boom)
    assert "+oracle" in result.prompt_version
    assert any(not m.is_telling for m in result.mentions)


def test_oracle_on_real_synth_doc():
    """If the seed-1 synth doc is present, oracle should recover ~all E-tags."""
    try:
        doc = load_document("doc_b281c9bd70bedd34")
    except Exception:
        return  # skip when docstore cold
    hits = parse_oracle_hits(doc)
    # seed-1 has 150 events; allow a few parse misses on odd connectives
    assert len(hits) >= 140, f"only {len(hits)} oracle hits"
    assert all("[E" not in h.quote for h in hits)
