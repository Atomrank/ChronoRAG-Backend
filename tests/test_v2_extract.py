"""Tests for app.v2.extract — mocked LLM, invented names only."""
from __future__ import annotations

import json
import re

import pytest

from app import llm
from app.config import settings
from app.ingest_v2 import Document, Page, Paragraph
from app.v2.extract import (
    _process_primary, extract_document, extract_window,
)
from app.v2.frames import FrameTracker
from app.v2.schemas import (
    FrameOpOut, MentionOut, ParticipantOut, WindowExtraction,
)


@pytest.fixture(autouse=True)
def _no_buildlog(monkeypatch):
    monkeypatch.setattr("app.v2.extract.buildlog.record", lambda *a, **k: None)


def _doc(*paras: str) -> Document:
    text_parts = []
    paragraphs = []
    pos = 0
    for i, p in enumerate(paras, start=1):
        if text_parts:
            text_parts.append("\n\n")
            pos += 2
        start = pos
        text_parts.append(p)
        end = start + len(p)
        paragraphs.append(Paragraph(f"p{i}", start, end, 1, 1, None))
        pos = end
    text = "".join(text_parts)
    return Document(text=text, pages=[Page(1, 0, len(text))], sections=[],
                    paragraphs=paragraphs, structure_source="none")


def _win(doc: Document, para_ids: list[str], *, context: list[str] | None = None,
         wid: str = "win_1") -> dict:
    owned = [doc.para(p) for p in para_ids]
    ctx = [doc.para(p) for p in (context or [])]
    start = min(p.start for p in owned)
    end = max(p.end for p in owned)
    if ctx:
        start = min(start, min(p.start for p in ctx))
    lines = []
    for p in ctx:
        lines.append(f"[{p.id}] (context, already processed) {doc.text[p.start:p.end]}")
    for p in owned:
        lines.append(f"[{p.id}] {doc.text[p.start:p.end]}")
    return {
        "id": wid, "start": start, "end": end,
        "para_ids": para_ids, "context_para_ids": context or [],
        "section_path": [], "pages": [1], "text": "\n\n".join(lines),
    }


def _empty_extraction(**kwargs) -> WindowExtraction:
    base = dict(frame_ops=[], mentions=[], relations=[], aliases=[], kinship=[])
    base.update(kwargs)
    return WindowExtraction(**base)


def test_quote_resolution_and_context_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    doc = _doc(
        "Earlier, Torin crossed the river alone.",
        "Then Mira opened the gate and called loudly.",
        "Pelin answered from the tower steps.",
    )
    win = _win(doc, ["p2", "p3"], context=["p1"])
    extraction = _empty_extraction(
        mentions=[
            MentionOut(
                local_id="m1", para_id="p1",
                quote="Torin crossed the river",
                description="Torin crosses the river",
                event_type="other", subject="", participants=[
                    ParticipantOut(name="Torin", role="agent")],
                location="", mode="occurs", posthumous=False, time_expressions=[],
            ),
            MentionOut(
                local_id="m2", para_id="p2",
                quote="Mira opened the gate",
                description="Mira opens the gate",
                event_type="other", subject="", participants=[
                    ParticipantOut(name="Mira", role="agent")],
                location="", mode="occurs", posthumous=False, time_expressions=[],
            ),
            MentionOut(
                local_id="m3", para_id="p3",
                quote="this quote is not in the text at all xyz",
                description="missing",
                event_type="other", subject="", participants=[],
                location="", mode="occurs", posthumous=False, time_expressions=[],
            ),
        ],
    )
    tracker = FrameTracker(unit_id="U0", unit_start=0, unit_end=len(doc.text))
    mentions, *_rest = _process_primary(
        doc, win, extraction, tracker, "doc_ctx", [0])
    ids = {m.local_id for m in mentions}
    assert "m2" in ids
    assert "m1" not in ids  # context paragraph rejected
    assert "m3" not in ids  # unresolved quote dropped
    mira = next(m for m in mentions if m.local_id == "m2")
    assert doc.text[mira.start:mira.end] == "Mira opened the gate"


def test_new_n_frame_mapping_and_telling(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    doc = _doc(
        "Vela said, listen to my tale of the old hill.",
        "Long ago Soren built a bridge of oak.",
        "Thus ended her tale, and the hall grew quiet.",
    )

    def chat_fn(system, user, temperature):
        return _empty_extraction(
            frame_ops=[
                FrameOpOut(
                    op="open", para_id="p1", quote="listen to my tale",
                    frame_type="recollection", narrator="Vela", listener="",
                    frame_ref="", summary="Vela recounts the hill",
                ),
                FrameOpOut(
                    op="close", para_id="p3", quote="Thus ended her tale",
                    frame_type="none", narrator="", listener="",
                    frame_ref="NEW:1", summary="",
                ),
            ],
            mentions=[
                MentionOut(
                    local_id="m1", para_id="p2",
                    quote="Soren built a bridge of oak",
                    description="Soren builds a bridge",
                    event_type="other", subject="", participants=[
                        ParticipantOut(name="Soren", role="agent")],
                    location="", mode="recounted", posthumous=False, time_expressions=[],
                ),
            ],
        )

    result = extract_document(doc, "doc_frame", n_samples=1, chat_fn=chat_fn)
    assert any(f.type == "recollection" and f.narrator == "Vela" for f in result.frames)
    reco = next(f for f in result.frames if f.type == "recollection")
    assert reco.close_at is not None
    telling = [m for m in result.mentions if m.is_telling]
    assert len(telling) == 1
    assert telling[0].tells_frame == reco.id
    assert telling[0].frame_id == "U0/F0"
    assert any(p["role"] == "agent" and p["surface"] == "Vela" for p in telling[0].participants)


def test_cache_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    doc = _doc("Neris lit a lamp beside the door.")
    win = _win(doc, ["p1"])
    payload = _empty_extraction(
        mentions=[
            MentionOut(
                local_id="m1", para_id="p1", quote="Neris lit a lamp",
                description="Neris lights a lamp", event_type="other", subject="",
                participants=[ParticipantOut(name="Neris", role="agent")],
                location="", mode="occurs", posthumous=False, time_expressions=[],
            ),
        ],
    ).model_dump()
    from app.v2 import extract as ex
    cache_file = ex._cache_path("doc_cache", "win_1", 0)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("LLM should not be called when cache hits")

    out = extract_window(doc, win, [], doc_id="doc_cache", sample=0,
                         temperature=0.0, chat_fn=boom)
    assert out.mentions[0].local_id == "m1"


def test_dedupe_mentions_by_char_span(tmp_path, monkeypatch):
    """Overlapping extract hits on the same (start, end) keep the first only."""
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "v2_window_max_paras", 4)
    monkeypatch.setattr(settings, "v2_window_chars", 50_000)
    monkeypatch.setattr(settings, "v2_window_overlap_paras", 0)
    monkeypatch.setattr(settings, "v2_extract_min_mention_ratio", 0.0)  # never split

    doc = _doc(
        "Torin Vale crossed the river.",
        "Mira Quill opened the gate.",
    )
    calls = {"n": 0}

    def chat_fn(system, user, temperature):
        calls["n"] += 1
        # Emit both events every call so a second window hitting the same paras
        # would duplicate without dedupe. With overlap_paras=0 we only get one
        # window — simulate duplicates by returning the same quote twice.
        return _empty_extraction(
            mentions=[
                MentionOut(
                    local_id="m1", para_id="p1", quote="Torin Vale crossed the river",
                    description="Torin crosses", event_type="other", subject="",
                    participants=[ParticipantOut(name="Torin Vale", role="agent")],
                    location="", mode="occurs", posthumous=False, time_expressions=[],
                ),
                MentionOut(
                    local_id="m1b", para_id="p1", quote="Torin Vale crossed the river",
                    description="Torin crosses again", event_type="other", subject="",
                    participants=[ParticipantOut(name="Torin Vale", role="agent")],
                    location="", mode="occurs", posthumous=False, time_expressions=[],
                ),
                MentionOut(
                    local_id="m2", para_id="p2", quote="Mira Quill opened the gate",
                    description="Mira opens", event_type="other", subject="",
                    participants=[ParticipantOut(name="Mira Quill", role="agent")],
                    location="", mode="occurs", posthumous=False, time_expressions=[],
                ),
            ],
        )

    result = extract_document(doc, "doc_dedupe", n_samples=1, chat_fn=chat_fn)
    occurs = [m for m in result.mentions if not m.is_telling]
    spans = [(m.start, m.end) for m in occurs]
    assert len(spans) == len(set(spans))
    assert len(occurs) == 2


def test_extraction_max_tokens_has_2x_headroom():
    """Budget covers max_paras events at extract_tokens_per_event with 2x headroom."""
    needed = 2 * settings.v2_window_max_paras * settings.extract_tokens_per_event
    assert settings.extraction_max_tokens() >= needed
    assert settings.extraction_max_tokens() >= settings.v2_extract_max_tokens


def test_over_budget_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "llm_max_input_chars", 80)
    doc = _doc("A" * 200)
    win = _win(doc, ["p1"])
    with pytest.raises(llm.InputTooLongError):
        extract_window(doc, win, [], doc_id="doc_big", sample=0,
                       temperature=0.0, chat_fn=lambda *a, **k: _empty_extraction())


def test_split_on_low_mention_ratio(tmp_path, monkeypatch):
    """Dense one-event-per-para text: under-filling LLM must trigger bisect retries.

    Invented names only. The mock returns at most one mention per call (first owned
    para). Without the split guard that yields ~1 mention for the whole doc; with it,
    smaller windows each get their own mention.
    """
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "v2_window_max_paras", 8)
    monkeypatch.setattr(settings, "v2_extract_min_mention_ratio", 0.5)
    monkeypatch.setattr(settings, "v2_extract_split_min_paras", 4)
    monkeypatch.setattr(settings, "v2_window_chars", 50_000)
    monkeypatch.setattr(settings, "v2_window_overlap_paras", 0)

    events = [
        ("Torin Vale", "crossed the river"),
        ("Mira Quill", "opened the gate"),
        ("Pelin Ash", "lit the lantern"),
        ("Soren Drift", "mended the roof"),
        ("Vela Thorn", "sealed the letter"),
        ("Neris Flint", "sounded the horn"),
        ("Kade Moss", "drew the map"),
        ("Liora Wren", "shared the bread"),
    ]
    paras = [f"{name} {action}." for name, action in events]
    doc = _doc(*paras)

    def chat_fn(system, user, temperature):
        # Parse owned para markers from the user message; emit only the first.
        owned = re.findall(r"^\[(p\d+)\] (?!\(context)(.+)$", user, re.M)
        if not owned:
            return _empty_extraction()
        pid, body = owned[0]
        # body ends with period from the paragraph
        quote = body.rstrip(".")[:40]
        name = body.split()[0] + " " + body.split()[1]
        return _empty_extraction(
            mentions=[
                MentionOut(
                    local_id="m1", para_id=pid, quote=quote,
                    description=f"{name} acts",
                    event_type="other", subject="", participants=[
                        ParticipantOut(name=name, role="agent")],
                    location="", mode="occurs", posthumous=False, time_expressions=[],
                ),
            ],
        )

    result = extract_document(doc, "doc_dense", n_samples=1, chat_fn=chat_fn)
    occurs = [m for m in result.mentions if not m.is_telling]
    # 8 paras → splits down to leaves of size < split_min (4) → at least 2 mentions;
    # with always-1-per-call we expect 4 leaves of size 2 → 4 mentions.
    assert len(occurs) >= 4, f"expected split to recover mentions, got {len(occurs)}"
    # Distinct paragraphs covered
    assert len({m.para_id for m in occurs}) >= 4


def test_no_split_when_ratio_ok(tmp_path, monkeypatch):
    """Adequate mention density must not bisect (avoids wasted LLM calls)."""
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "v2_window_max_paras", 8)
    monkeypatch.setattr(settings, "v2_extract_min_mention_ratio", 0.5)
    monkeypatch.setattr(settings, "v2_extract_split_min_paras", 4)
    monkeypatch.setattr(settings, "v2_window_overlap_paras", 0)

    doc = _doc(
        "Torin Vale crossed the river.",
        "Mira Quill opened the gate.",
        "Pelin Ash lit the lantern.",
        "Soren Drift mended the roof.",
    )
    calls = {"n": 0}

    def chat_fn(system, user, temperature):
        calls["n"] += 1
        owned = re.findall(r"^\[(p\d+)\] (?!\(context)(.+)$", user, re.M)
        mentions = []
        for i, (pid, body) in enumerate(owned):
            quote = body.rstrip(".")[:40]
            name = " ".join(body.split()[:2])
            mentions.append(MentionOut(
                local_id=f"m{i+1}", para_id=pid, quote=quote,
                description=f"{name} acts", event_type="other", subject="",
                participants=[ParticipantOut(name=name, role="agent")],
                location="", mode="occurs", posthumous=False, time_expressions=[],
            ))
        return _empty_extraction(mentions=mentions)

    result = extract_document(doc, "doc_ok", n_samples=1, chat_fn=chat_fn)
    occurs = [m for m in result.mentions if not m.is_telling]
    assert len(occurs) == 4
    assert calls["n"] == 1  # single window, no split
