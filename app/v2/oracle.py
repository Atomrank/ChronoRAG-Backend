"""Deterministic synth oracle extractor (extractor=oracle).

Parses templated synthetic text into the same ExtractionResult shape as the LLM
path so entities → coref → graph → answer run unchanged. E-IDs stay internal
(gold keys only) — they are stripped from mention quotes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .. import buildlog
from ..config import settings
from ..ingest_v2 import Document, windows as make_windows
from .extract import (
    ExtractedMention, ExtractedRelation, ExtractionResult,
    _dedupe_mentions_by_span, _make_telling_mentions,
)
from .frames import FrameOp, FrameTracker
from .prompts import PROMPT_VERSION

# Event tag — internal gold key only; stripped from quotes.
_E_TAG = re.compile(r"\[(E\d{4})\]")
_CONNECTIVE = re.compile(
    r"^(?:Afterwards|After that|Later|Then|Next|Before that|Earlier|"
    r"Prior to this|First),?\s+",
    re.I,
)

# Discourse wrappers (paragraph-level).
_RECOLLECT_BEGIN = re.compile(
    r"^(?P<nar>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)\s+"
    r"(?:began a recollection|paused and recalled earlier days|looked back)\b"
)
_TALE = re.compile(
    r"^(?P<nar>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)\s+told a tale from another land\b"
)
_SUPPOSE = re.compile(r"^Suppose things had gone otherwise\b")
_FORETELL_PASS = re.compile(r"^In time the foretelling came to pass\b")
_FORESAW = re.compile(  # chapter-1 paraphrase — no E-tag; skip as mention
    r"^[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+\s+foresaw that\b"
)

# Event body patterns (after stripping connective + E-tag).
_BIRTH = re.compile(
    r"^(?P<who>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)\s+was born in\s+(?P<place>.+)$"
)
_DEATH = re.compile(
    r"^(?P<who>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)\s+died at\s+(?P<place>.+)$"
)
_MEANWHILE = re.compile(
    r"^Meanwhile,\s+in another part of\s+(?P<place>.+?),\s+"
    r"(?P<who>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)\s+(?P<action>.+)$"
)
_SAME_HOUR = re.compile(
    r"^At the same hour,\s+"
    r"(?P<who>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)\s+(?P<action>.+?)\s+at\s+(?P<place>.+)$"
)
_STANDARD = re.compile(
    r"^(?P<who>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)\s+(?P<action>.+?)\s+at\s+(?P<place>.+)$"
)


@dataclass
class _Hit:
    eid: str
    start: int
    end: int
    quote: str          # without [E####]
    who: str
    action: str
    place: str
    event_type: str     # birth | death | other
    mode: str
    frame_kind: str     # main | recollection | separate_tale | prediction | hypothetical
    discourse: str      # main | recollection | foretelling | counterfactual |
                        # meanwhile | same-hour | tale-from-another-land
    narrator: str
    para_id: str


def _para_id_at(doc: Document, offset: int) -> str:
    for p in doc.paragraphs:
        if p.start <= offset < p.end or (offset == p.end and p.end == len(doc.text)):
            return p.id
        if p.start <= offset <= p.end:
            return p.id
    # nearest preceding
    best = ""
    for p in doc.paragraphs:
        if p.start <= offset:
            best = p.id
    return best or (doc.paragraphs[0].id if doc.paragraphs else "")


def _classify_wrapper(para: str) -> tuple[str, str, str]:
    """Return (discourse_label, frame_kind, narrator) for the paragraph prefix."""
    s = para.strip()
    if _SUPPOSE.match(s):
        return "counterfactual", "hypothetical", ""
    if _FORETELL_PASS.match(s):
        return "foretelling", "prediction", ""
    if _FORESAW.match(s):
        return "foretelling", "prediction", ""
    m = _TALE.match(s)
    if m:
        return "tale-from-another-land", "separate_tale", m.group("nar")
    m = _RECOLLECT_BEGIN.match(s)
    if m:
        return "recollection", "recollection", m.group("nar")
    if s.startswith("Meanwhile,"):
        return "meanwhile", "main", ""
    if "At the same hour," in s[:80] or s.lstrip().startswith("At the same hour,"):
        # may have connective prefix
        body = _CONNECTIVE.sub("", s)
        if body.startswith("At the same hour,"):
            return "same-hour", "main", ""
    body = _CONNECTIVE.sub("", s)
    if body.startswith("Meanwhile,"):
        return "meanwhile", "main", ""
    if body.startswith("At the same hour,"):
        return "same-hour", "main", ""
    return "main", "main", ""


def _parse_event_body(body: str) -> tuple[str, str, str, str] | None:
    """who, action, place, event_type — or None if unparseable."""
    body = body.strip().rstrip(".")
    body = _CONNECTIVE.sub("", body).strip()
    m = _BIRTH.match(body)
    if m:
        return m.group("who"), "was born", m.group("place"), "birth"
    m = _DEATH.match(body)
    if m:
        return m.group("who"), "died", m.group("place"), "death"
    m = _MEANWHILE.match(body)
    if m:
        return m.group("who"), m.group("action").strip(), m.group("place"), "other"
    m = _SAME_HOUR.match(body)
    if m:
        return m.group("who"), m.group("action").strip(), m.group("place"), "other"
    m = _STANDARD.match(body)
    if m:
        return m.group("who"), m.group("action").strip(), m.group("place"), "other"
    return None


def _mode_for(frame_kind: str, discourse: str) -> str:
    if frame_kind == "recollection":
        return "recounted"
    if frame_kind == "hypothetical" or discourse == "counterfactual":
        return "hypothetical"
    if frame_kind == "prediction" and discourse == "foretelling":
        # fulfilment is the occurrence; chapter-1 foresaw has no E-tag
        return "occurs"
    if frame_kind == "separate_tale":
        return "recounted"
    return "occurs"


def parse_oracle_hits(doc: Document) -> list[_Hit]:
    """Scan document text for [E####]-tagged synth events."""
    text = doc.text
    hits: list[_Hit] = []
    for m in _E_TAG.finditer(text):
        eid = m.group(1)
        tag_start, tag_end = m.start(), m.end()
        # Clause containing the tag: after previous sentence end / newline.
        lo = tag_start
        while lo > 0 and text[lo - 1] not in ".\n!?":
            lo -= 1
        if lo > 0 and text[lo - 1] in ".!?":
            # skip the punctuation and following spaces
            lo_scan = lo
            while lo_scan < tag_start and text[lo_scan].isspace():
                lo_scan += 1
            lo = lo_scan
        while lo < tag_start and text[lo].isspace():
            lo += 1
        hi = tag_end
        while hi < len(text) and text[hi] not in "\n":
            if text[hi] == ".":
                hi += 1
                break
            hi += 1
        sentence = text[lo:hi].strip()
        body = _E_TAG.sub("", sentence).strip()
        body = re.sub(r"\s+\.", ".", body)
        body = re.sub(r"\s{2,}", " ", body)
        parsed = _parse_event_body(body)
        if not parsed:
            continue
        who, action, place, etype = parsed
        pid = _para_id_at(doc, lo)
        p = doc.para(pid)
        para_text = text[p.start:p.end] if p else sentence
        discourse, frame_kind, narrator = _classify_wrapper(para_text)
        quote = body.rstrip(".")
        # Prefer a locate-able quote: include trailing period if present in text
        if not quote.endswith(".") and text[max(0, hi - 1):hi] == ".":
            quote = quote + "."
        hits.append(_Hit(
            eid=eid, start=lo, end=hi, quote=quote,
            who=who, action=action, place=place, event_type=etype,
            mode=_mode_for(frame_kind, discourse),
            frame_kind=frame_kind, discourse=discourse,
            narrator=narrator, para_id=pid,
        ))
    seen: set[str] = set()
    uniq: list[_Hit] = []
    for h in hits:
        if h.eid in seen:
            continue
        seen.add(h.eid)
        uniq.append(h)
    return uniq


def extract_document_oracle(
    doc: Document, doc_id: str, *,
    on_progress=None,
) -> ExtractionResult:
    """Build ExtractionResult from regex parse of templated synth text."""
    wins = make_windows(
        doc, settings.v2_window_chars, settings.v2_window_overlap_paras,
        break_level=0, max_paras=settings.v2_window_max_paras,
    )
    hits = parse_oracle_hits(doc)
    if on_progress:
        on_progress(0.2, f"oracle parsed {len(hits)} events")

    tracker = FrameTracker(unit_id="U0", unit_start=0, unit_end=len(doc.text))
    mention_seq = [0]
    mentions: list[ExtractedMention] = []
    relations: list[ExtractedRelation] = []

    # Open/close frames around non-main hits; process in document order
    open_frame_at: int | None = None
    open_kind: str | None = None
    open_nar = ""

    def _close_if_open(at: int) -> None:
        nonlocal open_frame_at, open_kind, open_nar
        if open_kind and open_kind != "main" and len(tracker.stack) > 1:
            tracker.apply([FrameOp(op="close", at=at, frame_ref=tracker.stack[-1])])
        open_frame_at, open_kind, open_nar = None, None, ""

    for i, h in enumerate(hits):
        # Frame transitions
        if h.frame_kind != "main":
            if open_kind != h.frame_kind or open_nar != h.narrator:
                _close_if_open(h.start)
                if h.frame_kind in ("recollection", "separate_tale",
                                    "prediction", "hypothetical"):
                    tracker.apply([FrameOp(
                        op="open", at=h.start, frame_type=h.frame_kind,
                        narrator=h.narrator, summary=h.discourse,
                    )])
                    open_frame_at, open_kind, open_nar = h.start, h.frame_kind, h.narrator
        else:
            _close_if_open(h.start)

        mention_seq[0] += 1
        mid = f"m{mention_seq[0]}"
        desc = f"{h.who} {h.action}"
        if h.place and h.action not in ("was born",):
            if " at " not in desc and h.place:
                desc = f"{h.who} {h.action} at {h.place}"
        if h.event_type == "birth":
            desc = f"birth of {h.who}"
        elif h.event_type == "death":
            desc = f"death of {h.who}"

        mentions.append(ExtractedMention(
            id=mid, local_id=f"oracle_{h.eid}", window_id="oracle",
            frame_id="",  # assigned after finish
            start=h.start, end=h.end, para_id=h.para_id,
            quote=h.quote, description=desc, mode=h.mode,
            event_type=h.event_type,
            subject=h.who if h.event_type in ("birth", "death") else "",
            participants=[{"surface": h.who, "role": "agent"}],
            location=h.place,
            time_expressions=[],
            posthumous=False, sample_idx=0,
        ))

        # Sequential narrative_order within same frame (adjacent hits)
        if i > 0 and hits[i - 1].frame_kind == h.frame_kind == "main":
            if hits[i - 1].discourse in ("meanwhile", "same-hour") and \
               h.discourse in ("meanwhile", "same-hour"):
                prev_id = mentions[-2].id
                relations.append(ExtractedRelation(
                    a=prev_id, b=mid, rel="simultaneous",
                    cue="explicit_connective", consistency=1.0,
                    quote=h.quote, start=h.start, end=h.end,
                ))
            elif h.discourse == "main" and hits[i - 1].discourse == "main":
                prev_id = mentions[-2].id
                relations.append(ExtractedRelation(
                    a=prev_id, b=mid, rel="before",
                    cue="narrative_order", consistency=1.0,
                    quote=h.quote, start=h.start, end=h.end,
                ))

    _close_if_open(len(doc.text))
    frames = tracker.finish()
    for w in tracker.warnings:
        buildlog.record(doc_id, "oracle_frame_warning", detail=w)

    for m in mentions:
        m.frame_id = tracker.frame_at(m.start).id

    telling = _make_telling_mentions(doc, frames, mention_seq)
    mentions.extend(telling)
    mentions, n_dup = _dedupe_mentions_by_span(mentions)
    if n_dup:
        buildlog.record(doc_id, "oracle_mention_dedupe", detail={"dropped": n_dup})

    buildlog.record(
        doc_id, "oracle_extract",
        detail={"hits": len(hits), "mentions": len(mentions),
                "frames": len(frames), "prompt_version": PROMPT_VERSION},
    )
    if on_progress:
        on_progress(1.0, "oracle extract complete")

    return ExtractionResult(
        frames=frames, mentions=mentions, relations=relations,
        aliases=[], kinship=[], windows=wins,
        prompt_version=f"{PROMPT_VERSION}+oracle", extract_samples=0,
    )
