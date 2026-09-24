"""Window extraction for Kaalkram v2: frames, mentions, relations, aliases, kinship."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import buildlog, llm
from ..config import settings
from ..ingest_v2 import Document, locate_quote, windows as make_windows
from .frames import Frame, FrameOp, FrameTracker
from .prompts import PROMPT_VERSION, extract_system
from .schemas import WindowExtraction


@dataclass
class ExtractedMention:
    id: str
    local_id: str
    window_id: str
    frame_id: str
    start: int
    end: int
    para_id: str
    quote: str
    description: str
    mode: str
    event_type: str = "other"
    subject: str = ""
    participants: list[dict] = field(default_factory=list)  # {surface, role}
    location: str = ""
    time_expressions: list[str] = field(default_factory=list)
    posthumous: bool = False
    is_telling: bool = False
    tells_frame: str | None = None
    sample_idx: int = 0
    event_id: str = ""  # filled by coref


@dataclass
class ExtractedRelation:
    a: str          # mention ids (global)
    b: str
    rel: str
    cue: str
    consistency: float = 1.0
    quote: str = ""
    start: int | None = None
    end: int | None = None


@dataclass
class ExtractedAlias:
    name_a: str
    name_b: str
    quote: str
    start: int | None = None
    end: int | None = None


@dataclass
class ExtractedKinship:
    parent: str
    child: str
    quote: str
    start: int | None = None
    end: int | None = None


@dataclass
class ExtractionResult:
    frames: list[Frame]
    mentions: list[ExtractedMention]
    relations: list[ExtractedRelation]
    aliases: list[ExtractedAlias]
    kinship: list[ExtractedKinship]
    windows: list[dict]
    prompt_version: str = PROMPT_VERSION
    extract_samples: int = 1


def _cache_path(doc_id: str, window_id: str, sample: int) -> Path:
    return settings.cache_path / f"{doc_id}_v2_{window_id}_s{sample}.json"


def _load_cache(doc_id: str, window_id: str, sample: int) -> dict | None:
    p = _cache_path(doc_id, window_id, sample)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_cache(doc_id: str, window_id: str, sample: int, data: dict) -> None:
    p = _cache_path(doc_id, window_id, sample)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _span_overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(1, min(a[1] - a[0], b[1] - b[0]))


def _first_sentence(text: str, at: int) -> tuple[int, int]:
    """Return (start, end) of the sentence containing `at`, clipped to nearby text."""
    n = len(text)
    lo = max(0, at - 200)
    hi = min(n, at + 400)
    chunk = text[lo:hi]
    # find sentence start
    rel = at - lo
    start_rel = 0
    for m in re.finditer(r"[.!?]\s+", chunk[:rel]):
        start_rel = m.end()
    end_rel = len(chunk)
    m2 = re.search(r"[.!?]", chunk[rel:])
    if m2:
        end_rel = rel + m2.end()
    return lo + start_rel, lo + end_rel


def _resolve_quote(doc: Document, quote: str, para_id: str | None,
                   win_start: int, win_end: int) -> tuple[int, int] | None:
    """Try paragraph span first, then the owned window span."""
    if para_id:
        base = para_id.split(".")[0]
        para = doc.para(base) or doc.para(para_id)
        if para:
            hit = locate_quote(doc.text[para.start:para.end], quote)
            if hit:
                return para.start + hit[0], para.start + hit[1]
    hit = locate_quote(doc.text[win_start:win_end], quote)
    if hit:
        return win_start + hit[0], win_start + hit[1]
    return None


def _user_message(stack: list[dict], section_path: list[str], text: str) -> str:
    return (
        f"FRAME STACK:\n{json.dumps(stack, ensure_ascii=False)}\n\n"
        f"SECTION: {' > '.join(section_path) if section_path else '(none)'}\n\n"
        f"{text}"
    )


def _call_extract(system: str, user: str, temperature: float,
                  call_meta: dict | None = None) -> WindowExtraction:
    """Call chat_structured; on ContentFilterError optionally retry via local LLM."""
    try:
        return llm.chat_structured(
            system, user, WindowExtraction, temperature=temperature,
            max_tokens=settings.extraction_max_tokens(),
            call_meta=call_meta)
    except llm.ContentFilterError:
        if settings.local_llm_base_url:
            meta = dict(call_meta or {})
            meta["local_retry"] = True
            return llm.chat_structured(
                system, user, WindowExtraction, temperature=temperature,
                max_tokens=settings.extraction_max_tokens(),
                deployment=f"local:{settings.local_llm_model or 'local'}",
                call_meta=meta)
        raise


def _extraction_from_dict(d: dict) -> WindowExtraction:
    return WindowExtraction.model_validate(d)


def extract_window(doc: Document, win: dict, stack: list[dict], *,
                   doc_id: str, sample: int, temperature: float,
                   chat_fn: Callable | None = None,
                   window_index: int | None = None) -> WindowExtraction:
    """Extract one window (with cache). Raises InputTooLongError rather than cutting."""
    cached = _load_cache(doc_id, win["id"], sample)
    if cached is not None:
        return _extraction_from_dict(cached)

    user = _user_message(stack, win.get("section_path") or [], win["text"])
    # Enforce budget: windows() sizes text; still raise if somehow over limit.
    if len(user) > settings.llm_max_input_chars:
        raise llm.InputTooLongError(
            f"window {win['id']} user message is {len(user)} chars > "
            f"llm_max_input_chars={settings.llm_max_input_chars}")

    call_meta = {
        "pipeline": "kaalkram_v2",
        "phase": "extract",
        "doc_id": doc_id,
        "window_id": win["id"],
        "window_index": window_index if window_index is not None else win["id"],
        "window_start": win.get("start"),
        "window_end": win.get("end"),
        "window_chars": len(win.get("text") or ""),
        "owned_paras": len(win.get("para_ids") or []),
        "sample": sample,
        "max_tokens_setting": settings.extraction_max_tokens(),
    }

    def _default(system, user_msg, temperature):
        return _call_extract(system, user_msg, temperature, call_meta=call_meta)

    fn = chat_fn or _default
    try:
        result = fn(extract_system(), user, temperature)
    except llm.OutputTruncatedError as exc:
        buildlog.record(doc_id, "extract_output_truncated", ref=win["id"],
                        detail={**call_meta, "error": str(exc)})
        raise
    except llm.ContentFilterError as exc:
        buildlog.record(doc_id, "content_filter", ref=win["id"],
                        detail={"sample": sample, "error": str(exc)})
        raise
    except Exception as exc:
        buildlog.record(doc_id, "failed_window", ref=win["id"],
                        detail={"sample": sample, "error": f"{type(exc).__name__}: {exc}"})
        raise

    payload = result.model_dump()
    _save_cache(doc_id, win["id"], sample, payload)
    return result


def _is_context_para(para_id: str, context: set[str]) -> bool:
    """True if para_id is a context paragraph (exact id or same base as a context id)."""
    if not para_id or not context:
        return False
    if para_id in context:
        return True
    base = para_id.split(".")[0]
    context_bases = {c.split(".")[0] for c in context}
    if base in context_bases:
        return True
    return any(para_id.startswith(c + ".") or c.startswith(para_id + ".") for c in context)


def _process_primary(doc: Document, win: dict, extraction: WindowExtraction,
                     tracker: FrameTracker, doc_id: str,
                     mention_seq: list[int]) -> tuple[
                         list[ExtractedMention], list[ExtractedRelation],
                         list[ExtractedAlias], list[ExtractedKinship],
                         dict[str, ExtractedMention]]:
    """Resolve quotes, apply frame ops, build mention/relation objects for sample 0."""
    context = set(win.get("context_para_ids") or [])
    local_to_mention: dict[str, ExtractedMention] = {}
    mentions: list[ExtractedMention] = []
    relations: list[ExtractedRelation] = []
    aliases: list[ExtractedAlias] = []
    kinship: list[ExtractedKinship] = []

    # Resolve frame ops in offset order; map NEW:n to the n-th open created in this window.
    sorted_raw: list[tuple[int, int, Any, tuple[int, int]]] = []
    for fo in extraction.frame_ops:
        if _is_context_para(fo.para_id, context):
            buildlog.record(doc_id, "rejected_context_frame_op", ref=win["id"],
                            detail={"para_id": fo.para_id})
            continue
        span = _resolve_quote(doc, fo.quote, fo.para_id, win["start"], win["end"])
        if span is None:
            buildlog.record(doc_id, "unresolved_quote", ref=win["id"],
                            detail={"kind": "frame_op", "quote": fo.quote, "para_id": fo.para_id})
            continue
        # closes before opens at the same offset (matches FrameTracker.apply)
        sorted_raw.append((span[0], 0 if fo.op == "close" else 1, fo, span))
    sorted_raw.sort(key=lambda x: (x[0], x[1]))

    open_n = 0
    new_map: dict[int, str] = {}
    for _at, _prio, fo, span in sorted_raw:
        if fo.op == "open":
            open_n += 1
            op = FrameOp(op="open", at=span[0], frame_type=fo.frame_type,
                         narrator=fo.narrator, listener=fo.listener, summary=fo.summary)
            before = {f.id for f in tracker.frames}
            tracker.apply([op])
            created = [f.id for f in tracker.frames if f.id not in before]
            if created:
                new_map[open_n] = created[-1]
        else:
            ref = fo.frame_ref or ""
            if ref.startswith("NEW:"):
                try:
                    n = int(ref.split(":", 1)[1])
                except ValueError:
                    n = -1
                mapped = new_map.get(n)
            else:
                mapped = ref or None
            tracker.apply([FrameOp(op="close", at=span[0], frame_ref=mapped)])

    for mo in extraction.mentions:
        if _is_context_para(mo.para_id, context):
            buildlog.record(doc_id, "rejected_context_mention", ref=win["id"],
                            detail={"local_id": mo.local_id, "para_id": mo.para_id})
            continue

        span = _resolve_quote(doc, mo.quote, mo.para_id, win["start"], win["end"])
        if span is None:
            buildlog.record(doc_id, "unresolved_quote", ref=win["id"],
                            detail={"kind": "mention", "local_id": mo.local_id, "quote": mo.quote})
            continue
        mention_seq[0] += 1
        mid = f"m{mention_seq[0]}"
        em = ExtractedMention(
            id=mid, local_id=mo.local_id, window_id=win["id"], frame_id="",  # after finish
            start=span[0], end=span[1], para_id=mo.para_id, quote=mo.quote,
            description=mo.description, mode=mo.mode, event_type=mo.event_type,
            subject=mo.subject,
            participants=[{"surface": p.name, "role": p.role} for p in mo.participants],
            location=mo.location, time_expressions=list(mo.time_expressions),
            posthumous=mo.posthumous, sample_idx=0,
        )
        mentions.append(em)
        local_to_mention[mo.local_id] = em

    for ro in extraction.relations:
        ma, mb = local_to_mention.get(ro.a), local_to_mention.get(ro.b)
        if not ma or not mb:
            continue
        span = _resolve_quote(doc, ro.quote, None, win["start"], win["end"])
        relations.append(ExtractedRelation(
            a=ma.id, b=mb.id, rel=ro.relation, cue=ro.cue, consistency=1.0,
            quote=ro.quote,
            start=span[0] if span else None, end=span[1] if span else None,
        ))

    for ao in extraction.aliases:
        span = _resolve_quote(doc, ao.quote, None, win["start"], win["end"])
        aliases.append(ExtractedAlias(
            name_a=ao.name_a, name_b=ao.name_b, quote=ao.quote,
            start=span[0] if span else None, end=span[1] if span else None,
        ))

    for ko in extraction.kinship:
        span = _resolve_quote(doc, ko.quote, None, win["start"], win["end"])
        kinship.append(ExtractedKinship(
            parent=ko.parent, child=ko.child, quote=ko.quote,
            start=span[0] if span else None, end=span[1] if span else None,
        ))

    return mentions, relations, aliases, kinship, local_to_mention


def _vote_relations(primary_rels: list[ExtractedRelation],
                    primary_mentions: list[ExtractedMention],
                    sample_extractions: list[tuple[WindowExtraction, dict]],
                    doc: Document) -> None:
    """Update consistency on primary relations from extra samples."""
    if not sample_extractions:
        for r in primary_rels:
            r.consistency = 1.0
        return
    # Build primary mention spans by local_id within window — samples use local ids
    # We match by quote-span overlap across samples' mentions.
    n_samples = 1 + len(sample_extractions)
    for rel in primary_rels:
        ma = next((m for m in primary_mentions if m.id == rel.a), None)
        mb = next((m for m in primary_mentions if m.id == rel.b), None)
        if not ma or not mb:
            rel.consistency = 1.0 / n_samples
            continue
        hits = 1  # primary itself
        for sextr, win in sample_extractions:
            # map sample local mentions to spans
            spans: dict[str, tuple[int, int]] = {}
            for mo in sextr.mentions:
                sp = _resolve_quote(doc, mo.quote, mo.para_id, win["start"], win["end"])
                if sp:
                    spans[mo.local_id] = sp
            found = False
            for ro in sextr.relations:
                if ro.relation != rel.rel:
                    continue
                sa, sb = spans.get(ro.a), spans.get(ro.b)
                if not sa or not sb:
                    continue
                if (_span_overlap(sa, (ma.start, ma.end)) >= 0.5
                        and _span_overlap(sb, (mb.start, mb.end)) >= 0.5):
                    found = True
                    break
            if found:
                hits += 1
        rel.consistency = hits / n_samples


def _make_telling_mentions(doc: Document, frames: list[Frame],
                           mention_seq: list[int]) -> list[ExtractedMention]:
    out: list[ExtractedMention] = []
    by_id = {f.id: f for f in frames}
    for fr in frames:
        if fr.type == "main":
            continue
        parent = by_id.get(fr.parent) if fr.parent else None
        parent_id = parent.id if parent else frames[0].id
        s, e = _first_sentence(doc.text, fr.open_at)
        quote = doc.text[s:e].strip()[:200]
        mention_seq[0] += 1
        parts = []
        if fr.narrator:
            parts.append({"surface": fr.narrator, "role": "agent"})
        if fr.listener:
            parts.append({"surface": fr.listener, "role": "present"})
        out.append(ExtractedMention(
            id=f"m{mention_seq[0]}", local_id=f"tell_{fr.id}", window_id="",
            frame_id=parent_id, start=s, end=e, para_id="", quote=quote,
            description=f"{fr.narrator or 'someone'} narrates: {fr.summary or fr.type}",
            mode="occurs", event_type="other", participants=parts,
            is_telling=True, tells_frame=fr.id, sample_idx=0,
        ))
    return out


def _owned_unit_spans(doc: Document, win: dict) -> list[tuple[str, int, int]]:
    """(para_id, start, end) for each owned unit, preserving dotted sub-spans."""
    owned = list(win.get("para_ids") or [])
    if not owned:
        return []
    # Prefer exact bodies from the window text markers (handles pN.k units).
    by_id: dict[str, tuple[int, int]] = {}
    cursor = win.get("start", 0)
    for block in (win.get("text") or "").split("\n\n"):
        m = re.match(r"^\[([^\]]+)\] (.*)$", block, re.S)
        if not m:
            continue
        pid, body = m.group(1), m.group(2)
        if body.startswith("(context"):
            continue
        idx = doc.text.find(body, cursor)
        if idx < 0:
            idx = doc.text.find(body, win.get("start", 0))
        if idx < 0:
            continue
        by_id[pid] = (idx, idx + len(body))
        cursor = idx + len(body)
    out: list[tuple[str, int, int]] = []
    for pid in owned:
        if pid in by_id:
            out.append((pid, by_id[pid][0], by_id[pid][1]))
            continue
        p = doc.para(pid) or doc.para(pid.split(".")[0])
        if p:
            out.append((pid, p.start, p.end))
    return out


def _split_window(doc: Document, win: dict, suffix: str) -> tuple[dict, dict]:
    """Bisect a window's owned paragraphs into two child windows (no shared context)."""
    units = _owned_unit_spans(doc, win)
    mid = max(1, len(units) // 2)
    left_u, right_u = units[:mid], units[mid:]

    def _child(parts: list[tuple[str, int, int]], tag: str) -> dict:
        if not parts:
            return {
                "id": f"{win['id']}{tag}{suffix}",
                "start": win["start"], "end": win["start"],
                "para_ids": [], "context_para_ids": [],
                "section_path": win.get("section_path") or [],
                "pages": [], "text": "",
            }
        start, end = parts[0][1], parts[-1][2]
        lines = [f"[{pid}] {doc.text[a:b]}" for pid, a, b in parts]
        return {
            "id": f"{win['id']}{tag}{suffix}",
            "start": start,
            "end": end,
            "para_ids": [pid for pid, _, _ in parts],
            "context_para_ids": [],
            "section_path": win.get("section_path") or [],
            "pages": doc.pages_for_span(start, end),
            "text": "\n\n".join(lines),
        }

    return _child(left_u, "a"), _child(right_u, "b")


def _needs_split(win: dict, n_mentions: int) -> bool:
    owned = len(win.get("para_ids") or [])
    if owned < settings.v2_extract_split_min_paras:
        return False
    ratio = n_mentions / max(1, owned)
    return ratio < settings.v2_extract_min_mention_ratio


def _extract_one_window(
    doc: Document, win: dict, tracker: FrameTracker, doc_id: str,
    mention_seq: list[int], n_samples: int, chat_fn: Callable | None,
    depth: int = 0, window_index: int | None = None,
) -> tuple[list, list, list, list]:
    """Extract a window; if mention recall looks too low, bisect and retry."""
    stack = tracker.state()
    try:
        primary = extract_window(doc, win, stack, doc_id=doc_id, sample=0,
                                 temperature=0.0, chat_fn=chat_fn,
                                 window_index=window_index)
    except llm.ContentFilterError:
        buildlog.record(doc_id, "dropped_window_content_filter", ref=win["id"],
                        detail={"local_llm": bool(settings.local_llm_base_url)})
        return [], [], [], []
    except llm.OutputTruncatedError as exc:
        buildlog.record(doc_id, "dropped_window_output_truncated", ref=win["id"],
                        detail={"error": str(exc), "depth": depth})
        return [], [], [], []
    except Exception as exc:
        buildlog.record(doc_id, "dropped_window", ref=win["id"],
                        detail={"error": f"{type(exc).__name__}: {exc}"})
        return [], [], [], []

    # Peek raw mention count (pre-resolution) for the recall guard
    raw_n = len(primary.mentions or [])
    if _needs_split(win, raw_n) and depth < 6:
        buildlog.record(
            doc_id, "extract_window_split", ref=win["id"],
            detail={"owned_paras": len(win.get("para_ids") or []),
                    "raw_mentions": raw_n, "depth": depth},
        )
        # Do not apply this under-filled extraction to the tracker; retry halves.
        left, right = _split_window(doc, win, suffix=f"_s{depth}")
        out_m, out_r, out_a, out_k = [], [], [], []
        for child in (left, right):
            if not child["para_ids"]:
                continue
            m, r, a, k = _extract_one_window(
                doc, child, tracker, doc_id, mention_seq, n_samples, chat_fn,
                depth + 1, window_index=window_index)
            out_m.extend(m); out_r.extend(r); out_a.extend(a); out_k.extend(k)
        return out_m, out_r, out_a, out_k

    mentions, relations, aliases, kinship, _ = _process_primary(
        doc, win, primary, tracker, doc_id, mention_seq)

    sample_extractions: list[tuple[WindowExtraction, dict]] = []
    for s in range(1, n_samples):
        try:
            sextr = extract_window(doc, win, stack, doc_id=doc_id, sample=s,
                                   temperature=0.4, chat_fn=chat_fn,
                                   window_index=window_index)
            sample_extractions.append((sextr, win))
        except Exception as exc:
            buildlog.record(doc_id, "failed_sample", ref=win["id"],
                            detail={"sample": s, "error": f"{type(exc).__name__}: {exc}"})
    _vote_relations(relations, mentions, sample_extractions, doc)
    return mentions, relations, aliases, kinship


def _dedupe_mentions_by_span(
    mentions: list[ExtractedMention],
) -> tuple[list[ExtractedMention], int]:
    """Keep the first mention for each (start, end) char span; drop later overlaps."""
    seen: set[tuple[int, int]] = set()
    out: list[ExtractedMention] = []
    dropped = 0
    for m in mentions:
        key = (m.start, m.end)
        if key in seen and not m.is_telling:
            dropped += 1
            continue
        # Telling mentions may share a span with content; keep distinct ids.
        if not m.is_telling:
            seen.add(key)
        out.append(m)
    return out, dropped


def extract_document(doc: Document, doc_id: str, *,
                     n_samples: int | None = None,
                     chat_fn: Callable | None = None,
                     on_progress: Callable[[float, str], None] | None = None,
                     ) -> ExtractionResult:
    """
    Process the whole document as ONE unit sequentially, carrying the frame stack
    across section boundaries (break_level=0).

    When settings.extractor == "oracle", use the deterministic synth regex path
    (same ExtractionResult schema; no LLM).
    """
    if (settings.extractor or "llm").lower() == "oracle":
        from .oracle import extract_document_oracle
        return extract_document_oracle(doc, doc_id, on_progress=on_progress)

    n_samples = n_samples if n_samples is not None else settings.v2_extract_samples
    wins = make_windows(
        doc, settings.v2_window_chars, settings.v2_window_overlap_paras,
        break_level=0, max_paras=settings.v2_window_max_paras,
    )
    unit_id = "U0"
    tracker = FrameTracker(unit_id=unit_id, unit_start=0, unit_end=len(doc.text))
    mention_seq = [0]
    all_mentions: list[ExtractedMention] = []
    all_relations: list[ExtractedRelation] = []
    all_aliases: list[ExtractedAlias] = []
    all_kinship: list[ExtractedKinship] = []

    for i, win in enumerate(wins):
        if on_progress:
            on_progress(i / max(1, len(wins)), f"extract {win['id']} ({i + 1}/{len(wins)})")
        mentions, relations, aliases, kinship = _extract_one_window(
            doc, win, tracker, doc_id, mention_seq, n_samples, chat_fn,
            window_index=i)
        all_mentions.extend(mentions)
        all_relations.extend(relations)
        all_aliases.extend(aliases)
        all_kinship.extend(kinship)

    frames = tracker.finish()
    for w in tracker.warnings:
        buildlog.record(doc_id, "frame_warning", detail=w)

    # Assign frames AFTER finish
    for m in all_mentions:
        m.frame_id = tracker.frame_at(m.start).id

    telling = _make_telling_mentions(doc, frames, mention_seq)
    all_mentions.extend(telling)

    all_mentions, n_dup = _dedupe_mentions_by_span(all_mentions)
    if n_dup:
        buildlog.record(doc_id, "extract_mention_dedupe",
                        detail={"dropped": n_dup, "kept": len(all_mentions)})
        keep_ids = {m.id for m in all_mentions}
        all_relations = [r for r in all_relations
                         if r.a in keep_ids and r.b in keep_ids]

    if on_progress:
        on_progress(1.0, "extract complete")
    return ExtractionResult(
        frames=frames, mentions=all_mentions, relations=all_relations,
        aliases=all_aliases, kinship=all_kinship, windows=wins,
        prompt_version=PROMPT_VERSION, extract_samples=n_samples,
    )
