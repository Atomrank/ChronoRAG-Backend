"""Event coreference for Kaalkram v2: mentions -> event ids."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import buildlog, llm
from ..config import settings
from .extract import ExtractedMention
from .prompts import COREF_SYSTEM
from .schemas import CorefDecision


@dataclass
class ResolvedEvent:
    id: str
    description: str
    mention_ids: list[str] = field(default_factory=list)
    first_offset: int = 0
    participants: list[str] = field(default_factory=list)  # entity ids
    event_type: str = "other"
    embedding: list[float] | None = None


@dataclass
class CorefResult:
    events: list[ResolvedEvent]
    mentions: list[ExtractedMention]  # with event_id set


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _entity_ids(m: ExtractedMention) -> set[str]:
    out: set[str] = set()
    for p in m.participants:
        eid = p.get("entity_id")
        if eid:
            out.add(eid)
    if m.subject and m.subject.startswith("ent_"):
        out.add(m.subject)
    return out


def _hard_block(a: ExtractedMention, b: ExtractedMention) -> str | None:
    """Return a reason string if these two must never merge; else None.

    Same-window pairs are allowed only when the LLM says so (not hard-blocked here).
    """
    modes = {a.mode, b.mode}
    if "hypothetical" in modes and modes != {"hypothetical"}:
        return "hypothetical_with_non_hypothetical"
    if a.is_telling and b.is_telling:
        return "two_telling"
    return None


def _passage(doc_text: str, m: ExtractedMention, radius: int) -> str:
    lo = max(0, m.start - radius)
    hi = min(len(doc_text), m.end + radius)
    return doc_text[lo:hi]


def _event_description(members: list[ExtractedMention]) -> str:
    members = sorted(members, key=lambda m: m.start)
    for m in members:
        if m.mode in ("occurs", "recounted"):
            return m.description
    return members[0].description if members else ""


def resolve(
    mentions: list[ExtractedMention],
    doc_text: str = "",
    *,
    doc_id: str = "",
    embed_fn: Callable | None = None,
    chat_fn: Callable | None = None,
) -> CorefResult:
    """
    Cluster mentions into events via candidate filtering + LLM CorefDecision.

    Parameters
    ----------
    mentions : entity-resolved mentions (participants should carry `entity_id`)
    doc_text : full document text for ±passage windows around each mention
    embed_fn : optional `list[str] -> list[list[float]]` (defaults to llm.embed)
    chat_fn : optional `(system, user, model_cls) -> CorefDecision`
              (defaults to llm.chat_structured at temperature 0)

    Returns
    -------
    CorefResult with events (`ev_<n>`) and mentions with `event_id` filled.
    Event description = earliest occurs/recounted mention, else the first by offset.
    """
    if not mentions:
        return CorefResult(events=[], mentions=[])

    embed_fn = embed_fn or llm.embed

    def _chat(system: str, user: str, model_cls):
        if chat_fn:
            return chat_fn(system, user, model_cls)
        return llm.chat_structured(system, user, model_cls, temperature=0.0)

    # Stable order by document offset.
    ordered = sorted(mentions, key=lambda m: (m.start, m.id))
    n = len(ordered)
    uf = _UF(n)

    try:
        emb = np.asarray(embed_fn([m.description for m in ordered]), dtype=np.float64)
    except Exception as exc:
        buildlog.record(doc_id, "failed_coref_embed",
                        detail={"error": f"{type(exc).__name__}: {exc}"})
        raise

    thr = settings.v2_coref_cosine
    top_k = settings.v2_coref_top
    radius = settings.v2_coref_passage_chars
    buildlog.record(doc_id, "coref_cosine_threshold", detail={"v2_coref_cosine": thr})

    entity_sets = [_entity_ids(m) for m in ordered]

    for i in range(n):
        mi = ordered[i]
        cands: list[tuple[float, int]] = []
        for j in range(i):
            if not (entity_sets[i] & entity_sets[j]):
                continue
            score = _cosine(emb[i], emb[j])
            if score >= thr:
                cands.append((score, j))
        cands.sort(reverse=True)
        cands = cands[:top_k]

        for score, j in cands:
            if uf.find(i) == uf.find(j):
                continue
            mj = ordered[j]
            block = _hard_block(mi, mj)
            if block:
                buildlog.record(doc_id, "coref_hard_block",
                                detail={"a": mi.id, "b": mj.id, "reason": block})
                continue

            user = (
                f"PASSAGE A (mention {mj.id}, mode={mj.mode}, "
                f"window={mj.window_id}, is_telling={mj.is_telling}):\n"
                f"{_passage(doc_text, mj, radius)}\n\n"
                f"DESCRIPTION A: {mj.description}\n\n"
                f"PASSAGE B (mention {mi.id}, mode={mi.mode}, "
                f"window={mi.window_id}, is_telling={mi.is_telling}):\n"
                f"{_passage(doc_text, mi, radius)}\n\n"
                f"DESCRIPTION B: {mi.description}\n"
            )
            if len(user) > settings.llm_max_input_chars:
                buildlog.record(doc_id, "coref_input_too_long",
                                detail={"a": mi.id, "b": mj.id, "chars": len(user)})
                continue
            try:
                decision: CorefDecision = _chat(COREF_SYSTEM, user, CorefDecision)
            except Exception as exc:
                buildlog.record(doc_id, "failed_coref_pair",
                                detail={"a": mi.id, "b": mj.id,
                                        "error": f"{type(exc).__name__}: {exc}"})
                continue

            # Same-window: only merge if the LLM explicitly says so (already required).
            if decision.same_event:
                uf.union(i, j)

    # Build events from components.
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)

    # Order event ids by earliest member offset.
    roots = sorted(comps.keys(), key=lambda r: min(ordered[i].start for i in comps[r]))
    events: list[ResolvedEvent] = []
    id_map: dict[str, str] = {}  # mention_id -> event_id

    for n_ev, root in enumerate(roots, start=1):
        members = [ordered[i] for i in sorted(comps[root], key=lambda i: ordered[i].start)]
        eid = f"ev_{n_ev}"
        parts: list[str] = []
        seen: set[str] = set()
        for m in members:
            for e in _entity_ids(m):
                if e not in seen:
                    seen.add(e)
                    parts.append(e)
            id_map[m.id] = eid
        events.append(ResolvedEvent(
            id=eid,
            description=_event_description(members),
            mention_ids=[m.id for m in members],
            first_offset=members[0].start,
            participants=parts,
            event_type=next((m.event_type for m in members
                             if m.event_type and m.event_type != "other"),
                            members[0].event_type),
        ))

    out_mentions: list[ExtractedMention] = []
    for m in mentions:
        out_mentions.append(ExtractedMention(
            id=m.id, local_id=m.local_id, window_id=m.window_id, frame_id=m.frame_id,
            start=m.start, end=m.end, para_id=m.para_id, quote=m.quote,
            description=m.description, mode=m.mode, event_type=m.event_type,
            subject=m.subject, participants=list(m.participants),
            location=m.location, time_expressions=list(m.time_expressions),
            posthumous=m.posthumous, is_telling=m.is_telling,
            tells_frame=m.tells_frame, sample_idx=m.sample_idx,
            event_id=id_map.get(m.id, ""),
        ))

    return CorefResult(events=events, mentions=out_mentions)
