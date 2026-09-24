"""
Constraint extractors. Each turns extracted facts into solver Constraints with
evidence. All rules are generic narrative/temporal logic — no book knowledge.

  explicit / time_expression / causal   relations the LLM read in the text
  succession   consecutive first-time "occurs" events in the SAME frame are in order
               ("occurs" = narrated as happening at this point of the frame's narration;
                "recounted" = narrated as earlier than its surroundings, e.g. pluperfect)
               (a default; it stops at every frame boundary, i.e. every flashback)
  frame        recalled/told events precede their telling; predicted events that
               later occur follow the prediction
  lifecycle    birth precedes, death follows, everything a person actively does
  genealogy    a parent's birth precedes the child's

Weights: each source starts from SOURCE_PRIOR and should be replaced by
calibrate_sources() on the dev split. LLM relations additionally carry their
self-consistency (fraction of extraction samples that produced them).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .frames import PAST_FRAMES, Frame
from .solver import Constraint

# Initial source weights. NOT final: overwrite with calibrate_sources() output
# and report the calibrated values alongside results.
SOURCE_PRIOR = {
    "explicit": 0.90, "time_expression": 0.85, "frame": 0.85, "causal": 0.75,
    "lifecycle": 0.70, "genealogy": 0.70, "succession": 0.60,
}
CUE_TO_SOURCE = {"explicit_connective": "explicit", "time_expression": "time_expression",
                 "causal": "causal", "narrative_order": "succession"}
ACTIVE_ROLES = ("agent", "patient", "present")


@dataclass
class Mention:
    id: str
    event_id: str                 # after coreference
    frame_id: str
    start: int
    end: int
    mode: str                     # occurs | recounted | predicted | hypothetical | habitual
    event_type: str = "other"     # birth | death | marriage | other
    subject: str | None = None    # canonical entity id for birth/death
    participants: list[tuple[str, str]] = field(default_factory=list)  # (entity_id, role)
    posthumous: bool = False
    is_telling: bool = False      # the act of narrating a frame (created by code)
    tells_frame: str | None = None


@dataclass
class LocalRelation:
    a: str                        # mention ids
    b: str
    rel: str                      # before | after | during | contains | simultaneous | overlap
    cue: str                      # explicit_connective | time_expression | causal | narrative_order
    consistency: float = 1.0      # fraction of extraction samples agreeing
    span: tuple[int, int] | None = None


@dataclass
class Kinship:
    parent: str                   # entity ids
    child: str
    span: tuple[int, int] | None = None


def _ev(m: Mention) -> dict:
    return {"span": [m.start, m.end], "mention": m.id}


def _w(source: str, weights: dict | None) -> float:
    return (weights or SOURCE_PRIOR).get(source, SOURCE_PRIOR.get(source, 0.5))


def from_local_relations(rels: list[LocalRelation], mentions: dict[str, Mention],
                         weights: dict | None = None) -> list[Constraint]:
    out = []
    for r in rels:
        ma, mb = mentions.get(r.a), mentions.get(r.b)
        if not ma or not mb or ma.event_id == mb.event_id:
            continue
        if "hypothetical" in (ma.mode, mb.mode):
            continue
        src = CUE_TO_SOURCE.get(r.cue, "explicit")
        ev = [_ev(ma), _ev(mb)] + ([{"span": list(r.span), "note": r.cue}] if r.span else [])
        out.append(Constraint(ma.event_id, mb.event_id, r.rel, _w(src, weights) * r.consistency,
                              src, ev, id=f"rel:{r.a}>{r.b}"))
    return out


def succession(mentions: list[Mention], weights: dict | None = None) -> list[Constraint]:
    first_occurs: dict[str, int] = {}
    for m in sorted(mentions, key=lambda m: m.start):
        if m.mode in ("occurs", "recounted"):
            first_occurs.setdefault(m.event_id, m.start)
    by_frame: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions:
        # only mentions narrated "as happening now" in their frame; an in-frame
        # flashback (mode recounted) gets its order from explicit relations instead
        if m.mode == "occurs" and first_occurs.get(m.event_id) == m.start:
            by_frame[m.frame_id].append(m)
    out = []
    for fid, ms in by_frame.items():
        ms.sort(key=lambda m: m.start)
        for a, b in zip(ms, ms[1:]):
            if a.event_id != b.event_id:
                out.append(Constraint(a.event_id, b.event_id, "before", _w("succession", weights),
                                      "succession", [_ev(a), _ev(b)],
                                      id=f"succ:{a.id}>{b.id}"))
    return out


def frame_constraints(frames: list[Frame], mentions: list[Mention],
                      weights: dict | None = None) -> list[Constraint]:
    telling = {m.tells_frame: m for m in mentions if m.is_telling and m.tells_frame}
    occurs_outside_prediction = {m.event_id for m in mentions
                                 if m.mode in ("occurs", "recounted") and not m.is_telling}
    ftype = {f.id: f.type for f in frames}
    out = []
    for m in mentions:
        if m.is_telling:
            continue
        t = telling.get(m.frame_id)
        if t is None or t.event_id == m.event_id:
            continue
        kind = ftype.get(m.frame_id)
        if kind in PAST_FRAMES and m.mode in ("occurs", "recounted"):
            out.append(Constraint(m.event_id, t.event_id, "before", _w("frame", weights), "frame",
                                  [_ev(m), _ev(t)], id=f"frame:{m.id}<{t.id}"))
        elif kind == "prediction" and m.event_id in occurs_outside_prediction:
            out.append(Constraint(t.event_id, m.event_id, "before", _w("frame", weights), "frame",
                                  [_ev(t), _ev(m)], id=f"frame:{t.id}<{m.id}"))
    return out


def lifecycle(mentions: list[Mention], weights: dict | None = None) -> list[Constraint]:
    births: dict[str, Mention] = {}
    deaths: dict[str, Mention] = {}
    for m in sorted(mentions, key=lambda m: m.start):
        if m.subject and m.mode in ("occurs", "recounted"):
            if m.event_type == "birth":
                births.setdefault(m.subject, m)
            elif m.event_type == "death":
                deaths.setdefault(m.subject, m)
    acts: dict[str, dict[str, Mention]] = defaultdict(dict)     # entity -> event -> mention
    for m in mentions:
        if m.mode not in ("occurs", "recounted") or m.is_telling:
            continue
        for ent, role in m.participants:
            if role in ACTIVE_ROLES:
                acts[ent].setdefault(m.event_id, m)
    out = []
    for ent, events in acts.items():
        b, d = births.get(ent), deaths.get(ent)
        for ev_id, m in events.items():
            if b and ev_id != b.event_id and not (m.event_type == "birth" and m.subject == ent):
                out.append(Constraint(b.event_id, ev_id, "before", _w("lifecycle", weights),
                                      "lifecycle", [_ev(b), _ev(m)], id=f"birth:{ent}<{ev_id}"))
            if d and ev_id != d.event_id and not m.posthumous:
                out.append(Constraint(ev_id, d.event_id, "before", _w("lifecycle", weights),
                                      "lifecycle", [_ev(m), _ev(d)], id=f"death:{ent}>{ev_id}"))
    return out


def genealogy(kin: list[Kinship], mentions: list[Mention],
              weights: dict | None = None) -> list[Constraint]:
    births = {}
    for m in sorted(mentions, key=lambda m: m.start):
        if m.event_type == "birth" and m.subject and m.mode in ("occurs", "recounted"):
            births.setdefault(m.subject, m)
    out = []
    for k in kin:
        bp, bc = births.get(k.parent), births.get(k.child)
        if bp and bc and bp.event_id != bc.event_id:
            ev = [_ev(bp), _ev(bc)] + ([{"span": list(k.span), "note": "kinship"}] if k.span else [])
            out.append(Constraint(bp.event_id, bc.event_id, "before", _w("genealogy", weights),
                                  "genealogy", ev, id=f"gen:{k.parent}>{k.child}"))
    return out


def all_constraints(frames, mentions, rels, kin, weights=None) -> list[Constraint]:
    mdict = {m.id: m for m in mentions}
    return (from_local_relations(rels, mdict, weights) + succession(mentions, weights)
            + frame_constraints(frames, mentions, weights) + lifecycle(mentions, weights)
            + genealogy(kin, mentions, weights))


def calibrate_sources(constraints: list[Constraint],
                      gold_pairs: list[tuple[str, str, str]],
                      alpha: float = 1.0, beta: float = 1.0) -> dict:
    """Estimate each source's precision from gold (event_a, event_b, before|after)
    pairs on the DEV split. Beta(alpha, beta) smoothing. Only constraints whose two
    events are both a gold pair are counted. Returns {source: {weight, n, agree}}."""
    gold = {}
    for a, b, lab in gold_pairs:
        if lab == "before":
            gold[(a, b)] = True
            gold[(b, a)] = False
        elif lab == "after":
            gold[(b, a)] = True
            gold[(a, b)] = False
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c in constraints:
        if c.rel not in ("before", "after"):
            continue
        a, b = (c.a, c.b) if c.rel == "before" else (c.b, c.a)
        if (a, b) in gold:
            tally[c.source][1] += 1
            tally[c.source][0] += gold[(a, b)]
    return {s: {"weight": (agree + alpha) / (n + alpha + beta), "n": n, "agree": agree}
            for s, (agree, n) in tally.items()}
