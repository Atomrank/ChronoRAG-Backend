"""Persist Kaalkram v2 tables for a document (clear + insert)."""
from __future__ import annotations

import json

from pgvector.psycopg import Vector

from .. import llm
from ..db import pg
from .constraints import Kinship, LocalRelation, Mention, SOURCE_PRIOR, all_constraints
from .coref import ResolvedEvent
from .entities import ResolvedEntity
from .extract import ExtractedMention, ExtractedRelation
from .frames import Frame
from .solver import TemporalGraph


def _clear(doc_id: str) -> None:
    with pg() as cur:
        for table in ("v2_relations", "v2_mentions", "v2_events", "v2_entities",
                      "v2_frames", "v2_graphs"):
            cur.execute(f"DELETE FROM {table} WHERE doc_id = %s", (doc_id,))


def load_weights(doc_id: str) -> dict | None:
    """Return previously calibrated source weights, or None."""
    with pg() as cur:
        cur.execute("SELECT weights FROM v2_graphs WHERE doc_id = %s", (doc_id,))
        row = cur.fetchone()
    if not row or not row["weights"]:
        return None
    w = row["weights"]
    if isinstance(w, str):
        w = json.loads(w)
    # calibrate_sources returns {source: {weight, n, agree}}; flatten if needed
    out = {}
    for k, v in (w or {}).items():
        if isinstance(v, dict) and "weight" in v:
            out[k] = float(v["weight"])
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out or None


def _kinship_to_json(kinship: list | None) -> list[dict]:
    out = []
    for k in kinship or []:
        if isinstance(k, Kinship):
            out.append({
                "parent": k.parent, "child": k.child,
                "span": list(k.span) if k.span else None,
            })
        elif isinstance(k, dict):
            out.append(k)
        else:
            span = None
            if getattr(k, "start", None) is not None and getattr(k, "end", None) is not None:
                span = [k.start, k.end]
            out.append({"parent": k.parent, "child": k.child, "span": span})
    return out


def persist(
    doc_id: str,
    *,
    frames: list[Frame],
    entities: list[ResolvedEntity],
    events: list[ResolvedEvent],
    mentions: list[ExtractedMention],
    relations: list[ExtractedRelation],
    graph: TemporalGraph,
    repair_stats: dict,
    weights: dict,
    prompt_version: str,
    first_quotes: dict[str, str] | None = None,
    kinship: list | None = None,
) -> None:
    """Replace all v2 rows for doc_id. Embeds events as description + first quote."""
    first_quotes = first_quotes or {}
    _clear(doc_id)

    with pg() as cur:
        cur.executemany(
            """INSERT INTO v2_frames
               (id, doc_id, unit_id, type, parent, narrator, listener,
                open_at, close_at, depth, summary, auto_closed)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(f.id, doc_id, f.unit_id, f.type, f.parent, f.narrator, f.listener,
              f.open_at, f.close_at, f.depth, f.summary, f.auto_closed)
             for f in frames],
        )
        cur.executemany(
            """INSERT INTO v2_entities (id, doc_id, canonical, surfaces)
               VALUES (%s,%s,%s,%s)""",
            [(e.id, doc_id, e.canonical, json.dumps(e.surfaces)) for e in entities],
        )

    # Embed events
    texts = []
    for ev in events:
        q = first_quotes.get(ev.id, "")
        texts.append(f"{ev.description} | {q}" if q else ev.description)
    vectors: list[list[float]] = []
    if texts:
        vectors = llm.embed(texts)

    with pg() as cur:
        rows = []
        for i, ev in enumerate(events):
            emb = Vector(vectors[i]) if i < len(vectors) else None
            rows.append((
                ev.id, doc_id, ev.description, ev.event_type, ev.first_offset,
                json.dumps(ev.participants), emb, None,
            ))
        cur.executemany(
            """INSERT INTO v2_events
               (id, doc_id, description, event_type, first_offset, participants,
                embedding, partition_frame)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            rows,
        )
        cur.executemany(
            """INSERT INTO v2_mentions
               (id, doc_id, event_id, frame_id, window_id, char_start, char_end,
                para_id, quote, description, mode, event_type, subject_entity,
                participants, location, time_expressions, posthumous, is_telling,
                tells_frame, sample_idx)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(
                m.id, doc_id, m.event_id or None, m.frame_id, m.window_id,
                m.start, m.end, m.para_id, m.quote, m.description, m.mode,
                m.event_type, m.subject if m.subject.startswith("ent_") else None,
                json.dumps(m.participants), m.location,
                json.dumps(m.time_expressions), m.posthumous, m.is_telling,
                m.tells_frame, m.sample_idx,
            ) for m in mentions],
        )
        cur.executemany(
            """INSERT INTO v2_relations
               (doc_id, mention_a, mention_b, rel, cue, consistency, quote,
                char_start, char_end)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(
                doc_id, r.a, r.b, r.rel, r.cue, r.consistency, r.quote,
                r.start, r.end,
            ) for r in relations],
        )
        stats = {
            "repair": repair_stats,
            "graph": graph.stats(),
            "kinship": _kinship_to_json(kinship),
        }
        cur.execute(
            """INSERT INTO v2_graphs (doc_id, graph, stats, weights, prompt_version)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (doc_id) DO UPDATE SET
                 graph = EXCLUDED.graph, stats = EXCLUDED.stats,
                 weights = EXCLUDED.weights, prompt_version = EXCLUDED.prompt_version,
                 created_at = now()""",
            (doc_id, json.dumps(graph.to_dict()),
             json.dumps(stats),
             json.dumps(weights), prompt_version),
        )


def _row_participants(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, str):
        return json.loads(raw)
    return list(raw)


def load_mentions(doc_id: str) -> list[ExtractedMention]:
    with pg() as cur:
        cur.execute(
            """SELECT id, event_id, frame_id, window_id, char_start, char_end, para_id,
                      quote, description, mode, event_type, subject_entity, participants,
                      location, time_expressions, posthumous, is_telling, tells_frame,
                      sample_idx
               FROM v2_mentions WHERE doc_id = %s ORDER BY char_start""",
            (doc_id,),
        )
        rows = cur.fetchall()
    out = []
    for r in rows:
        parts = _row_participants(r["participants"])
        subject = r["subject_entity"] or ""
        out.append(ExtractedMention(
            id=r["id"], local_id=r["id"], window_id=r["window_id"] or "",
            frame_id=r["frame_id"] or "", start=r["char_start"], end=r["char_end"],
            para_id=r["para_id"] or "", quote=r["quote"] or "",
            description=r["description"] or "", mode=r["mode"] or "occurs",
            event_type=r["event_type"] or "other", subject=subject,
            participants=parts, location=r["location"] or "",
            time_expressions=_row_participants(r["time_expressions"]),
            posthumous=bool(r["posthumous"]), is_telling=bool(r["is_telling"]),
            tells_frame=r["tells_frame"], sample_idx=r["sample_idx"] or 0,
            event_id=r["event_id"] or "",
        ))
    return out


def load_frames(doc_id: str) -> list[Frame]:
    with pg() as cur:
        cur.execute(
            """SELECT id, unit_id, type, parent, narrator, listener, open_at, close_at,
                      depth, summary, auto_closed
               FROM v2_frames WHERE doc_id = %s ORDER BY open_at""",
            (doc_id,),
        )
        rows = cur.fetchall()
    return [
        Frame(
            id=r["id"], unit_id=r["unit_id"], type=r["type"], parent=r["parent"],
            narrator=r["narrator"] or "", listener=r["listener"] or "",
            open_at=r["open_at"], close_at=r["close_at"], depth=r["depth"] or 0,
            summary=r["summary"] or "", auto_closed=bool(r["auto_closed"]),
        )
        for r in rows
    ]


def load_relations(doc_id: str) -> list[ExtractedRelation]:
    with pg() as cur:
        cur.execute(
            """SELECT mention_a, mention_b, rel, cue, consistency, quote, char_start, char_end
               FROM v2_relations WHERE doc_id = %s""",
            (doc_id,),
        )
        rows = cur.fetchall()
    return [
        ExtractedRelation(
            a=r["mention_a"], b=r["mention_b"], rel=r["rel"], cue=r["cue"],
            consistency=float(r["consistency"] if r["consistency"] is not None else 1.0),
            quote=r["quote"] or "", start=r["char_start"], end=r["char_end"],
        )
        for r in rows
    ]


def load_events(doc_id: str) -> list[ResolvedEvent]:
    with pg() as cur:
        cur.execute(
            """SELECT id, description, event_type, first_offset, participants
               FROM v2_events WHERE doc_id = %s ORDER BY first_offset""",
            (doc_id,),
        )
        rows = cur.fetchall()
    out = []
    for r in rows:
        parts = _row_participants(r["participants"])
        ids = []
        for p in parts:
            if isinstance(p, str):
                ids.append(p)
            elif isinstance(p, dict) and p.get("entity_id"):
                ids.append(p["entity_id"])
        out.append(ResolvedEvent(
            id=r["id"], description=r["description"] or "",
            event_type=r["event_type"] or "other",
            first_offset=r["first_offset"] or 0, participants=ids,
        ))
    return out


def load_graph(doc_id: str) -> dict | None:
    with pg() as cur:
        cur.execute(
            "SELECT graph, stats, weights, prompt_version, created_at FROM v2_graphs WHERE doc_id = %s",
            (doc_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    g = row["graph"]
    if isinstance(g, str):
        g = json.loads(g)
    stats = row["stats"]
    if isinstance(stats, str):
        stats = json.loads(stats)
    weights = row["weights"]
    if isinstance(weights, str):
        weights = json.loads(weights)
    return {
        "graph": g, "stats": stats, "weights": weights,
        "prompt_version": row["prompt_version"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


def load_stored_kinship(doc_id: str) -> list[Kinship]:
    data = load_graph(doc_id)
    if not data:
        return []
    stats = data.get("stats") or {}
    raw = stats.get("kinship") or []
    out = []
    for k in raw:
        span = k.get("span")
        out.append(Kinship(
            parent=k["parent"], child=k["child"],
            span=tuple(span) if span and len(span) == 2 else None,
        ))
    return out


def _to_constraint_mentions(mentions: list[ExtractedMention]) -> list[Mention]:
    out = []
    for m in mentions:
        parts = []
        for p in m.participants:
            if isinstance(p, dict):
                eid = p.get("entity_id")
                if eid:
                    parts.append((eid, p.get("role") or "present"))
            elif isinstance(p, (list, tuple)) and len(p) >= 1:
                parts.append((p[0], p[1] if len(p) > 1 else "present"))
        subject = m.subject if (m.subject or "").startswith("ent_") else None
        out.append(Mention(
            id=m.id, event_id=m.event_id, frame_id=m.frame_id,
            start=m.start, end=m.end, mode=m.mode, event_type=m.event_type,
            subject=subject, participants=parts, posthumous=m.posthumous,
            is_telling=m.is_telling, tells_frame=m.tells_frame,
        ))
    return out


def rebuild_graph(doc_id: str, weights: dict | None = None) -> dict:
    """Rebuild TemporalGraph from persisted extraction (no re-extraction / LLM).

    Uses calibrated ``weights`` when given, else stored weights, else SOURCE_PRIOR.
    Updates ``v2_graphs`` only (mentions/events/frames untouched).
    """
    frames = load_frames(doc_id)
    mentions = load_mentions(doc_id)
    relations = load_relations(doc_id)
    events = load_events(doc_id)
    kinship = load_stored_kinship(doc_id)
    if not events:
        raise ValueError(f"no v2 events for {doc_id}; build kaalkram_v2 first")

    flat = weights if weights is not None else (load_weights(doc_id) or dict(SOURCE_PRIOR))
    # Flatten calibration records {src: {weight,n,agree}} -> {src: float}
    merged = dict(SOURCE_PRIOR)
    for k, v in flat.items():
        if isinstance(v, dict) and "weight" in v:
            merged[k] = float(v["weight"])
        elif isinstance(v, (int, float)):
            merged[k] = float(v)

    cm = _to_constraint_mentions(mentions)
    rels = [
        LocalRelation(
            a=r.a, b=r.b, rel=r.rel, cue=r.cue, consistency=r.consistency,
            span=(r.start, r.end) if r.start is not None else None,
        )
        for r in relations
    ]
    cons = all_constraints(frames, cm, rels, kinship, merged)
    event_ids = [e.id for e in events]
    discourse = {e.id: e.first_offset for e in events}
    g = TemporalGraph(event_ids, discourse)
    for c in cons:
        g.add(c)
    repair_stats = g.repair()

    existing = load_graph(doc_id) or {}
    prompt_version = existing.get("prompt_version") or ""
    prev_stats = existing.get("stats") or {}
    stats = {
        "repair": repair_stats,
        "graph": g.stats(),
        "kinship": prev_stats.get("kinship") or _kinship_to_json(kinship),
    }
    store_w = weights if weights is not None else merged
    with pg() as cur:
        cur.execute(
            """UPDATE v2_graphs
               SET graph = %s, stats = %s, weights = %s, created_at = now()
               WHERE doc_id = %s""",
            (json.dumps(g.to_dict()), json.dumps(stats), json.dumps(store_w), doc_id),
        )
        if cur.rowcount == 0:
            cur.execute(
                """INSERT INTO v2_graphs (doc_id, graph, stats, weights, prompt_version)
                   VALUES (%s,%s,%s,%s,%s)""",
                (doc_id, json.dumps(g.to_dict()), json.dumps(stats),
                 json.dumps(store_w), prompt_version),
            )
    return {
        "constraints": len(cons),
        "repair": repair_stats,
        "graph": g.stats(),
        "weights": store_w,
        "graph_obj": g,
        "constraints_list": cons,
    }


def load_timeline(doc_id: str) -> dict:
    data = load_graph(doc_id)
    if not data:
        return {"order": [], "note": "display order only", "error": "no graph"}
    g = TemporalGraph.from_dict(data["graph"])
    order = g.linear_extension()
    with pg() as cur:
        cur.execute(
            "SELECT id, description, first_offset FROM v2_events WHERE doc_id = %s",
            (doc_id,),
        )
        by_id = {r["id"]: dict(r) for r in cur.fetchall()}
    return {
        "note": "display order only — not a claimed total chronology",
        "order": [{"id": eid, **by_id.get(eid, {})} for eid in order],
    }
