import json
import traceback
import uuid

from .db import pg


def create(doc_id: str, kind: str) -> str:
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    with pg() as cur:
        cur.execute(
            """INSERT INTO jobs (id, doc_id, kind, status, stage, progress)
               VALUES (%s,%s,%s,'queued','',0)""",
            (job_id, doc_id, kind),
        )
    return job_id


def update(job_id: str, *, status=None, stage=None, progress=None,
           detail=None, error=None) -> None:
    sets, args = [], []
    if status is not None:
        sets.append("status = %s"); args.append(status)
    if stage is not None:
        sets.append("stage = %s"); args.append(stage)
    if progress is not None:
        sets.append("progress = %s"); args.append(float(progress))
    if detail is not None:
        sets.append("detail = %s"); args.append(json.dumps(detail))
    if error is not None:
        sets.append("error = %s"); args.append(error)
    if status in ("done", "error"):
        sets.append("finished_at = now()")
    if not sets:
        return
    args.append(job_id)
    with pg() as cur:
        cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = %s", args)


def get(job_id: str) -> dict | None:
    with pg() as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def latest(doc_id: str, kind: str) -> dict | None:
    with pg() as cur:
        cur.execute(
            """SELECT * FROM jobs WHERE doc_id = %s AND kind = %s
               ORDER BY started_at DESC LIMIT 1""",
            (doc_id, kind),
        )
        row = cur.fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------
# Runners (executed in a FastAPI BackgroundTask thread)
# ------------------------------------------------------------
def run_naive(job_id: str, doc_id: str) -> None:
    from . import docstore, naive_rag
    try:
        update(job_id, status="running", stage="chunk+embed", progress=0.02)
        doc = docstore.load_document(doc_id)
        count = naive_rag.build(
            doc_id, doc,
            on_progress=lambda p, msg: update(job_id, progress=p * 0.98 + 0.02, stage=msg),
        )
        update(job_id, status="done", stage="complete", progress=1.0,
               detail={"chunks": count})
    except Exception as exc:
        traceback.print_exc()
        update(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


def _doc_title(doc_id: str) -> str:
    with pg() as cur:
        cur.execute("SELECT title FROM documents WHERE id = %s", (doc_id,))
        row = cur.fetchone()
    return (row["title"] if row else "") or doc_id


def run_kaalkram(job_id: str, doc_id: str) -> None:
    from . import docstore, graph, passes
    from .config import settings
    from .ingest import extract_windows_for_doc
    try:
        doc = docstore.load_document(doc_id)
        windows = extract_windows_for_doc(doc)
        title = _doc_title(doc_id)

        # ---- Pass 1: 0.00 -> 0.42
        update(job_id, status="running", stage="pass 1: reading windows", progress=0.01)
        obs = passes.run_pass1(
            doc_id, windows,
            on_progress=lambda p, msg: update(job_id, progress=0.01 + p * 0.41, stage=msg),
        )

        # ---- Pass 0: 0.42 -> 0.48 (book-specific taxonomy)
        update(job_id, stage="pass 0: inventing stage taxonomy", progress=0.42)
        taxonomy = passes.run_pass0(
            doc_id, title, obs,
            on_progress=lambda p, msg: update(job_id, progress=0.42 + p * 0.06, stage=msg),
        )

        # ---- Pass 2: 0.48 -> 0.80
        update(job_id, stage="pass 2: merging duplicates", progress=0.48)
        events = passes.run_pass2(
            doc_id, obs, taxonomy=taxonomy,
            on_progress=lambda p, msg: update(job_id, progress=0.48 + p * 0.32, stage=msg),
        )

        # ---- Pass 3 + persist: 0.80 -> 0.93
        update(job_id, stage="pass 3: ordering the story", progress=0.80)
        events = passes.run_pass3(events, taxonomy=taxonomy)
        passes.persist_events(
            doc_id, events,
            on_progress=lambda p, msg: update(job_id, progress=0.82 + p * 0.11, stage=msg),
        )

        # ---- Graph: 0.93 -> 1.00
        update(job_id, stage="building event graph", progress=0.94)
        edges, stats = graph.build_edges(events)
        graph.push(doc_id, events, edges)

        from . import buildlog
        buildlog.flush_llm_events(doc_id, job_id=job_id)
        majors = sum(1 for e in events if e["category"] == "major")
        merges = sum(e["merge_count"] - 1 for e in events)
        update(job_id, status="done", stage="complete", progress=1.0, detail={
            "events": len(events),
            "major": majors,
            "minor": len(events) - majors,
            "merges": merges,
            "windows": len(windows),
            "extraction_mode": settings.extraction_mode,
            "taxonomy": [s["name"] for s in taxonomy],
            **stats,
        })
    except Exception as exc:
        traceback.print_exc()
        update(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


def _to_constraint_mentions(mentions):
    """Map extracted mentions (with entity_id + event_id) to constraints.Mention."""
    from .v2.constraints import Mention
    out = []
    for m in mentions:
        parts = []
        for p in m.participants:
            eid = p.get("entity_id")
            if eid:
                parts.append((eid, p.get("role") or "present"))
        subject = m.subject if (m.subject or "").startswith("ent_") else None
        out.append(Mention(
            id=m.id, event_id=m.event_id, frame_id=m.frame_id,
            start=m.start, end=m.end, mode=m.mode, event_type=m.event_type,
            subject=subject, participants=parts, posthumous=m.posthumous,
            is_telling=m.is_telling, tells_frame=m.tells_frame,
        ))
    return out


def run_kaalkram_v2(job_id: str, doc_id: str) -> None:
    from . import buildlog, docstore
    from .config import settings
    from .v2 import constraints, coref, entities, extract, persist
    from .v2.constraints import Kinship, LocalRelation, SOURCE_PRIOR
    from .v2.solver import TemporalGraph
    try:
        update(job_id, status="running", stage="load document", progress=0.01)
        doc = docstore.load_document(doc_id)

        update(job_id, stage="extract windows", progress=0.05)
        extr = extract.extract_document(
            doc, doc_id,
            on_progress=lambda p, msg: update(job_id, progress=0.05 + p * 0.40, stage=msg),
        )

        update(job_id, stage="entity resolution", progress=0.46)
        ent = entities.resolve(
            doc.text, extr.mentions, aliases=extr.aliases, kinship=extr.kinship,
            doc_id=doc_id,
        )

        update(job_id, stage="event coreference", progress=0.58)
        cof = coref.resolve(ent.mentions, doc_text=doc.text, doc_id=doc_id)

        update(job_id, stage="constraints + graph", progress=0.70)
        weights = persist.load_weights(doc_id) or dict(SOURCE_PRIOR)
        cm = _to_constraint_mentions(cof.mentions)
        rels = [
            LocalRelation(a=r.a, b=r.b, rel=r.rel, cue=r.cue, consistency=r.consistency,
                          span=(r.start, r.end) if r.start is not None else None)
            for r in extr.relations
        ]
        kin = [
            Kinship(parent=k.parent, child=k.child,
                    span=(k.start, k.end) if k.start is not None else None)
            for k in ent.kinship
        ]
        cons = constraints.all_constraints(extr.frames, cm, rels, kin, weights)
        event_ids = [e.id for e in cof.events]
        discourse = {e.id: e.first_offset for e in cof.events}
        g = TemporalGraph(event_ids, discourse)
        for c in cons:
            g.add(c)
        repair_stats = g.repair()

        update(job_id, stage="persist + embed", progress=0.85)
        first_quotes: dict[str, str] = {}
        by_event: dict[str, list] = {}
        for m in cof.mentions:
            if m.event_id:
                by_event.setdefault(m.event_id, []).append(m)
        for eid, ms in by_event.items():
            ms_sorted = sorted(ms, key=lambda x: x.start)
            first_quotes[eid] = ms_sorted[0].quote if ms_sorted else ""

        persist.persist(
            doc_id, frames=extr.frames, entities=ent.entities, events=cof.events,
            mentions=cof.mentions, relations=extr.relations, graph=g,
            repair_stats=repair_stats, weights=weights,
            prompt_version=extr.prompt_version, first_quotes=first_quotes,
            kinship=kin,
        )

        buildlog.flush_llm_events(doc_id, job_id=job_id)
        frames_by_type: dict[str, int] = {}
        for f in extr.frames:
            frames_by_type[f.type] = frames_by_type.get(f.type, 0) + 1
        mentions_by_mode: dict[str, int] = {}
        for m in cof.mentions:
            mentions_by_mode[m.mode] = mentions_by_mode.get(m.mode, 0) + 1
        cons_by_source: dict[str, int] = {}
        for c in cons:
            cons_by_source[c.source] = cons_by_source.get(c.source, 0) + 1
        removed_by_source: dict[str, int] = {}
        removed_weight = 0.0
        for rem in g.removed:
            info = rem.get("removed") or rem
            for src in info.get("sources") or ["?"]:
                removed_by_source[src] = removed_by_source.get(src, 0) + 1
            removed_weight += float(info.get("p") or 0.0)

        update(job_id, status="done", stage="complete", progress=1.0, detail={
            "windows": len(extr.windows),
            "units": 1,
            "frames_by_type": frames_by_type,
            "mentions_by_mode": mentions_by_mode,
            "events": len(cof.events),
            "entities": len(ent.entities),
            "constraints_by_source": cons_by_source,
            "removed_edges_by_source": removed_by_source,
            "removed_weight": removed_weight,
            "build_events": buildlog.summary(doc_id),
            "prompt_version": extr.prompt_version,
            "extraction_mode": settings.extraction_mode,
            "extractor": settings.extractor,
            "extract_samples": extr.extract_samples,
            "repair": repair_stats,
            "graph": g.stats(),
        })
    except Exception as exc:
        traceback.print_exc()
        update(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
