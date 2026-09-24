"""Kaalkram v2 query engine: parse → ground → relation → verbalise."""
from __future__ import annotations

import json
import re
import time
from typing import Callable

from pgvector.psycopg import Vector

from .. import docstore, llm, naive_rag
from ..config import settings
from ..db import pg
from ..schemas import Citation, PipelineAnswer
from ..textutil import scrub_llm_text
from .persist import load_graph
from .prompts import GROUND_SYSTEM, QUESTION_SYSTEM, VERBALISE_SYSTEM
from .schemas import GroundingChoice, QuestionParse, VerbalisedAnswer
from .solver import TemporalGraph


def _events_for_entities(doc_id: str, entity_ids: list[str]) -> list[dict]:
    if not entity_ids:
        return []
    with pg() as cur:
        cur.execute(
            """SELECT id, description, event_type, first_offset, participants
               FROM v2_events WHERE doc_id = %s""",
            (doc_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    want = set(entity_ids)
    out = []
    for r in rows:
        parts = r.get("participants") or []
        if isinstance(parts, str):
            parts = json.loads(parts)
        if want & set(parts):
            out.append(r)
    return out


def _embed_events(doc_id: str, text: str, k: int) -> list[dict]:
    vec = llm.embed([text])[0]
    with pg() as cur:
        cur.execute(
            """SELECT id, description, event_type, first_offset, participants,
                      1 - (embedding <=> %s) AS score
               FROM v2_events WHERE doc_id = %s AND embedding IS NOT NULL
               ORDER BY embedding <=> %s LIMIT %s""",
            (Vector(vec), doc_id, Vector(vec), k),
        )
        return [dict(r) for r in cur.fetchall()]


def _match_entities(doc_id: str, names: list[str]) -> list[str]:
    """Map question entity strings to entity ids via surfaces table."""
    if not names:
        return []
    with pg() as cur:
        cur.execute(
            "SELECT id, canonical, surfaces FROM v2_entities WHERE doc_id = %s",
            (doc_id,),
        )
        rows = cur.fetchall()
    ids: list[str] = []
    norms = {re.sub(r"[^\w\s]", "", n.lower()).strip() for n in names if n}
    for r in rows:
        surfaces = r["surfaces"]
        if isinstance(surfaces, str):
            surfaces = json.loads(surfaces)
        cands = [r["canonical"], *(surfaces or [])]
        for s in cands:
            ns = re.sub(r"[^\w\s]", "", (s or "").lower()).strip()
            if ns in norms or any(ns and (ns in n or n in ns) for n in norms):
                ids.append(r["id"])
                break
    return ids


def _first_quote(doc_id: str, event_id: str) -> str:
    with pg() as cur:
        cur.execute(
            """SELECT quote FROM v2_mentions
               WHERE doc_id = %s AND event_id = %s
               ORDER BY char_start LIMIT 1""",
            (doc_id, event_id),
        )
        row = cur.fetchone()
    return (row["quote"] if row else "") or ""


def _mention_spans(doc_id: str, event_id: str) -> list[list[int]]:
    with pg() as cur:
        cur.execute(
            """SELECT char_start, char_end FROM v2_mentions
               WHERE doc_id = %s AND event_id = %s ORDER BY char_start""",
            (doc_id, event_id),
        )
        return [[r["char_start"], r["char_end"]] for r in cur.fetchall()]


def _ground_one(
    doc_id: str, description: str, entity_names: list[str],
    *, chat_fn: Callable | None = None, trace: list[str],
) -> tuple[str | None, str | None, float, list[dict]]:
    """Return (event_id|None, detail|None, confidence, candidates)."""
    ent_ids = _match_entities(doc_id, entity_names)
    by_ent = _events_for_entities(doc_id, ent_ids)
    by_emb = _embed_events(doc_id, description, settings.v2_ground_embed_top)
    seen, cands = set(), []
    for e in by_ent + by_emb:
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        cands.append(e)
    if not cands:
        trace.append(f"grounding '{description}': no candidates")
        return None, "not_found", 0.0, []

    lines = []
    for i, e in enumerate(cands):
        q = scrub_llm_text(_first_quote(doc_id, e["id"]))
        desc = scrub_llm_text(e.get("description") or "")
        lines.append(f"[{i}] {desc}\n    quote: {q}")
    user = f"DESCRIPTION: {scrub_llm_text(description)}\n\nCANDIDATES:\n" + "\n".join(lines)
    fn = chat_fn or (lambda sys, usr, cls, **kw: llm.chat_structured(sys, usr, cls, **kw))
    choice = fn(GROUND_SYSTEM, user, GroundingChoice, temperature=0.0)
    if choice.ambiguous:
        trace.append(f"grounding '{description}': ambiguous among {len(cands)}")
        return None, "ambiguous", 0.0, cands
    if choice.choice < 0 or choice.choice >= len(cands):
        trace.append(f"grounding '{description}': not found")
        return None, "not_found", 0.0, cands
    eid = cands[choice.choice]["id"]
    score = float(cands[choice.choice].get("score") or 0.8)
    trace.append(f"grounded '{description}' -> {eid} ({choice.reason})")
    return eid, None, max(0.1, min(1.0, score)), cands


def _chain_confidence(rel_info: dict) -> float:
    chain = rel_info.get("chain") or []
    if not chain:
        return 1.0
    p = 1.0
    for step in chain:
        p *= float(step.get("p") or step.get("weight") or 1.0)
    return p


def _passages_for_events(doc_id: str, event_ids: list[str], doc_text: str) -> list[dict]:
    pad = settings.v2_passage_pad_chars
    out = []
    for eid in event_ids:
        spans = _mention_spans(doc_id, eid)
        for s, e in spans:
            a, b = max(0, s - pad), min(len(doc_text), e + pad)
            pages = []
            try:
                doc = docstore.load_document(doc_id)
                pages = doc.pages_for_span(a, b)
            except Exception:
                pass
            out.append({
                "event_id": eid, "span": [a, b], "pages": pages,
                "text": doc_text[a:b],
            })
    return out


def _verbalise(
    relation: str, detail: str | None, chain: list, passages: list[dict],
    question: str, *, chat_fn: Callable | None = None,
) -> VerbalisedAnswer:
    clean_passages = []
    for p in passages:
        cp = dict(p)
        if "text" in cp:
            cp["text"] = scrub_llm_text(cp["text"])
        clean_passages.append(cp)
    user = (
        f"QUESTION: {scrub_llm_text(question)}\n"
        f"COMPUTED RELATION: {relation}\n"
        f"DETAIL: {detail or ''}\n"
        f"CHAIN: {json.dumps(chain, ensure_ascii=False)}\n"
        f"PASSAGES:\n{json.dumps(clean_passages, ensure_ascii=False)}\n"
    )
    fn = chat_fn or (lambda sys, usr, cls, **kw: llm.chat_structured(sys, usr, cls, **kw))
    return fn(VERBALISE_SYSTEM, user, VerbalisedAnswer, temperature=0.0)


def _factual(doc_id: str, question: str, k_retrieve: int) -> PipelineAnswer:
    """Hybrid lexical + embedding retrieval over mention passages; naive fallback."""
    t0 = time.perf_counter()
    before = llm.usage_snapshot()
    k = max(k_retrieve, settings.v2_factual_top_k)
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) > 2}
    vec = llm.embed([question])[0]
    with pg() as cur:
        cur.execute(
            """SELECT m.id, m.event_id, m.char_start, m.char_end, m.quote, m.description,
                      1 - (e.embedding <=> %s) AS score
               FROM v2_mentions m
               JOIN v2_events e ON e.id = m.event_id AND e.doc_id = m.doc_id
               WHERE m.doc_id = %s AND e.embedding IS NOT NULL
               ORDER BY e.embedding <=> %s LIMIT %s""",
            (Vector(vec), doc_id, Vector(vec), max(k * 4, k)),
        )
        hits = [dict(r) for r in cur.fetchall()]
    if not hits:
        return naive_rag.answer(doc_id, question, k_retrieve=k_retrieve)

    # Rerank: combine embed score with lexical overlap (BM25-style proxy without extra deps)
    for h in hits:
        text = f"{h.get('quote') or ''} {h.get('description') or ''}".lower()
        toks = {t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 2}
        lex = (len(q_tokens & toks) / max(1, len(q_tokens))) if q_tokens else 0.0
        h["hybrid"] = (
            settings.v2_factual_embed_weight * float(h.get("score") or 0)
            + (1.0 - settings.v2_factual_embed_weight) * lex
        )
    hits.sort(key=lambda h: h["hybrid"], reverse=True)
    hits = hits[:k]

    doc = docstore.load_document(doc_id)
    pad = settings.v2_passage_pad_chars
    retrieved = []
    for i, h in enumerate(hits):
        a, b = max(0, h["char_start"] - pad), min(len(doc.text), h["char_end"] + pad)
        retrieved.append({
            "rank": i + 1, "unit_id": h["event_id"] or h["id"],
            "spans": [[h["char_start"], h["char_end"]]],
            "tokens": max(1, (b - a) // 4),
            "score": float(h.get("hybrid") or 0),
            "text": scrub_llm_text(doc.text[a:b]),
        })
    from ..schemas import NaiveAnswer
    context = "\n\n---\n\n".join(
        f"[Passage {i + 1}]\n{r['text']}" for i, r in enumerate(retrieved[:settings.v2_factual_top_k])
    )
    user = f"PASSAGES:\n\n{context}\n\nQUESTION: {scrub_llm_text(question)}"
    result = llm.chat_structured(
        naive_rag.SYSTEM, user, NaiveAnswer, temperature=0.0,
        max_tokens=800, degrade_on_filter=True)
    after = llm.usage_snapshot()
    return PipelineAnswer(
        pipeline="kaalkram_v2",
        answer=result.answer,
        relation=result.relation,
        confidence=max(0.0, min(1.0, result.confidence)),
        cited_spans=[r["spans"][0] for r in retrieved[:settings.v2_factual_top_k]],
        latency_ms=int((time.perf_counter() - t0) * 1000),
        prompt_tokens=after["prompt"] - before["prompt"],
        completion_tokens=after["completion"] - before["completion"],
        retrieved=retrieved,
        trace=["factual: hybrid mention retrieval"],
    )


def answer(
    doc_id: str, question: str, k_retrieve: int | None = None,
    *, chat_fn: Callable | None = None, graph: TemporalGraph | None = None,
) -> PipelineAnswer:
    t0 = time.perf_counter()
    before = llm.usage_snapshot()
    trace: list[str] = []
    k_retrieve = k_retrieve or settings.v2_ground_embed_top

    fn = chat_fn or (lambda sys, usr, cls, **kw: llm.chat_structured(sys, usr, cls, **kw))
    parse: QuestionParse = fn(QUESTION_SYSTEM, question, QuestionParse, temperature=0.0)
    # Pairwise before/after questions are order; models sometimes emit before_after_x.
    if (parse.qtype == "before_after_x"
            and (parse.event_a or "").strip() and (parse.event_b or "").strip()
            and parse.direction == "none"):
        parse = parse.model_copy(update={"qtype": "order"})
        trace.append("coerced qtype before_after_x -> order (both events present)")
    trace.append(f"parsed qtype={parse.qtype}")

    if parse.qtype == "factual":
        return _factual(doc_id, question, k_retrieve)

    data = load_graph(doc_id) if graph is None else None
    if graph is None:
        if not data:
            after = llm.usage_snapshot()
            return PipelineAnswer(
                pipeline="kaalkram_v2", answer="No v2 graph built for this document.",
                relation="cannot_determine", confidence=0.0,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                prompt_tokens=after["prompt"] - before["prompt"],
                completion_tokens=after["completion"] - before["completion"],
                trace=trace + ["no graph"],
            )
        graph = TemporalGraph.from_dict(data["graph"])

    doc = docstore.load_document(doc_id)

    def ground(desc: str):
        return _ground_one(doc_id, desc, parse.entities, chat_fn=chat_fn, trace=trace)

    relation = "cannot_determine"
    detail = None
    confidence = 0.0
    chain: list = []
    cited: list[list[int]] = []
    retrieved: list[dict] = []
    event_ids_used: list[str] = []

    if parse.qtype == "order":
        a_id, d1, c1, cands_a = ground(parse.event_a)
        b_id, d2, c2, cands_b = ground(parse.event_b)
        for i, e in enumerate(cands_a + cands_b):
            retrieved.append({
                "rank": i + 1, "unit_id": e["id"],
                "spans": _mention_spans(doc_id, e["id"]),
                "tokens": 50, "score": float(e.get("score") or 0),
            })
        if d1 or d2:
            detail = d1 or d2
            confidence = 0.0
        else:
            info = graph.relation(a_id, b_id)
            relation = info["label"]
            detail = info.get("detail")
            chain = info.get("chain") or []
            confidence = _chain_confidence(info) if relation != "cannot_determine" else c1 * c2
            event_ids_used = [a_id, b_id]
            for eid in event_ids_used:
                cited.extend(_mention_spans(doc_id, eid))

    elif parse.qtype == "before_after_x":
        x_id, d, cx, cands = ground(parse.event_a or parse.event_b)
        for i, e in enumerate(cands):
            retrieved.append({
                "rank": i + 1, "unit_id": e["id"],
                "spans": _mention_spans(doc_id, e["id"]),
                "tokens": 50, "score": float(e.get("score") or 0),
            })
        if d:
            detail = d
        else:
            # predecessors/successors from closure via pairwise relation over events
            with pg() as cur:
                cur.execute("SELECT id, participants FROM v2_events WHERE doc_id = %s", (doc_id,))
                all_ev = [dict(r) for r in cur.fetchall()]
            x_parts = set()
            for e in all_ev:
                if e["id"] == x_id:
                    parts = e.get("participants") or []
                    if isinstance(parts, str):
                        parts = json.loads(parts)
                    x_parts = set(parts)
            related = []
            for e in all_ev:
                if e["id"] == x_id:
                    continue
                info = graph.relation(e["id"], x_id)
                lab = info["label"]
                want = parse.direction  # before => events before X; after => after X
                if want == "before" and lab == "before":
                    related.append((e, info))
                elif want == "after" and lab == "after":
                    related.append((e, info))
                elif want == "before" and info.get("detail") is None and lab == "after":
                    continue
                elif want == "after" and lab == "before":
                    # e before x means e is before; for "after x" we want x before e
                    info2 = graph.relation(x_id, e["id"])
                    if info2["label"] == "before":
                        related.append((e, info2))
            # filter: share entity OR top-N by embedding similarity
            sim = {e["id"]: e for e in _embed_events(doc_id, parse.event_a or question,
                                                       settings.v2_before_after_sim_top)}
            kept = []
            for e, info in related:
                parts = e.get("participants") or []
                if isinstance(parts, str):
                    parts = json.loads(parts)
                if (x_parts & set(parts)) or e["id"] in sim:
                    kept.append(e["id"])
            order = graph.linear_extension()
            kept.sort(key=lambda eid: order.index(eid) if eid in order else 10**9)
            event_ids_used = [x_id] + kept
            relation = "not_applicable"
            detail = "display_order_only"
            confidence = cx
            for eid in event_ids_used:
                cited.extend(_mention_spans(doc_id, eid))
            trace.append(f"before_after_x: {len(kept)} related (display order only)")

    elif parse.qtype == "sequence":
        ids = []
        details = []
        for desc in parse.events_list:
            eid, d, c, cands = ground(desc)
            for e in cands:
                if not any(r["unit_id"] == e["id"] for r in retrieved):
                    retrieved.append({
                        "rank": len(retrieved) + 1, "unit_id": e["id"],
                        "spans": _mention_spans(doc_id, e["id"]),
                        "tokens": 50, "score": float(e.get("score") or 0),
                    })
            if d:
                details.append(d)
            else:
                ids.append(eid)
        unordered_pairs = []
        labels = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                info = graph.relation(ids[i], ids[j])
                labels[(ids[i], ids[j])] = info["label"]
                if info["label"] == "cannot_determine":
                    unordered_pairs.append((ids[i], ids[j], info.get("detail")))
        if details:
            detail = details[0]
            relation = "cannot_determine"
        else:
            relation = "not_applicable"
            detail = "unordered_pairs" if unordered_pairs else "sequence_ok"
            confidence = 1.0
            event_ids_used = ids
            for eid in ids:
                cited.extend(_mention_spans(doc_id, eid))
            trace.append(f"sequence unordered pairs: {unordered_pairs}")

    elif parse.qtype == "state":
        # lifecycle of entity vs event X
        x_id, d, cx, cands = ground(parse.event_a)
        for i, e in enumerate(cands):
            retrieved.append({
                "rank": i + 1, "unit_id": e["id"],
                "spans": _mention_spans(doc_id, e["id"]),
                "tokens": 50, "score": float(e.get("score") or 0),
            })
        if d:
            detail = d
        else:
            ent_ids = _match_entities(doc_id, parse.entities)
            with pg() as cur:
                cur.execute(
                    """SELECT id, description, event_type, first_offset FROM v2_events
                       WHERE doc_id = %s AND event_type IN ('birth','death')""",
                    (doc_id,),
                )
                life = [dict(r) for r in cur.fetchall()]
            # filter to entity participants
            relevant = []
            for e in life:
                # check via mentions subject
                with pg() as cur:
                    cur.execute(
                        """SELECT subject_entity, participants FROM v2_mentions
                           WHERE doc_id = %s AND event_id = %s LIMIT 5""",
                        (doc_id, e["id"]),
                    )
                    ms = cur.fetchall()
                for m in ms:
                    sub = m["subject_entity"]
                    parts = m["participants"] or []
                    if isinstance(parts, str):
                        parts = json.loads(parts)
                    pids = {sub} if sub else set()
                    for p in parts:
                        if isinstance(p, dict) and p.get("entity_id"):
                            pids.add(p["entity_id"])
                        elif isinstance(p, str):
                            pids.add(p)
                    if pids & set(ent_ids):
                        relevant.append(e)
                        break
            alive = True
            present = True
            for e in relevant:
                info = graph.relation(e["id"], x_id)
                if e["event_type"] == "death" and info["label"] == "before":
                    alive = False
                if e["event_type"] == "birth" and info["label"] == "after":
                    # birth after X => not yet born at X
                    alive = False
                    present = False
            relation = "not_applicable"
            detail = f"alive={alive};present={present}"
            confidence = cx
            event_ids_used = [x_id] + [e["id"] for e in relevant]
            for eid in event_ids_used:
                cited.extend(_mention_spans(doc_id, eid))

    else:
        relation = "cannot_determine"
        detail = "unsupported_qtype"

    passages = _passages_for_events(doc_id, event_ids_used, doc.text)
    try:
        verb = _verbalise(relation, detail, chain, passages, question, chat_fn=chat_fn)
        answer_text = verb.answer
        if verb.relation != relation:
            trace.append(f"verbalise changed relation {verb.relation} -> kept {relation}")
    except Exception as exc:
        answer_text = f"Relation: {relation}" + (f" ({detail})" if detail else "")
        trace.append(f"verbalise failed: {type(exc).__name__}: {exc}")

    # rank retrieved: grounded, then other candidates, then chain events
    seen_u = {r["unit_id"] for r in retrieved}
    for eid in event_ids_used:
        if eid not in seen_u:
            retrieved.append({
                "rank": len(retrieved) + 1, "unit_id": eid,
                "spans": _mention_spans(doc_id, eid), "tokens": 50, "score": 0.0,
            })
    for i, r in enumerate(retrieved):
        r["rank"] = i + 1

    after = llm.usage_snapshot()
    pages_cite = []
    for sp in cited[:8]:
        try:
            pgs = doc.pages_for_span(sp[0], sp[1])
            if pgs:
                pages_cite.append(Citation(label=f"p. {pgs[0]}", pages=pgs))
        except Exception:
            pass

    # Frontend reads these from trace (Task 9: evidence chain + cannot_determine detail).
    if detail:
        trace.append(f"detail={detail}")
    if chain:
        # Attach page numbers to evidence spans when possible.
        enriched = []
        for step in chain:
            s = dict(step)
            ev_out = []
            for ev in (step.get("evidence") or []):
                e2 = dict(ev)
                span = ev.get("span") or []
                if len(span) == 2:
                    try:
                        e2["pages"] = doc.pages_for_span(int(span[0]), int(span[1]))
                        e2["quote"] = doc.text[int(span[0]):int(span[1])]
                    except Exception:
                        pass
                ev_out.append(e2)
            s["evidence"] = ev_out
            enriched.append(s)
        trace.append("chain=" + json.dumps(enriched, ensure_ascii=False))

    # Enrich retrieved units with quote + pages for the evidence panel.
    for r in retrieved:
        spans = r.get("spans") or []
        if spans and "preview" not in r:
            a, b = spans[0][0], spans[0][1]
            try:
                r["preview"] = doc.text[max(0, a):min(len(doc.text), b)][:400]
                r["pages"] = doc.pages_for_span(a, b)
            except Exception:
                pass

    return PipelineAnswer(
        pipeline="kaalkram_v2",
        answer=answer_text,
        relation=relation,
        confidence=float(confidence),
        cited_spans=cited,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        prompt_tokens=after["prompt"] - before["prompt"],
        completion_tokens=after["completion"] - before["completion"],
        citations=pages_cite,
        retrieved=retrieved,
        trace=trace,
    )
