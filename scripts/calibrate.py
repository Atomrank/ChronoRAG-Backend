"""
Calibrate v2 source weights on the DEV gold split, rebuild the graph, report on TEST.

  python scripts/calibrate.py DOC_ID data/gold/verified.jsonl

Maps gold events -> v2 events by mention/evidence span overlap
(>= settings.eval_calibrate_overlap_frac). Calls calibrate_sources on DEV pairs
only, stores weights in v2_graphs.weights, rebuilds the temporal graph without
re-extraction, prints the weight table, then reports ordering accuracy on TEST.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import eval_runner as E  # noqa: E402
from app.config import settings  # noqa: E402
from app.v2 import persist  # noqa: E402
from app.v2.constraints import (  # noqa: E402
    LocalRelation, SOURCE_PRIOR, all_constraints, calibrate_sources,
)
from app.v2.solver import TemporalGraph  # noqa: E402


def _span_overlap_frac(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Intersection / min(lengths) — same convention as extract._span_overlap."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(1, min(a[1] - a[0], b[1] - b[0]))


def map_gold_event_to_v2(
    evidence_span: tuple[int, int],
    mentions: list,
    *,
    thr: float | None = None,
) -> str | None:
    """Best v2 event_id whose mention span overlaps evidence by >= thr."""
    thr = settings.eval_calibrate_overlap_frac if thr is None else thr
    best_id, best_ov = None, 0.0
    for m in mentions:
        if not m.event_id:
            continue
        ov = _span_overlap_frac((m.start, m.end), evidence_span)
        if ov >= thr and ov > best_ov:
            best_ov, best_id = ov, m.event_id
    return best_id


def gold_to_v2_pairs(
    questions: list[E.GoldQuestion],
    mentions: list,
) -> tuple[list[tuple[str, str, str]], dict]:
    """Return (event_a, event_b, label) in v2 ids for order questions with before/after."""
    pairs = []
    stats = {"mapped": 0, "unmapped": 0, "skipped_unordered": 0, "by_qid": {}}
    for q in questions:
        if q.qtype != "order" or not q.events or {"A", "B"} - set(q.events):
            continue
        if q.gold_label not in ("before", "after"):
            stats["skipped_unordered"] += 1
            continue
        spans: dict[str, tuple[int, int]] = {}
        for ev in q.evidence:
            if ev.group in ("A", "B") and ev.char_start is not None and ev.char_end is not None:
                spans[ev.group] = (ev.char_start, ev.char_end)
        if "A" not in spans or "B" not in spans:
            stats["unmapped"] += 1
            stats["by_qid"][q.id] = "missing_spans"
            continue
        va = map_gold_event_to_v2(spans["A"], mentions)
        vb = map_gold_event_to_v2(spans["B"], mentions)
        if not va or not vb or va == vb:
            stats["unmapped"] += 1
            stats["by_qid"][q.id] = f"map_fail a={va} b={vb}"
            continue
        pairs.append((va, vb, q.gold_label))
        stats["mapped"] += 1
        stats["by_qid"][q.id] = {"a": va, "b": vb, "label": q.gold_label}
    return pairs, stats


def report_test_accuracy(graph: TemporalGraph, pairs: list[tuple[str, str, str]]) -> dict:
    """Ordering accuracy of graph.relation vs gold on TEST-mapped pairs."""
    n, correct, by_label = 0, 0, defaultdict(lambda: [0, 0])
    for a, b, lab in pairs:
        if a not in graph.events or b not in graph.events:
            continue
        pred = graph.relation(a, b).get("label")
        n += 1
        by_label[lab][1] += 1
        if pred == lab:
            correct += 1
            by_label[lab][0] += 1
    return {
        "n": n,
        "accuracy": (correct / n) if n else None,
        "correct": correct,
        "by_gold_label": {k: {"correct": v[0], "n": v[1],
                              "acc": (v[0] / v[1] if v[1] else None)}
                         for k, v in by_label.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate v2 source weights on DEV gold")
    ap.add_argument("doc_id")
    ap.add_argument("gold", type=Path, help="verified gold JSONL (char offsets set)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Beta prior alpha")
    ap.add_argument("--beta", type=float, default=1.0, help="Beta prior beta")
    args = ap.parse_args()

    questions = E.load_gold(args.gold)
    missing = [q.id for q in questions for e in q.evidence if e.char_start is None]
    if missing:
        raise SystemExit(
            f"gold has unverified evidence ({len(missing)} questions); "
            f"run: python -m app.eval_runner verify --doc {args.doc_id} "
            f"--gold {args.gold} --out <verified.jsonl>"
        )

    mentions = persist.load_mentions(args.doc_id)
    if not mentions:
        raise SystemExit(f"no v2 mentions for {args.doc_id}; build kaalkram_v2 first")

    # Constraints under SOURCE_PRIOR (no DB write) — calibrate_sources only needs
    # source labels + event endpoints, not the prior numeric weights.
    frames = persist.load_frames(args.doc_id)
    relations = persist.load_relations(args.doc_id)
    kinship = persist.load_stored_kinship(args.doc_id)
    cm = persist._to_constraint_mentions(mentions)
    rels = [
        LocalRelation(
            a=r.a, b=r.b, rel=r.rel, cue=r.cue, consistency=r.consistency,
            span=(r.start, r.end) if r.start is not None else None,
        )
        for r in relations
    ]
    cons = all_constraints(frames, cm, rels, kinship, dict(SOURCE_PRIOR))

    dev_qs = E.filter_by_split(questions, "dev")
    test_qs = E.filter_by_split(questions, "test")
    print(f"questions: {len(questions)}  "
          f"dev={len(dev_qs)} ({settings.eval_dev_fraction:.0%})  "
          f"test={len(test_qs)}")
    print(f"overlap thr={settings.eval_calibrate_overlap_frac}")

    dev_pairs, dev_map = gold_to_v2_pairs(dev_qs, mentions)
    test_pairs, test_map = gold_to_v2_pairs(test_qs, mentions)
    print(f"DEV mapped pairs: {dev_map['mapped']}  unmapped={dev_map['unmapped']}  "
          f"skipped_unordered={dev_map['skipped_unordered']}")
    print(f"TEST mapped pairs: {test_map['mapped']}  unmapped={test_map['unmapped']}")

    if not dev_pairs:
        raise SystemExit("no DEV pairs mapped; cannot calibrate")

    cal = calibrate_sources(cons, dev_pairs, alpha=args.alpha, beta=args.beta)
    # Keep prior for sources with no DEV evidence
    store = {s: {"weight": SOURCE_PRIOR[s], "n": 0, "agree": 0} for s in SOURCE_PRIOR}
    store.update(cal)

    print("\n=== calibrated source weights (DEV) ===")
    print(f"{'source':20s}  {'weight':>7s}  {'agree':>5s}  {'n':>5s}  {'prior':>7s}")
    for src in sorted(store, key=lambda s: -store[s]["weight"]):
        info = store[src]
        prior = SOURCE_PRIOR.get(src, 0.5)
        print(f"{src:20s}  {info['weight']:7.3f}  {info.get('agree', 0):5d}  "
              f"{info.get('n', 0):5d}  {prior:7.3f}")

    rebuilt = persist.rebuild_graph(args.doc_id, weights=store)
    g: TemporalGraph = rebuilt["graph_obj"]
    print(f"\nrebuilt graph: constraints={rebuilt['constraints']}  "
          f"repair={rebuilt['repair']}")

    test_rep = report_test_accuracy(g, test_pairs)
    print("\n=== TEST ordering (mapped pairs, graph.relation) ===")
    if test_rep["n"]:
        print(f"accuracy={test_rep['accuracy']:.3f}  "
              f"correct={test_rep['correct']}/{test_rep['n']}")
        for lab, row in sorted(test_rep["by_gold_label"].items()):
            print(f"  gold={lab}: {row['correct']}/{row['n']}  acc={row['acc']:.3f}")
    else:
        print("no TEST pairs mapped")
    print(
        "\nFor full metrics on TEST:\n"
        f"  python -m app.eval_runner run --doc {args.doc_id} --gold {args.gold} "
        f"--pipeline kaalkram_v2 --split test"
    )


if __name__ == "__main__":
    main()
