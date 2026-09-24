"""
Benchmark metrics for Naive RAG vs Kaalkram v1 vs Kaalkram v2.

Pure functions, no DB or LLM. Every metric is computed from per-question
records, so any number in a report can be recomputed later from the stored
eval_items rows.

Relevance is judged on CHARACTER SPANS of the cleaned document text, so a
naive chunk, a v1 event (mapped via its pages) and a v2 event (exact offsets)
are all scored on the same yardstick.

Span  = (start, end) half-open character interval.
Unit  = one retrieved item: {"rank": int, "spans": [(s, e), ...], "tokens": int?}
Gold  = list of evidence items: {"span": (s, e), "group": "A" | "B" | None}
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Sequence

LABELS = ("before", "after", "cannot_determine")
INVERSE = {"before": "after", "after": "before", "cannot_determine": "cannot_determine"}

# Relevance thresholds (hyperparameters — report them).
GOLD_COVER_FRAC = 0.5   # a gold span is "found" if one unit covers >= 50% of it
UNIT_INSIDE_FRAC = 0.5  # a unit is "relevant" if it finds a gold span or >= 50% of it lies in gold


# ============================================================
# Span helpers
# ============================================================
def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _unit_cover(unit_spans: Sequence[tuple[int, int]], gold: tuple[int, int]) -> float:
    glen = max(1, gold[1] - gold[0])
    return sum(overlap(s, gold) for s in unit_spans) / glen


def _unit_inside(unit_spans: Sequence[tuple[int, int]], golds: Sequence[tuple[int, int]]) -> float:
    ulen = sum(max(0, e - s) for s, e in unit_spans) or 1
    inside = sum(overlap(s, g) for s in unit_spans for g in golds)
    return min(1.0, inside / ulen)


def unit_finds(unit: dict, gold_span: tuple[int, int], thr: float = GOLD_COVER_FRAC) -> bool:
    return _unit_cover(unit["spans"], gold_span) >= thr


def unit_relevant(unit: dict, golds: Sequence[tuple[int, int]]) -> bool:
    return (any(unit_finds(unit, g) for g in golds)
            or _unit_inside(unit["spans"], golds) >= UNIT_INSIDE_FRAC)


def _ranked(units: Iterable[dict]) -> list[dict]:
    return sorted(units, key=lambda u: u["rank"])


# ============================================================
# Retrieval metrics (per question)
# ============================================================
def recall_at_k(units: list[dict], gold: list[dict], k: int) -> float | None:
    if not gold:
        return None
    top = _ranked(units)[:k]
    found = sum(1 for g in gold if any(unit_finds(u, g["span"]) for u in top))
    return found / len(gold)


def precision_at_k(units: list[dict], gold: list[dict], k: int) -> float | None:
    if not gold:
        return None
    golds = [g["span"] for g in gold]
    top = _ranked(units)[:k]
    return sum(unit_relevant(u, golds) for u in top) / k


def hit_at_k(units: list[dict], gold: list[dict], k: int) -> float | None:
    if not gold:
        return None
    golds = [g["span"] for g in gold]
    return float(any(unit_relevant(u, golds) for u in _ranked(units)[:k]))


def reciprocal_rank(units: list[dict], gold: list[dict]) -> float | None:
    if not gold:
        return None
    golds = [g["span"] for g in gold]
    for i, u in enumerate(_ranked(units), start=1):
        if unit_relevant(u, golds):
            return 1.0 / i
    return 0.0


def _relevant_units_dedup(units: list[dict], golds: list[tuple[int, int]]) -> list[dict]:
    """Unique relevant retrieval units (by span set), same relevance as precision/recall."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for u in _ranked(units):
        if not unit_relevant(u, golds):
            continue
        key = tuple(sorted(u["spans"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def ndcg_at_k(units: list[dict], gold: list[dict], k: int) -> float | None:
    """Binary relevance. Ideal DCG over the deduplicated relevant units, capped at k.

    DCG also credits each unique span-set at most once (first occurrence in rank
    order) so ndcg stays in [0, 1] when retrieval returns duplicate covers.
    """
    if not gold:
        return None
    golds = [g["span"] for g in gold]
    ranked = _ranked(units)
    n_ideal = min(k, len(_relevant_units_dedup(ranked, golds)))
    if n_ideal == 0:
        return 0.0
    seen: set[tuple] = set()
    dcg = 0.0
    for i, u in enumerate(ranked[:k], start=1):
        if not unit_relevant(u, golds):
            continue
        key = tuple(sorted(u["spans"]))
        if key in seen:
            continue
        seen.add(key)
        dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return dcg / ideal


def pair_recall_at_k(units: list[dict], gold: list[dict], k: int) -> float | None:
    """1 if evidence for BOTH event A and event B is in the top k (ordering questions)."""
    groups: dict[str, list[tuple[int, int]]] = {}
    for g in gold:
        if g.get("group") in ("A", "B"):
            groups.setdefault(g["group"], []).append(g["span"])
    if set(groups) != {"A", "B"}:
        return None
    top = _ranked(units)[:k]
    ok = all(any(unit_finds(u, s) for u in top for s in spans) for spans in groups.values())
    return float(ok)


def recall_at_budget(units: list[dict], gold: list[dict], budget_tokens: int) -> float | None:
    """Recall when retrieval is cut at a fixed context budget, so a pipeline that
    returns bigger units is not credited for simply showing the LLM more text."""
    if not gold:
        return None
    taken, used = [], 0
    for u in _ranked(units):
        t = u.get("tokens") or max(1, sum(e - s for s, e in u["spans"]) // 4)
        if used + t > budget_tokens and taken:
            break
        taken.append(u)
        used += t
    found = sum(1 for g in gold if any(unit_finds(u, g["span"]) for u in taken))
    return found / len(gold)


def retrieval_metrics(units: list[dict], gold: list[dict], ks: Sequence[int],
                      budgets: Sequence[int] = ()) -> dict:
    out: dict = {"mrr": reciprocal_rank(units, gold)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(units, gold, k)
        out[f"precision@{k}"] = precision_at_k(units, gold, k)
        out[f"hit@{k}"] = hit_at_k(units, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(units, gold, k)
        out[f"pair_recall@{k}"] = pair_recall_at_k(units, gold, k)
    for b in budgets:
        out[f"recall@{b}tok"] = recall_at_budget(units, gold, b)
    return out


# ============================================================
# Citation quality (per question)
# ============================================================
def citation_pr(cited: list[tuple[int, int]], gold: list[dict]) -> dict:
    golds = [g["span"] for g in gold]
    if not golds:
        return {"citation_precision": None, "citation_recall": None}
    if not cited:
        return {"citation_precision": 0.0, "citation_recall": 0.0}
    prec = sum(any(_unit_cover([c], g) >= GOLD_COVER_FRAC for g in golds)
               or _unit_inside([c], golds) >= UNIT_INSIDE_FRAC for c in cited) / len(cited)
    rec = sum(any(_unit_cover([c], g) >= GOLD_COVER_FRAC for c in cited) for g in golds) / len(golds)
    return {"citation_precision": prec, "citation_recall": rec}


# ============================================================
# Ordering / answer metrics (over many questions)
# ============================================================
def classification_report(gold: list[str], pred: list[str]) -> dict:
    n = len(gold)
    if n == 0:
        return {}
    per = {}
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[c] = {"precision": pr, "recall": rc, "f1": f1, "support": tp + fn}
    present = [c for c in LABELS if per[c]["support"] > 0]
    committed = [i for i, p in enumerate(pred) if p != "cannot_determine"]
    gold_cd = [i for i, g in enumerate(gold) if g == "cannot_determine"]
    confusion = {g: {p: sum(1 for a, b in zip(gold, pred) if a == g and b == p) for p in LABELS}
                 for g in LABELS}
    return {
        "n": n,
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / n,
        "macro_f1": sum(per[c]["f1"] for c in present) / len(present) if present else None,
        "per_class": per,
        "coverage": len(committed) / n,
        "selective_accuracy": (sum(gold[i] == pred[i] for i in committed) / len(committed)
                               if committed else None),
        # committed before/after answers on pairs the text leaves unordered
        "made_up_order_rate": (sum(pred[i] != "cannot_determine" for i in gold_cd) / len(gold_cd)
                               if gold_cd else None),
        "confusion": confusion,
    }


def risk_coverage(gold: list[str], pred: list[str], conf: list[float]) -> dict:
    """Sort by confidence, report accuracy at each coverage level and the area
    under the risk-coverage curve (AURC, lower is better)."""
    order = sorted(range(len(gold)), key=lambda i: -conf[i])
    points, correct = [], 0
    for j, i in enumerate(order, start=1):
        correct += gold[i] == pred[i]
        points.append({"coverage": j / len(order), "accuracy": correct / j})
    aurc = sum(1 - p["accuracy"] for p in points) / len(points) if points else None
    return {"curve": points, "aurc": aurc}


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for m in range(i, j + 1):
            ranks[order[m]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall's tau-b (tie-corrected)."""
    n = len(x)
    if n < 2:
        return None
    conc = disc = tx = ty = 0
    for i, j in combinations(range(n), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx * dy > 0:
            conc += 1
        else:
            disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom else None


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def order_correlation(system_pos: dict[str, float], gold_order: list[str]) -> dict:
    """Global-order agreement over gold events the system also placed."""
    common = [e for e in gold_order if e in system_pos]
    gx = list(range(len(common)))
    sy = [system_pos[e] for e in common]
    return {"n_events": len(common), "kendall_tau_b": kendall_tau_b(gx, sy),
            "spearman_rho": spearman_rho(gx, sy)}


def sequence_scores(pred_seq: list[str], gold_seq: list[str]) -> dict:
    pos = {e: i for i, e in enumerate(pred_seq)}
    common = [e for e in gold_seq if e in pos]
    return {
        "exact_match": float(pred_seq == gold_seq),
        "kendall_tau_b": kendall_tau_b(list(range(len(common))), [pos[e] for e in common]),
        "item_recall": len(common) / len(gold_seq) if gold_seq else None,
    }


# ============================================================
# Consistency (no gold needed)
# ============================================================
def symmetry_consistency(pairs: list[tuple[str, str]]) -> float | None:
    """pairs = [(answer to 'A before B?', answer to 'B before A?')] as labels.
    Consistent when the second is the inverse of the first."""
    if not pairs:
        return None
    return sum(INVERSE[a] == b for a, b in pairs) / len(pairs)


def transitivity_consistency(triples: list[dict]) -> dict:
    """triples = [{"ab": label, "bc": label, "ac": label}], each label for
    '<first> relative to <second>'. A triple is inconsistent if its committed
    answers form a directed cycle (e.g. a<b, b<c, c<a)."""
    def edges(t: dict) -> set[tuple[str, str]]:
        out = set()
        for key, (x, y) in {"ab": ("a", "b"), "bc": ("b", "c"), "ac": ("a", "c")}.items():
            if t[key] == "before":
                out.add((x, y))
            elif t[key] == "after":
                out.add((y, x))
        return out

    def cyclic(es: set[tuple[str, str]]) -> bool:
        return ({("a", "b"), ("b", "c"), ("c", "a")} <= es
                or {("b", "a"), ("c", "b"), ("a", "c")} <= es)

    if not triples:
        return {"n": 0, "consistency_all": None, "consistency_committed": None}
    flags = [cyclic(edges(t)) for t in triples]
    full = [f for t, f in zip(triples, flags)
            if all(t[k] != "cannot_determine" for k in ("ab", "bc", "ac"))]
    return {
        "n": len(triples),
        "consistency_all": 1 - sum(flags) / len(flags),
        "n_fully_committed": len(full),
        "consistency_committed": (1 - sum(full) / len(full)) if full else None,
    }


# ============================================================
# Statistics
# ============================================================
def bootstrap_ci(values: Sequence[float], stat: Callable = None, n: int = 10000,
                 alpha: float = 0.05, seed: int = 13) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"value": None, "lo": None, "hi": None, "n": 0}
    stat = stat or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    boots = sorted(stat([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(n))
    lo = boots[int((alpha / 2) * n)]
    hi = boots[min(n - 1, int((1 - alpha / 2) * n))]
    return {"value": stat(vals), "lo": lo, "hi": hi, "n": len(vals)}


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n: int = 10000,
                          seed: int = 13) -> dict:
    """CI for mean(a) - mean(b) on the same questions, plus a one-sided p-value
    for 'a is not better than b'."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"diff": None, "lo": None, "hi": None, "p_one_sided": None}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        diffs.append(sum(x - y for x, y in s) / len(s))
    diffs.sort()
    obs = sum(x - y for x, y in pairs) / len(pairs)
    return {"diff": obs, "lo": diffs[int(0.025 * n)], "hi": diffs[int(0.975 * n) - 1],
            "p_one_sided": sum(d <= 0 for d in diffs) / n, "n": len(pairs)}


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test on paired correctness."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)   # a right, b wrong
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)   # b right, a wrong
    m = b + c
    if m == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    k = min(b, c)
    p = sum(math.comb(m, i) for i in range(0, k + 1)) / 2 ** m
    return {"a_only": b, "b_only": c, "p_value": min(1.0, 2 * p)}


def percentile(values: Sequence[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


# ============================================================
# Aggregation over a run
# ============================================================
@dataclass
class ItemRecord:
    """One question answered by one pipeline. Mirrors an eval_items row."""
    question_id: str
    qtype: str                      # order | sequence | before_after_x | state | factual
    stratum: str | None             # aligned | inverted | unordered | None
    gold_label: str | None
    pred_label: str | None
    confidence: float | None
    gold_evidence: list[dict]       # [{"span": (s,e), "group": "A"|"B"|None}]
    retrieved: list[dict]           # [{"rank", "spans", "tokens"}]
    cited_spans: list[tuple[int, int]]
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    gold_sequence: list[str] | None = None
    pred_sequence: list[str] | None = None
    pair_key: str | None = None     # links a question to its reversed twin
    triple_key: str | None = None   # links the three questions of a triple
    triple_slot: str | None = None  # "ab" | "bc" | "ac"
    reversed_: bool = False


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(items: list[ItemRecord], ks: Sequence[int] = (1, 3, 5, 8, 10, 20),
              budgets: Sequence[int] = (1000, 2000, 4000), n_boot: int = 10000) -> dict:
    per_q = []
    for it in items:
        r = retrieval_metrics(it.retrieved, it.gold_evidence, ks, budgets)
        r.update(citation_pr(it.cited_spans, it.gold_evidence))
        per_q.append(r)

    keys = sorted({k for r in per_q for k in r})
    retrieval = {k: bootstrap_ci([r.get(k) for r in per_q], n=n_boot) for k in keys}

    # ordering questions
    ordq = [it for it in items if it.gold_label in LABELS and not it.reversed_]
    ordering = {"overall": classification_report([i.gold_label for i in ordq],
                                                 [i.pred_label or "cannot_determine" for i in ordq])}
    if ordq:
        ordering["overall"]["accuracy_ci"] = bootstrap_ci(
            [float(i.gold_label == i.pred_label) for i in ordq], n=n_boot)
    for s in ("aligned", "inverted", "unordered"):
        sub = [i for i in ordq if i.stratum == s]
        if sub:
            rep = classification_report([i.gold_label for i in sub],
                                        [i.pred_label or "cannot_determine" for i in sub])
            rep["accuracy_ci"] = bootstrap_ci([float(i.gold_label == i.pred_label) for i in sub],
                                              n=n_boot)
            ordering[s] = rep
    conf_items = [i for i in ordq if i.confidence is not None]
    if conf_items:
        ordering["risk_coverage"] = risk_coverage(
            [i.gold_label for i in conf_items],
            [i.pred_label or "cannot_determine" for i in conf_items],
            [i.confidence for i in conf_items])
    for s in ("aligned", "inverted", "unordered"):
        if s not in ordering:
            continue
        sub_conf = [i for i in ordq if i.stratum == s and i.confidence is not None]
        if sub_conf:
            ordering[s]["risk_coverage"] = risk_coverage(
                [i.gold_label for i in sub_conf],
                [i.pred_label or "cannot_determine" for i in sub_conf],
                [i.confidence for i in sub_conf])

    # sequences
    # only scored when the pipeline actually returned a sequence
    seqs = [sequence_scores(i.pred_sequence, i.gold_sequence)
            for i in items if i.gold_sequence and i.pred_sequence is not None]
    sequence = {"n": len(seqs),
                "exact_match": _mean([s["exact_match"] for s in seqs]),
                "kendall_tau_b": _mean([s["kendall_tau_b"] for s in seqs]),
                "item_recall": _mean([s["item_recall"] for s in seqs])} if seqs else {}

    # consistency
    by_pair: dict[str, dict[bool, str]] = {}
    for it in items:
        if it.pair_key and it.pred_label:
            by_pair.setdefault(it.pair_key, {})[it.reversed_] = it.pred_label
    sym = symmetry_consistency([(d[False], d[True]) for d in by_pair.values()
                                if False in d and True in d])
    by_triple: dict[str, dict[str, str]] = {}
    for it in items:
        if it.triple_key and it.triple_slot and it.pred_label:
            by_triple.setdefault(it.triple_key, {})[it.triple_slot] = it.pred_label
    trans = transitivity_consistency([t for t in by_triple.values()
                                      if {"ab", "bc", "ac"} <= set(t)])

    lat = [it.latency_ms for it in items]
    return {
        "n_items": len(items),
        "retrieval": retrieval,
        "ordering": ordering,
        "sequence": sequence,
        "consistency": {"symmetry": sym, "transitivity": trans},
        "cost": {
            "latency_p50_ms": percentile(lat, 0.5),
            "latency_p95_ms": percentile(lat, 0.95),
            "latency_mean_ms": _mean(lat),
            "prompt_tokens_mean": _mean([i.prompt_tokens for i in items]),
            "completion_tokens_mean": _mean([i.completion_tokens for i in items]),
            "prompt_tokens_total": sum(i.prompt_tokens for i in items),
            "completion_tokens_total": sum(i.completion_tokens for i in items),
        },
        "thresholds": {"gold_cover_frac": GOLD_COVER_FRAC, "unit_inside_frac": UNIT_INSIDE_FRAC,
                       "ks": list(ks), "budgets": list(budgets), "bootstrap_n": n_boot},
    }


def compare_runs(a: list[ItemRecord], b: list[ItemRecord], n_boot: int = 10000) -> dict:
    """Paired significance between two pipelines on the same question ids."""
    ma = {i.question_id: i for i in a if not i.reversed_}
    mb = {i.question_id: i for i in b if not i.reversed_}
    common = [q for q in ma if q in mb and ma[q].gold_label in LABELS]
    ca = [ma[q].pred_label == ma[q].gold_label for q in common]
    cb = [mb[q].pred_label == mb[q].gold_label for q in common]
    out = {"n": len(common), "mcnemar": mcnemar_exact(ca, cb),
           "accuracy_diff": paired_bootstrap_diff([float(x) for x in ca],
                                                  [float(x) for x in cb], n=n_boot)}
    for s in ("aligned", "inverted", "unordered"):
        qs = [q for q in common if ma[q].stratum == s]
        if qs:
            out[s] = {"n": len(qs), "mcnemar": mcnemar_exact(
                [ma[q].pred_label == ma[q].gold_label for q in qs],
                [mb[q].pred_label == mb[q].gold_label for q in qs])}
    return out
