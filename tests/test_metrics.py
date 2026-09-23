import math
import pytest

from app import metrics as M

GOLD = [{"span": (100, 200), "group": "A"}, {"span": (500, 560), "group": "B"}]


def U(rank, *spans, tokens=None):
    return {"rank": rank, "spans": list(spans), "tokens": tokens}


def test_recall_precision_hit_mrr():
    units = [U(1, (0, 90)), U(2, (120, 400)), U(3, (480, 600)), U(4, (900, 1000))]
    assert M.recall_at_k(units, GOLD, 1) == 0
    assert M.recall_at_k(units, GOLD, 2) == 0.5           # (120,400) covers 80% of A
    assert M.recall_at_k(units, GOLD, 3) == 1.0
    assert M.precision_at_k(units, GOLD, 4) == 0.5
    assert M.hit_at_k(units, GOLD, 1) == 0.0 and M.hit_at_k(units, GOLD, 2) == 1.0
    assert M.reciprocal_rank(units, GOLD) == 0.5
    assert M.pair_recall_at_k(units, GOLD, 2) == 0.0      # only A so far
    assert M.pair_recall_at_k(units, GOLD, 3) == 1.0


def test_partial_cover_threshold():
    assert not M.unit_finds(U(1, (100, 140)), (100, 200))  # 40% < 50%
    assert M.unit_finds(U(1, (100, 150)), (100, 200))       # exactly 50%
    assert M.unit_finds(U(1, (100, 130), (160, 190)), (100, 200))   # multi-span event


def test_ndcg():
    units = [U(1, (100, 200)), U(2, (0, 50)), U(3, (500, 560))]
    ideal = 1 + 1 / math.log2(3)
    assert M.ndcg_at_k(units, GOLD, 3) == pytest.approx((1 + 1 / math.log2(4)) / ideal)


def test_budget_recall_penalises_big_units():
    big = [U(1, (0, 10000), tokens=2500)]
    small = [U(1, (100, 200), tokens=30), U(2, (500, 560), tokens=20)]
    assert M.recall_at_budget(big, GOLD, 2000) == 1.0      # first unit always allowed
    assert M.recall_at_budget(small, GOLD, 40) == 0.5
    assert M.recall_at_budget(small, GOLD, 60) == 1.0


def test_classification_and_abstention():
    gold = ["before", "after", "cannot_determine", "cannot_determine", "before"]
    pred = ["before", "before", "cannot_determine", "after", "cannot_determine"]
    r = M.classification_report(gold, pred)
    assert r["accuracy"] == pytest.approx(0.4)
    assert r["coverage"] == pytest.approx(3 / 5)
    assert r["selective_accuracy"] == pytest.approx(1 / 3)
    assert r["made_up_order_rate"] == pytest.approx(0.5)
    assert r["confusion"]["after"]["before"] == 1


def test_kendall_and_spearman():
    assert M.kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1)
    assert M.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    scipy = pytest.importorskip("scipy.stats")
    x, y = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 6]
    assert M.kendall_tau_b(x, y) == pytest.approx(scipy.kendalltau(x, y).statistic)
    assert M.spearman_rho(x, y) == pytest.approx(scipy.spearmanr(x, y).statistic)


def test_consistency():
    assert M.symmetry_consistency([("before", "after"), ("before", "before"),
                                   ("cannot_determine", "cannot_determine")]) == pytest.approx(2 / 3)
    t = M.transitivity_consistency([
        {"ab": "before", "bc": "before", "ac": "after"},              # a<b<c<a : cycle
        {"ab": "before", "bc": "before", "ac": "before"},             # fine
        {"ab": "before", "bc": "cannot_determine", "ac": "after"},    # incomplete, not a cycle
    ])
    assert t["consistency_all"] == pytest.approx(2 / 3)
    assert t["consistency_committed"] == pytest.approx(0.5)


def test_mcnemar_and_bootstrap():
    a = [True] * 30 + [False] * 5
    b = [False] * 20 + [True] * 10 + [False] * 5
    r = M.mcnemar_exact(a, b)
    assert r["a_only"] == 20 and r["b_only"] == 0 and r["p_value"] < 1e-4
    scipy = pytest.importorskip("scipy.stats")
    assert r["p_value"] == pytest.approx(scipy.binomtest(0, 20, 0.5).pvalue)
    ci = M.bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], n=2000)
    assert ci["lo"] <= ci["value"] <= ci["hi"]


def test_aggregate_end_to_end():
    items = []
    for i, (g, p, s) in enumerate([("before", "before", "aligned"), ("after", "before", "inverted"),
                                   ("before", "before", "inverted"),
                                   ("cannot_determine", "after", "unordered")]):
        items.append(M.ItemRecord(
            question_id=f"q{i}", qtype="order", stratum=s, gold_label=g, pred_label=p,
            confidence=0.9 - i * 0.1, gold_evidence=GOLD,
            retrieved=[U(1, (100, 200)), U(2, (500, 560))], cited_spans=[(100, 200)],
            latency_ms=100 * (i + 1), prompt_tokens=1000, completion_tokens=100, pair_key=f"q{i}"))
    items.append(M.ItemRecord(question_id="q0__rev", qtype="probe", stratum=None, gold_label=None,
                              pred_label="after", confidence=None, gold_evidence=[], retrieved=[],
                              cited_spans=[], latency_ms=50, prompt_tokens=0, completion_tokens=0,
                              pair_key="q0", reversed_=True))
    s = M.aggregate(items, ks=(1, 2), budgets=(), n_boot=500)
    assert s["retrieval"]["recall@2"]["value"] == 1.0
    assert s["retrieval"]["pair_recall@1"]["value"] == 0.0
    assert s["ordering"]["overall"]["accuracy"] == pytest.approx(0.5)
    assert s["ordering"]["inverted"]["accuracy"] == pytest.approx(0.5)
    assert s["ordering"]["unordered"]["made_up_order_rate"] == 1.0
    assert s["consistency"]["symmetry"] == 1.0
    assert s["retrieval"]["citation_recall"]["value"] == 0.5
