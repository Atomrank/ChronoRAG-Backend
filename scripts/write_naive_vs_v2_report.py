"""Emit docs/results/synth_s1/naive_vs_v2.md from known run ids."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import eval_runner as E
from app.config import settings

NAIVE = "run_20260924_232553_naive_765a1c"
V2 = "run_20260924_231311_kaalkram_v2_a51196"
OUT = ROOT / "docs" / "results" / "synth_s1"


def head(run_id: str) -> dict:
    s = json.loads((settings.eval_path / run_id / "summary.json").read_text())
    r0 = s["repeats"]["0"]
    o, ret = r0["ordering"], r0["retrieval"]

    def row(st):
        x = o.get(st) or {}
        return {
            "accuracy": x.get("accuracy"),
            "coverage": x.get("coverage"),
            "selective_accuracy": x.get("selective_accuracy"),
            "n": x.get("n"),
        }

    return {
        "run_id": run_id,
        "errors": s.get("errors"),
        "overall": row("overall"),
        "aligned": row("aligned"),
        "inverted": row("inverted"),
        "unordered": row("unordered"),
        "hit@5": (ret.get("hit@5") or {}).get("value"),
        "mrr": (ret.get("mrr") or {}).get("value"),
        "ndcg@10": (ret.get("ndcg@10") or {}).get("value"),
        "recall@5": (ret.get("recall@5") or {}).get("value"),
    }


def f(x):
    return f"{x:.3f}" if isinstance(x, float) else ""


def main() -> None:
    a, b = head(NAIVE), head(V2)
    ids_path = OUT / "rerun_ids.json"
    ids = json.loads(ids_path.read_text()) if ids_path.is_file() else {"doc_id": "doc_b281c9bd70bedd34"}
    ids["naive"] = NAIVE
    ids["v2_llm"] = V2
    ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")

    lines = [
        "# Synth seed-1: naive vs kaalkram_v2 (llm extractor)",
        "",
        "gold: `synth_s1.jsonl` (cleaned questions, same for both)",
        f"naive: `{a['run_id']}`",
        f"v2_llm: `{b['run_id']}`",
        "",
        "| metric | naive | v2_llm |",
        "|---|---:|---:|",
    ]
    for name, na, vb in [
        ("overall acc", a["overall"]["accuracy"], b["overall"]["accuracy"]),
        ("overall coverage", a["overall"]["coverage"], b["overall"]["coverage"]),
        ("selective acc", a["overall"]["selective_accuracy"], b["overall"]["selective_accuracy"]),
        ("aligned acc", a["aligned"]["accuracy"], b["aligned"]["accuracy"]),
        ("inverted acc", a["inverted"]["accuracy"], b["inverted"]["accuracy"]),
        ("unordered acc", a["unordered"]["accuracy"], b["unordered"]["accuracy"]),
        ("aligned cov", a["aligned"]["coverage"], b["aligned"]["coverage"]),
        ("inverted cov", a["inverted"]["coverage"], b["inverted"]["coverage"]),
        ("unordered cov", a["unordered"]["coverage"], b["unordered"]["coverage"]),
        ("hit@5", a["hit@5"], b["hit@5"]),
        ("mrr", a["mrr"], b["mrr"]),
        ("recall@5", a["recall@5"], b["recall@5"]),
        ("ndcg@10", a["ndcg@10"], b["ndcg@10"]),
    ]:
        lines.append(f"| {name} | {f(na)} | {f(vb)} |")

    cmp_ = E.compare(V2, NAIVE)
    lines += [
        "",
        "## Paired compare (v2_llm as A, naive as B)",
        "",
        "```json",
        json.dumps(cmp_, indent=2),
        "```",
        "",
    ]
    out = OUT / "naive_vs_v2.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
