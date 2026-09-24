"""Naive rebuild + eval on synth seed-1 (same gold as step 7) for v2 head-to-head."""
from __future__ import annotations

import json
import time
from pathlib import Path

from app import eval_runner as E
from app import jobs
from app.config import settings
from app.db import pool

DOC = "doc_b281c9bd70bedd34"
GOLD = Path(__file__).resolve().parents[1] / "data" / "gold" / "synth_s1.jsonl"
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "synth_s1"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    pool()
    job_id = jobs.create(DOC, "naive")
    log(f"start naive build {job_id}")
    jobs.run_naive(job_id, DOC)
    row = jobs.get(job_id)
    log(f"naive build -> {row['status']} err={row.get('error')}")
    if row["status"] != "done":
        raise SystemExit(row)

    log("eval naive ...")
    rid = E.run(DOC, GOLD, "naive", repeats=1, n_triples=0, workers=4, use_db=True)
    log(f"eval naive -> {rid}")

    ids_path = OUT / "rerun_ids.json"
    ids = json.loads(ids_path.read_text()) if ids_path.is_file() else {"doc_id": DOC}
    ids["naive"] = rid
    ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")

    # head-to-head md if v2_llm present
    v2 = ids.get("v2_llm")
    if v2:
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

        a, b = head(rid), head(v2)
        lines = [
            "# Synth seed-1: naive vs kaalkram_v2 (llm extractor)",
            "",
            f"gold: `{GOLD.name}` (cleaned questions, same for both)",
            f"naive: `{a['run_id']}`",
            f"v2_llm: `{b['run_id']}`",
            "",
            "| metric | naive | v2_llm |",
            "|---|---:|---:|",
        ]
        def f(x):
            return f"{x:.3f}" if isinstance(x, float) else ""
        rows = [
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
        ]
        for name, na, vb in rows:
            lines.append(f"| {name} | {f(na)} | {f(vb)} |")
        # paired compare
        cmp_ = E.compare(v2, rid)
        lines += ["", "## Paired compare (v2_llm vs naive)", "",
                  f"```json", json.dumps(cmp_, indent=2), "```", ""]
        out = OUT / "naive_vs_v2.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        log(f"wrote {out}")
        print(json.dumps({"naive": a, "v2_llm": b, "compare": cmp_}, indent=2, default=str))


if __name__ == "__main__":
    main()
