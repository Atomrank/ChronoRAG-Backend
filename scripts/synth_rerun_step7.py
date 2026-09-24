"""Step 7: rebuild synth (v1 + v2-llm + v2-oracle) and re-eval.

  python -u scripts/synth_rerun_step7.py

Writes run ids to docs/results/synth_s1/rerun_ids.json and appends a
scoreboard snippet to docs/results/synth_s1/rerun_scoreboard.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import jobs, llm  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import pool  # noqa: E402
from app import eval_runner as E  # noqa: E402

DOC = "doc_b281c9bd70bedd34"
GOLD = ROOT / "data" / "gold" / "synth_s1.jsonl"
OUT_DIR = ROOT / "docs" / "results" / "synth_s1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IDS_PATH = OUT_DIR / "rerun_ids.json"
LOG = OUT_DIR / "rerun.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def clear_extract_caches(doc_id: str) -> list[str]:
    removed = []
    for p in settings.cache_path.glob(f"{doc_id}_v2_*.json"):
        p.unlink(missing_ok=True)
        removed.append(p.name)
    for name in ("pass1", "pass2", "pass0"):
        p = settings.cache_path / f"{doc_id}_{name}.json"
        if p.exists():
            p.unlink()
            removed.append(p.name)
    return removed


def run_job(kind: str, doc_id: str, *, extractor: str | None = None) -> dict:
    if extractor is not None:
        settings.extractor = extractor
        os.environ["EXTRACTOR"] = extractor
    job_id = jobs.create(doc_id, kind)
    log(f"start job {job_id} kind={kind} extractor={settings.extractor}")
    runners = {
        "kaalkram": jobs.run_kaalkram,
        "kaalkram_v2": jobs.run_kaalkram_v2,
    }
    runners[kind](job_id, doc_id)
    row = jobs.get(job_id)
    log(f"job {job_id} -> {row['status']} stage={row['stage']} err={row.get('error')}")
    if row["status"] != "done":
        raise SystemExit(f"build failed: {row}")
    return row


def run_eval(pipeline: str) -> str:
    log(f"eval {pipeline} ...")
    rid = E.run(DOC, GOLD, pipeline, repeats=1, n_triples=0, workers=4, use_db=True)
    log(f"eval {pipeline} -> {rid}")
    return rid


def headline(rid: str) -> dict:
    s = json.loads((settings.eval_path / rid / "summary.json").read_text())["repeats"]["0"]
    o = s["ordering"]
    r = s["retrieval"]

    def acc(st):
        x = o.get(st) or {}
        return {
            "accuracy": x.get("accuracy"),
            "coverage": x.get("coverage"),
            "selective_accuracy": x.get("selective_accuracy"),
            "n": x.get("n"),
        }

    return {
        "run_id": rid,
        "n_items": s["n_items"],
        "errors": json.loads((settings.eval_path / rid / "summary.json").read_text()).get("errors"),
        "overall": acc("overall"),
        "aligned": acc("aligned"),
        "inverted": acc("inverted"),
        "unordered": acc("unordered"),
        "hit@5": (r.get("hit@5") or {}).get("value"),
        "mrr": (r.get("mrr") or {}).get("value"),
        "ndcg@10": (r.get("ndcg@10") or {}).get("value"),
        "recall@5": (r.get("recall@5") or {}).get("value"),
    }


def main() -> None:
    pool()
    if LOG.exists():
        LOG.unlink()
    log(f"doc={DOC} gold={GOLD} extraction_mode={settings.extraction_mode}")
    if not GOLD.is_file():
        raise SystemExit(f"missing gold: {GOLD}")

    removed = clear_extract_caches(DOC)
    log(f"cleared {len(removed)} cache files")

    ids: dict = {"doc_id": DOC, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # --- v1 build + eval ---
    settings.extractor = "llm"
    run_job("kaalkram", DOC, extractor="llm")
    ids["v1"] = run_eval("kaalkram_v1")

    # --- v2 llm build + eval ---
    llm.drain_events()
    removed = clear_extract_caches(DOC)
    log(f"cleared {len(removed)} cache files before v2-llm")
    run_job("kaalkram_v2", DOC, extractor="llm")
    ids["v2_llm"] = run_eval("kaalkram_v2")

    # --- v2 oracle build + eval (overwrites v2 graph for doc) ---
    llm.drain_events()
    removed = clear_extract_caches(DOC)
    log(f"cleared {len(removed)} cache files before v2-oracle")
    run_job("kaalkram_v2", DOC, extractor="oracle")
    ids["v2_oracle"] = run_eval("kaalkram_v2")
    # restore default extractor for any later processes
    settings.extractor = "llm"
    os.environ["EXTRACTOR"] = "llm"

    ids["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    board = {k: headline(v) for k, v in ids.items() if k.startswith("v")}
    ids["headlines"] = board
    IDS_PATH.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    log(f"wrote {IDS_PATH}")

    md = ["# Synth seed-1 rerun scoreboard", "",
          f"doc: `{DOC}`", "",
          "| pipeline | run_id | acc | cov | sel | aligned | inverted | unordered | hit@5 | ndcg@10 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("v1", "v2_llm", "v2_oracle"):
        h = board[name]
        o = h["overall"]
        def f(x):
            return f"{x:.3f}" if isinstance(x, float) else ""
        md.append(
            f"| {name} | `{h['run_id']}` | {f(o.get('accuracy'))} | {f(o.get('coverage'))} | "
            f"{f(o.get('selective_accuracy'))} | {f((h['aligned'] or {}).get('accuracy'))} | "
            f"{f((h['inverted'] or {}).get('accuracy'))} | {f((h['unordered'] or {}).get('accuracy'))} | "
            f"{f(h.get('hit@5'))} | {f(h.get('ndcg@10'))} |"
        )
    (OUT_DIR / "rerun_scoreboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    main()
