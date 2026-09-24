"""
Rerun extraction-only on the synth doc and dump per-call LLM telemetry.

  python -u scripts/extract_instrumentation.py [--doc DOC_ID]

Clears v1 pass1/pass2 and v2 window extract caches for the doc, runs:
  - v2 extract.extract_document
  - v1 extract_windows_for_doc + pass1 (+ pass2 on those observations)

Writes:
  docs/results/extract_instrumentation_synth.md
  docs/results/extract_instrumentation_synth.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import buildlog, llm
from app.config import settings
from app.db import pg
from app.docstore import load_document
from app.ingest import extract_windows_for_doc
from app import passes
from app.v2 import extract as v2_extract

DOC_DEFAULT = "doc_b281c9bd70bedd34"
OUT_MD = ROOT / "docs" / "results" / "extract_instrumentation_synth.md"
OUT_JSONL = ROOT / "docs" / "results" / "extract_instrumentation_synth.jsonl"


def _clear_extract_caches(doc_id: str) -> list[str]:
    removed = []
    for p in settings.cache_path.glob(f"{doc_id}_v2_*.json"):
        p.unlink(missing_ok=True)
        removed.append(p.name)
    for name in ("pass1", "pass2"):
        p = settings.cache_path / f"{doc_id}_{name}.json"
        if p.exists():
            p.unlink()
            removed.append(p.name)
    return removed


def _pdf_for_doc(doc_id: str) -> Path | None:
    candidates = [
        ROOT / "data" / "synth" / "s1" / "synth.pdf",
        ROOT / "data" / "uploads" / f"{doc_id}.pdf",
    ]
    try:
        with pg() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='documents'")
            cols = {r["column_name"] for r in cur.fetchall()}
            if "source_path" in cols:
                cur.execute("SELECT source_path FROM documents WHERE id = %s", (doc_id,))
                row = cur.fetchone()
                if row and row.get("source_path"):
                    candidates.insert(0, Path(row["source_path"]))
            elif "path" in cols:
                cur.execute("SELECT path FROM documents WHERE id = %s", (doc_id,))
                row = cur.fetchone()
                if row and row.get("path"):
                    candidates.insert(0, Path(row["path"]))
    except Exception as exc:
        print(f"  (doc path lookup skipped: {exc})", flush=True)
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def _summarize(events: list[dict]) -> str:
    calls = [e for e in events if e.get("kind") == "extract_llm_call"]
    trunc = [e for e in events if e.get("kind") == "extract_output_truncated"]
    by_pipe = Counter(c.get("pipeline") for c in calls)
    by_phase = Counter((c.get("pipeline"), c.get("phase")) for c in calls)
    fr = Counter(c.get("finish_reason") for c in calls)
    salvaged = sum(1 for c in calls if c.get("json_salvaged"))
    lines = [
        "# Extract instrumentation (synth)",
        "",
        f"doc_id: `{DOC}`",
        f"extract_llm_call count: **{len(calls)}**",
        f"extract_output_truncated count: **{len(trunc)}**",
        f"json_salvaged count: **{salvaged}**",
        "",
        "## finish_reason histogram",
        "",
    ]
    for k, n in fr.most_common():
        lines.append(f"- `{k}`: {n}")
    lines += ["", "## by pipeline / phase", ""]
    for (pipe, phase), n in sorted(by_phase.items(), key=lambda x: str(x[0])):
        lines.append(f"- `{pipe}` / `{phase}`: {n}")
    lines += ["", "## Truncation verdict", ""]
    if trunc:
        lines.append(
            f"**CONFIRMED**: {len(trunc)} call(s) hit finish_reason length/max_tokens "
            "(hard error; partial output not accepted)."
        )
    else:
        lines.append(
            "**REFUTED as the sole cause for this run**: no call returned "
            "finish_reason length/max_tokens. Under-recall may still come from "
            "salience filtering, window size, or model omission."
        )
    lines += ["", "## Per-call table", "",
              "| pipe | phase | win | start-end | in_chars | max_tok | finish | "
              "out_tok | n_items | salvage |",
              "|------|-------|-----|-----------|----------|---------|--------|"
              "---------|---------|---------|"]
    for c in calls:
        lines.append(
            f"| {c.get('pipeline')} | {c.get('phase')} | {c.get('window_index')} | "
            f"{c.get('window_start')}-{c.get('window_end')} | {c.get('input_chars')} | "
            f"{c.get('max_tokens')} | {c.get('finish_reason')} | "
            f"{c.get('completion_tokens')} | {c.get('n_items')} | {c.get('json_salvaged')} |"
        )
    if trunc:
        lines += ["", "## Truncated calls (errors)", ""]
        for t in trunc:
            lines.append(f"- `{json.dumps(t, default=str)[:500]}`")
    lines.append("")
    return "\n".join(lines)


DOC = DOC_DEFAULT


def main() -> None:
    global DOC
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=DOC_DEFAULT)
    ap.add_argument("--skip-v1", action="store_true")
    ap.add_argument("--skip-v2", action="store_true")
    args = ap.parse_args()
    DOC = args.doc

    print(f"clearing extract caches for {DOC} ...", flush=True)
    removed = _clear_extract_caches(DOC)
    print(f"  removed {len(removed)} files", flush=True)

    llm.drain_events()
    all_events: list[dict] = []

    if not args.skip_v2:
        print("=== v2 extract_document ===", flush=True)
        doc = load_document(DOC)

        def prog(p, msg):
            print(f"  [{p:.0%}] {msg}", flush=True)

        result = v2_extract.extract_document(doc, DOC, n_samples=1, on_progress=prog)
        ev = llm.drain_events()
        all_events.extend(ev)
        buildlog.flush_llm_events(DOC, job_id=None)
        print(f"  mentions={len(result.mentions)} frames={len(result.frames)} "
              f"windows={len(result.windows)} llm_events={len(ev)}", flush=True)

    if not args.skip_v1:
        print("=== v1 pass1 + pass2 ===", flush=True)
        doc = load_document(DOC)
        windows = extract_windows_for_doc(doc)
        print(f"  extract_windows={len(windows)} "
              f"(max_paras={settings.v2_window_max_paras} "
              f"max_chars={settings.v2_window_chars} "
              f"max_tokens={settings.extraction_max_tokens()})", flush=True)

        def prog1(p, msg):
            print(f"  [pass1 {p:.0%}] {msg}", flush=True)

        obs = passes.run_pass1(DOC, windows, on_progress=prog1)
        ev1 = llm.drain_events()
        all_events.extend(ev1)
        buildlog.flush_llm_events(DOC)
        print(f"  pass1 observations={len(obs)} llm_events={len(ev1)}", flush=True)

        # Minimal taxonomy so pass2 can run without inventing stages via LLM.
        tax = [{"name": "main", "description": "story events", "is_framing": False}]
        passes.save_taxonomy(DOC, tax)

        def prog2(p, msg):
            print(f"  [pass2 {p:.0%}] {msg}", flush=True)

        events = passes.run_pass2(DOC, obs, taxonomy=tax, on_progress=prog2)
        ev2 = llm.drain_events()
        all_events.extend(ev2)
        buildlog.flush_llm_events(DOC)
        print(f"  pass2 events={len(events)} llm_events={len(ev2)}", flush=True)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text(
        "\n".join(json.dumps(e, default=str) for e in all_events) + ("\n" if all_events else ""),
        encoding="utf-8",
    )
    report = _summarize(all_events)
    OUT_MD.write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"wrote {OUT_MD}", flush=True)
    trunc_n = sum(1 for e in all_events if e.get("kind") == "extract_output_truncated")
    print(f"SUMMARY_ERRORS_TRUNCATION={trunc_n}", flush=True)


if __name__ == "__main__":
    main()
