"""
Gold-set proposal: a long-context model that is NOT the pipeline model reads the whole
document and proposes events + ordering questions with verbatim evidence. Code then
(1) verifies every quote against the text, (2) assigns the stratum deterministically
from where the quotes sit, and (3) exports a CSV for humans to confirm labels only.

  python -m app.gold_propose propose --doc DOC_ID --n 120 --out data/gold/sabha_proposed.csv
  # humans fill the `confirm` column: Y, N, or a corrected label (before/after/cannot_determine)
  python -m app.gold_propose import --doc DOC_ID --csv data/gold/sabha_proposed.csv \
                                    --out data/gold/sabha_v1.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import docstore, llm
from .config import settings
from .eval_runner import ORDER_TEMPLATE
from .ingest_v2 import Document, locate_quote

PROPOSE_SYSTEM = """You are building a test set for a system that answers questions about the
ORDER OF EVENTS in a book. You will read the entire book text.

1. List events (id E1, E2, ...) that are clearly narrated, each with a short neutral
   description (no page numbers) and ONE verbatim quote of 8-30 words that states it.
   Prefer quotes that are distinctive (they must occur only once in the book).
2. Propose ordering questions between pairs of these events, with a label for the FIRST
   event relative to the SECOND: before, after, or cannot_determine.
   Aim for a mix:
   - pairs where the book TELLS them in a different order from when they HAPPENED
     (flashbacks, backstories told later, prophecies or vows fulfilled later);
   - pairs told in the order they happened;
   - pairs the book does NOT order (parallel or unrelated events, separate stories):
     label cannot_determine.
   The label must follow from the text of THIS book alone, not outside knowledge.
   For every question give a one-line justification pointing to the text.
3. Also propose a few simple factual questions (who/what/where) with a verbatim quote,
   as a non-temporal control."""


class PEvent(BaseModel):
    id: str
    desc: str
    quote: str


class PQuestion(BaseModel):
    a: str
    b: str
    label: Literal["before", "after", "cannot_determine"]
    justification: str


class PFactual(BaseModel):
    question: str
    answer: str
    quote: str


class GoldProposal(BaseModel):
    events: list[PEvent]
    questions: list[PQuestion]
    factual: list[PFactual] = Field(description="about 10% as many as ordering questions")


def _stratum(label: str, pos_a: int, pos_b: int) -> str:
    if label == "cannot_determine":
        return "unordered"
    told_a_first = pos_a < pos_b
    return "aligned" if (label == "before") == told_a_first else "inverted"


def propose(doc_id: str, n: int, out_csv: Path) -> dict:
    doc: Document = docstore.load_document(doc_id)
    body = "\n\n".join(doc.text[p.start:p.end] for p in doc.paragraphs)
    user = (f"Target: about {n} ordering questions.\n\nBOOK TEXT:\n\n{body}")
    prop = llm.chat_structured(PROPOSE_SYSTEM, user, GoldProposal, temperature=0.2,
                               max_tokens=16000, deployment=settings.azure_gold_deployment,
                               max_input_chars=settings.gold_max_input_chars)
    loc, rejected = {}, []
    for e in prop.events:
        span = locate_quote(doc.text, e.quote)
        if span is None:
            rejected.append({"event": e.id, "why": "quote not found exactly once"})
        else:
            loc[e.id] = (e, span)
    rows = []
    for i, q in enumerate(prop.questions, start=1):
        if q.a not in loc or q.b not in loc or q.a == q.b:
            rejected.append({"question": i, "why": "event missing or unverified"})
            continue
        (ea, sa), (eb, sb) = loc[q.a], loc[q.b]
        rows.append({
            "id": f"q{i:03d}", "stratum": _stratum(q.label, sa[0], sb[0]),
            "question": ORDER_TEMPLATE.format(a=ea.desc, b=eb.desc),
            "proposed_label": q.label, "justification": q.justification,
            "event_a_id": q.a, "event_a": ea.desc, "quote_a": ea.quote,
            "pages_a": ",".join(map(str, doc.pages_for_span(*sa))),
            "event_b_id": q.b, "event_b": eb.desc, "quote_b": eb.quote,
            "pages_b": ",".join(map(str, doc.pages_for_span(*sb))),
            "confirm": "", "verified_by": "", "notes": ""})
    for j, f in enumerate(prop.factual, start=1):
        span = locate_quote(doc.text, f.quote)
        if span is None:
            rejected.append({"factual": j, "why": "quote not found exactly once"})
            continue
        rows.append({"id": f"f{j:03d}", "stratum": "", "question": f.question,
                     "proposed_label": "", "justification": f.answer, "event_a_id": "",
                     "event_a": "", "quote_a": f.quote,
                     "pages_a": ",".join(map(str, doc.pages_for_span(*span))),
                     "event_b_id": "", "event_b": "", "quote_b": "", "pages_b": "",
                     "confirm": "", "verified_by": "", "notes": ""})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    report = {"proposed_events": len(prop.events), "proposed_questions": len(prop.questions),
              "exported_rows": len(rows), "rejected": rejected,
              "strata": {s: sum(r["stratum"] == s for r in rows)
                         for s in ("aligned", "inverted", "unordered")},
              "deployment": settings.azure_gold_deployment}
    out_csv.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rejected"}, indent=2))
    return report


def import_confirmed(doc_id: str, csv_path: Path, out_jsonl: Path) -> dict:
    """Keep rows a human confirmed (Y) or relabelled; recompute stratum for relabels."""
    doc = docstore.load_document(doc_id)
    kept = dropped = 0
    lines = []
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            c = (r.get("confirm") or "").strip().lower()
            if c in ("", "n", "no"):
                dropped += 1
                continue
            if r["id"].startswith("f"):
                span = locate_quote(doc.text, r["quote_a"])
                lines.append({"id": r["id"], "qtype": "factual", "question": r["question"],
                              "gold_answer": r["justification"],
                              "evidence": [{"group": None, "quote": r["quote_a"],
                                            "char_start": span[0], "char_end": span[1]}],
                              "verified_by": r.get("verified_by") or None})
                kept += 1
                continue
            label = r["proposed_label"] if c in ("y", "yes") else c
            if label not in ("before", "after", "cannot_determine"):
                dropped += 1
                continue
            sa, sb = locate_quote(doc.text, r["quote_a"]), locate_quote(doc.text, r["quote_b"])
            lines.append({
                "id": r["id"], "qtype": "order", "stratum": _stratum(label, sa[0], sb[0]),
                "question": r["question"], "gold_label": label,
                "events": {"A": {"id": r["event_a_id"], "desc": r["event_a"]},
                           "B": {"id": r["event_b_id"], "desc": r["event_b"]}},
                "evidence": [{"group": "A", "quote": r["quote_a"], "char_start": sa[0],
                              "char_end": sa[1]},
                             {"group": "B", "quote": r["quote_b"], "char_start": sb[0],
                              "char_end": sb[1]}],
                "verified_by": r.get("verified_by") or None})
            kept += 1
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
                         encoding="utf-8")
    print(f"kept {kept}, dropped {dropped} -> {out_jsonl}")
    return {"kept": kept, "dropped": dropped}


def main():
    ap = argparse.ArgumentParser(prog="python -m app.gold_propose")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--doc", required=True)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("import")
    i.add_argument("--doc", required=True)
    i.add_argument("--csv", type=Path, required=True)
    i.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "propose":
        propose(a.doc, a.n, a.out)
    else:
        import_confirmed(a.doc, a.csv, a.out)


if __name__ == "__main__":
    main()
