"""
Automatic "silver" gold set — no human labelling.

  python -m app.gold_auto --doc DOC_ID --n 150 --out data/gold/sabha_silver.jsonl

1. PROPOSE candidate events (verbatim quotes) and candidate pairs.
   - One call if the document fits `gold_max_input_chars`;
   - otherwise two passes: events per paragraph-aligned chunk, then pairs chosen from
     the compact event list. The proposer's own labels are NOT used.
2. VERIFY every quote with locate_quote (exact, once) — misquotes are dropped.
3. JUDGE each pair with every model in `gold_judges`, each asked twice with A/B swapped
   (catches position bias), from the two passages only.
4. KEEP a pair only if every answer agrees:
   - before/after: all judges, both orders, same answer;
   - cannot_determine: additionally confirmed by `gold_fulltext_judge`, which reads the
     whole document (local passages cannot prove that the text never links two events).
     Without a full-text judge, cannot_determine pairs are dropped and the report says so.
5. Stratum (aligned / inverted / unordered) is computed from quote positions, not by a model.

Outputs the gold JSONL, a report (counts, drop reasons, per-judge agreement, pairwise
Cohen's kappa between judges) and `spot_check.csv` (30 random kept rows) for anyone who
later wants to estimate the silver labels' accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import docstore, llm
from .config import settings
from .eval_runner import ORDER_TEMPLATE
from .gold_propose import PROPOSE_SYSTEM, GoldProposal, PEvent, PFactual, _stratum
from .ingest_v2 import Document, locate_quote

PASSAGE_CHARS = 1500

EVENTS_SYSTEM = """You read one chunk of a book and list events that are clearly narrated in it.
For each event give an id (C<chunk>-E<n>), a short neutral description, and ONE verbatim quote
of 8-30 words that states the event and is likely to occur only once in the book.
Include events that are recalled or told by characters, and events foretold, as well as events
narrated as happening. Also give 1-2 simple factual questions (who/what/where) about this chunk,
each with its answer and a verbatim quote."""

PAIRS_SYSTEM = """You are given a list of events from a book, in the order the book tells them,
each with a short description and the narration mode hint in its description. Choose pairs of
events for a test of ORDER-OF-EVENTS questions. Aim for a mix of:
- pairs where the book probably tells them in a different order than they happened
  (backstories told later, recollections, prophecies or vows fulfilled later);
- pairs told in the order they happened;
- pairs the book probably does not order at all (separate stories, parallel events).
Use only the ids given. Do not label them; another step will."""

JUDGE_SYSTEM = """You judge the order of two events in a book, using ONLY the two passages given.
Answer for the FIRST event relative to the SECOND:
- before / after: the passages state or clearly imply the order (using only common sense that
  holds in any story: causes precede effects, people are born before they act);
- cannot_determine: the passages do not settle the order;
- invalid: a passage does not actually describe its event, the description is ambiguous, or the
  event is only foretold/imagined and never happens in these passages.
Do not use any knowledge of the book beyond the passages."""

FULLTEXT_SYSTEM = """You judge the order of two events using the WHOLE book text given. Answer for
the FIRST event relative to the SECOND: before, after, cannot_determine (the book never states or
implies their order, e.g. separate stories or parallel events with no link), or invalid.
Use only this text and common sense that holds in any story; no outside knowledge of the book."""


class ChunkEvents(BaseModel):
    events: list[PEvent]
    factual: list[PFactual]


class Pair(BaseModel):
    a: str
    b: str
    intent: Literal["told_out_of_order", "told_in_order", "not_ordered"]


class PairList(BaseModel):
    pairs: list[Pair]


class Judgement(BaseModel):
    label: Literal["before", "after", "cannot_determine", "invalid"]
    reason: str = Field(description="One sentence citing the passage wording")


INVERT = {"before": "after", "after": "before", "cannot_determine": "cannot_determine",
          "invalid": "invalid"}


# ============================================================
# 1-2. propose + verify
# ============================================================
def _chunks(doc: Document, max_chars: int) -> list[str]:
    out, cur, size = [], [], 0
    for p in doc.paragraphs:
        t = doc.text[p.start:p.end]
        if cur and size + len(t) > max_chars:
            out.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(t)
        size += len(t) + 2
    if cur:
        out.append("\n\n".join(cur))
    return out


def propose_candidates(doc: Document, n: int) -> tuple[dict, list[Pair], list[PFactual], dict]:
    """Returns (verified events {id: (PEvent, span)}, pairs, factual, info)."""
    body = "\n\n".join(doc.text[p.start:p.end] for p in doc.paragraphs)
    events: list[PEvent] = []
    factual: list[PFactual] = []
    pairs: list[Pair] = []
    info = {"mode": None, "gold_model": settings.azure_gold_deployment}
    if len(body) <= settings.gold_max_input_chars:
        info["mode"] = "single_call"
        prop: GoldProposal = llm.chat_structured(
            PROPOSE_SYSTEM, f"Target: about {n} ordering questions.\n\nBOOK TEXT:\n\n{body}",
            GoldProposal, temperature=0.2, max_tokens=16000,
            deployment=settings.azure_gold_deployment,
            max_input_chars=settings.gold_max_input_chars)
        events, factual = prop.events, prop.factual
        intent = {"before": "told_in_order", "after": "told_out_of_order",
                  "cannot_determine": "not_ordered"}
        pairs = [Pair(a=q.a, b=q.b, intent=intent[q.label]) for q in prop.questions]
    else:
        info["mode"] = "two_pass"
        chunk_max = max(20_000, settings.gold_max_input_chars // 3)
        chunks = _chunks(doc, chunk_max)
        info["chunks"] = len(chunks)
        for ci, ch in enumerate(chunks, start=1):
            res: ChunkEvents = llm.chat_structured(
                EVENTS_SYSTEM, f"CHUNK {ci} of {len(chunks)}:\n\n{ch}", ChunkEvents,
                temperature=0.2, max_tokens=8000, deployment=settings.azure_gold_deployment)
            events += res.events
            factual += res.factual
    verified: dict[str, tuple[PEvent, tuple[int, int]]] = {}
    rejected = []
    for e in events:
        span = locate_quote(doc.text, e.quote)
        if span is None:
            rejected.append(e.id)
        else:
            verified[e.id] = (e, span)
    info["events_proposed"] = len(events)
    info["events_verified"] = len(verified)
    info["events_rejected_quote"] = len(rejected)
    if info["mode"] == "two_pass":
        ordered = sorted(verified.values(), key=lambda x: x[1][0])
        listing = "\n".join(f"{e.id}: {e.desc}" for e, _ in ordered)
        res: PairList = llm.chat_structured(
            PAIRS_SYSTEM, f"Choose about {n} pairs.\n\nEVENTS IN BOOK ORDER:\n{listing}",
            PairList, temperature=0.2, max_tokens=12000, deployment=settings.azure_gold_deployment)
        pairs = res.pairs
    pairs = [p for p in pairs if p.a in verified and p.b in verified and p.a != p.b]
    seen, uniq = set(), []
    for p in pairs:
        k = frozenset((p.a, p.b))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    info["pairs_candidate"] = len(uniq)
    return verified, uniq, factual, info


# ============================================================
# 3-4. judge + consensus
# ============================================================
def _passage(doc: Document, span: tuple[int, int]) -> str:
    s = max(0, span[0] - PASSAGE_CHARS)
    e = min(len(doc.text), span[1] + PASSAGE_CHARS)
    return doc.text[s:e]


def _ask(judge: str, system: str, user: str) -> str:
    try:
        return llm.chat_structured(system, user, Judgement, temperature=0.0, max_tokens=300,
                                   deployment=judge,
                                   max_input_chars=settings.gold_max_input_chars).label
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def judge_pair(doc: Document, ea, sa, eb, sb, judges: list[str]) -> dict:
    votes = {}
    for j in judges:
        u1 = (f"FIRST EVENT: {ea.desc}\nPASSAGE 1:\n{_passage(doc, sa)}\n\n"
              f"SECOND EVENT: {eb.desc}\nPASSAGE 2:\n{_passage(doc, sb)}")
        u2 = (f"FIRST EVENT: {eb.desc}\nPASSAGE 1:\n{_passage(doc, sb)}\n\n"
              f"SECOND EVENT: {ea.desc}\nPASSAGE 2:\n{_passage(doc, sa)}")
        votes[j] = (_ask(j, JUDGE_SYSTEM, u1), INVERT.get(_ask(j, JUDGE_SYSTEM, u2), "error"))
    return votes


def consensus(votes: dict[str, tuple[str, str]]) -> tuple[str | None, str]:
    answers = [a for pair in votes.values() for a in pair]
    if any(a.startswith("error") for a in answers):
        return None, "judge_error"
    if any(a == "invalid" for a in answers):
        return None, "invalid"
    for v in votes.values():
        if v[0] != v[1]:
            return None, "position_inconsistent"
    if len(set(answers)) != 1:
        return None, "judges_disagree"
    return answers[0], "agreed"


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ============================================================
# run
# ============================================================
def build(doc_id: str, n: int, out: Path, seed: int = 13) -> dict:
    doc = docstore.load_document(doc_id)
    judges = [j.strip() for j in settings.gold_judges.split(",") if j.strip()]
    if not judges:
        raise SystemExit("GOLD_JUDGES is empty")
    verified, pairs, factual, info = propose_candidates(doc, n)

    rows, drops = [], Counter()
    per_judge: dict[str, list[str]] = {j: [] for j in judges}
    fulltext = settings.gold_fulltext_judge.strip()
    body = doc.text
    for i, p in enumerate(pairs, start=1):
        (ea, sa), (eb, sb) = verified[p.a], verified[p.b]
        votes = judge_pair(doc, ea, sa, eb, sb, judges)
        for j, v in votes.items():
            per_judge[j].append(v[0])
        label, why = consensus(votes)
        if label == "cannot_determine":
            if not fulltext:
                label, why = None, "unordered_needs_fulltext_judge"
            else:
                ft = _ask(fulltext, FULLTEXT_SYSTEM,
                          f"FIRST EVENT: {ea.desc}\nSECOND EVENT: {eb.desc}\n\nBOOK TEXT:\n\n{body}")
                if ft != "cannot_determine":
                    label, why = None, f"fulltext_judge_says_{ft}"
        if label is None:
            drops[why] += 1
            continue
        rows.append({
            "id": f"q{i:03d}", "qtype": "order", "stratum": _stratum(label, sa[0], sb[0]),
            "question": ORDER_TEMPLATE.format(a=ea.desc, b=eb.desc), "gold_label": label,
            "events": {"A": {"id": ea.id, "desc": ea.desc}, "B": {"id": eb.id, "desc": eb.desc}},
            "evidence": [{"group": "A", "quote": ea.quote, "char_start": sa[0], "char_end": sa[1]},
                         {"group": "B", "quote": eb.quote, "char_start": sb[0], "char_end": sb[1]}],
            "verified_by": "auto:" + "+".join(judges + ([fulltext] if fulltext else [])),
        })
    for k, f in enumerate(factual, start=1):
        span = locate_quote(doc.text, f.quote)
        if span:
            rows.append({"id": f"f{k:03d}", "qtype": "factual", "question": f.question,
                         "gold_answer": f.answer,
                         "evidence": [{"group": None, "quote": f.quote, "char_start": span[0],
                                       "char_end": span[1]}],
                         "verified_by": "auto:quote_only"})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    order_rows = [r for r in rows if r["qtype"] == "order"]
    kappas = {}
    for x in range(len(judges)):
        for y in range(x + 1, len(judges)):
            kappas[f"{judges[x]}~{judges[y]}"] = cohen_kappa(per_judge[judges[x]],
                                                             per_judge[judges[y]])
    report = {
        **info, "judges": judges, "fulltext_judge": fulltext or None,
        "pairs_kept": len(order_rows), "factual_kept": len(rows) - len(order_rows),
        "dropped": dict(drops),
        "strata": dict(Counter(r["stratum"] for r in order_rows)),
        "labels": dict(Counter(r["gold_label"] for r in order_rows)),
        "judge_kappa": kappas,
        "same_model_as_pipeline": llm.model_id(None) in {
            llm.model_id(j) for j in judges + [settings.azure_gold_deployment]},
        "note": ("Silver labels: unanimous model agreement, not human-verified. "
                 "Estimate their accuracy with spot_check.csv if anyone can review it."),
    }
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    rng = random.Random(seed)
    sample = rng.sample(order_rows, min(30, len(order_rows)))
    with out.with_name(out.stem + "_spot_check.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "question", "gold_label", "quote_a", "quote_b",
                                           "reviewer_label"])
        w.writeheader()
        for r in sample:
            w.writerow({"id": r["id"], "question": r["question"], "gold_label": r["gold_label"],
                        "quote_a": r["evidence"][0]["quote"], "quote_b": r["evidence"][1]["quote"],
                        "reviewer_label": ""})
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(prog="python -m app.gold_auto")
    ap.add_argument("--doc", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a.doc, a.n, a.out)


if __name__ == "__main__":
    main()
