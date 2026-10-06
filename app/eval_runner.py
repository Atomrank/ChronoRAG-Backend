"""
Benchmark runner. Every number in a report comes from rows this writes.

  python -m app.eval_runner verify  --doc DOC_ID --gold data/gold/sabha_raw.jsonl \
                                    --out data/gold/sabha_v1.jsonl
  python -m app.eval_runner run     --doc DOC_ID --gold data/gold/sabha_v1.jsonl \
                                    --pipeline naive --repeats 3 --probes 40
  python -m app.eval_runner compare RUN_A RUN_B
  python -m app.eval_runner report  RUN_ID [RUN_ID ...]

Gold JSONL, one question per line (see docs/EVALUATION.md):
  {"id": "q001", "qtype": "order", "stratum": "inverted",
   "question": "Did X happen before or after Y?",
   "gold_label": "before",
   "events": {"A": {"id": "E1", "desc": "X"}, "B": {"id": "E2", "desc": "Y"}},
   "evidence": [{"group": "A", "quote": "..."}, {"group": "B", "quote": "..."}],
   "verified_by": "AT"}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel

from . import docstore, llm, metrics
from .config import ROOT, settings
from .textutil import scrub_llm_text

KS = (1, 3, 5, 8, 10, 20)
BUDGETS = (1000, 2000, 4000)
ORDER_TEMPLATE = "Did {a} happen before or after {b}?"   # generic English, no book content


# ============================================================
# Gold format
# ============================================================
class GoldEvidence(BaseModel):
    group: Literal["A", "B"] | None = None
    quote: str
    char_start: int | None = None
    char_end: int | None = None


class GoldEvent(BaseModel):
    id: str
    desc: str


class GoldQuestion(BaseModel):
    id: str
    qtype: Literal["order", "sequence", "before_after_x", "state", "factual"]
    stratum: Literal["aligned", "inverted", "unordered"] | None = None
    question: str
    gold_label: Literal["before", "after", "cannot_determine"] | None = None
    gold_answer: str | None = None
    gold_sequence: list[str] | None = None
    events: dict[str, GoldEvent] | None = None
    evidence: list[GoldEvidence] = []
    verified_by: str | None = None


def _sha1_file(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def load_gold(path: Path) -> list[GoldQuestion]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                out.append(GoldQuestion.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
    ids = [q.id for q in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in gold set")
    return out


def question_split(question_id: str, *, fraction: float | None = None) -> Literal["dev", "test"]:
    """Stable hash of question id -> 'dev' or 'test'.

    Fraction ``eval_dev_fraction`` (default 0.3) of ids land in the calibration
    (dev) split; the rest are held out as test. Same id always maps the same way.
    """
    frac = settings.eval_dev_fraction if fraction is None else fraction
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"eval_dev_fraction must be in [0, 1], got {frac}")
    # Uniform in [0, 1) from first 8 hex digits of sha1 — stable across runs/machines.
    h = int(hashlib.sha1(question_id.encode("utf-8")).hexdigest()[:8], 16)
    u = h / 0x100000000
    return "dev" if u < frac else "test"


def filter_by_split(questions: list[GoldQuestion],
                    split: Literal["all", "dev", "test"]) -> list[GoldQuestion]:
    if split == "all":
        return list(questions)
    return [q for q in questions if question_split(q.id) == split]


def verify_gold(doc_id: str, questions: list[GoldQuestion]) -> tuple[list[GoldQuestion], list[dict]]:
    """Locate every evidence quote in the document text. A question is kept only
    if ALL its quotes are found exactly once — misquoted gold is dropped, not guessed."""
    from .ingest_v2 import locate_quote
    doc = docstore.load_document(doc_id)
    kept, rejected = [], []
    for q in questions:
        bad = []
        for ev in q.evidence:
            loc = locate_quote(doc.text, ev.quote)
            if loc is None:
                bad.append(ev.quote[:80])
            else:
                ev.char_start, ev.char_end = loc
        if bad or (q.qtype == "order" and not q.evidence):
            rejected.append({"id": q.id, "unmatched_quotes": bad})
        else:
            kept.append(q)
    return kept, rejected


# ============================================================
# Consistency probes (no gold needed)
# ============================================================
def make_probes(questions: list[GoldQuestion], n_triples: int, seed: int = 13) -> list[dict]:
    """Reversed twins of every order question, plus random event triples asked
    pairwise. Only used for symmetry / transitivity consistency."""
    probes: list[dict] = []
    events: dict[str, str] = {}
    for q in questions:
        if q.qtype != "order" or not q.events or {"A", "B"} - set(q.events):
            continue
        a, b = q.events["A"], q.events["B"]
        da, db = scrub_llm_text(a.desc), scrub_llm_text(b.desc)
        events[a.id], events[b.id] = da, db
        probes.append({"id": f"{q.id}__rev", "question": ORDER_TEMPLATE.format(a=db, b=da),
                       "pair_key": q.id, "reversed": True})
    rng = random.Random(seed)
    ids = sorted(events)
    seen = set()
    for _ in range(n_triples * 20):
        if len(seen) >= n_triples or len(ids) < 3:
            break
        t = tuple(sorted(rng.sample(ids, 3)))
        if t in seen:
            continue
        seen.add(t)
        a, b, c = t
        key = "tri_" + "_".join(t)
        for slot, (x, y) in {"ab": (a, b), "bc": (b, c), "ac": (a, c)}.items():
            probes.append({"id": f"{key}__{slot}",
                           "question": ORDER_TEMPLATE.format(a=events[x], b=events[y]),
                           "triple_key": key, "triple_slot": slot})
    return probes


# ============================================================
# Pipelines
# ============================================================
def _adapters() -> dict[str, Callable]:
    from . import naive_rag, query_engine
    from .v2 import query as query_v2
    return {
        "naive": lambda doc_id, q, k: naive_rag.answer(doc_id, q, k_retrieve=k),
        "kaalkram_v1": lambda doc_id, q, k: query_engine.answer(doc_id, q, k_retrieve=k),
        "kaalkram_v2": lambda doc_id, q, k: query_v2.answer(doc_id, q, k_retrieve=k),
    }


def _git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain"))}


def _label(rel: str | None) -> tuple[str | None, str | None]:
    """Map pipeline relation → eval label. Returns (pred_label, error).

    `not_applicable` (non-order answers) scores as abstain. Unrecognised values
    become a counted error rather than a silent null pred_label.
    """
    if rel is None:
        return None, "missing_relation"
    if rel == "not_applicable":
        return "cannot_determine", None
    if rel in metrics.LABELS:
        return rel, None
    return None, f"unrecognised_relation:{rel}"


def _to_record(q: dict, ans: dict | None,
               err: str | None) -> tuple[metrics.ItemRecord, str | None]:
    """Build an ItemRecord; return (record, err) with label issues counted as errors."""
    ev = [{"span": (e["char_start"], e["char_end"]), "group": e.get("group")}
          for e in q.get("evidence", []) if e.get("char_start") is not None]
    ans = ans or {}
    pred: str | None = None
    if not err and ans:
        pred, label_err = _label(ans.get("relation"))
        if label_err:
            err = label_err
            pred = None
    rec = metrics.ItemRecord(
        question_id=q["id"], qtype=q.get("qtype", "probe"), stratum=q.get("stratum"),
        gold_label=q.get("gold_label"),
        pred_label=pred,
        confidence=ans.get("confidence"),
        gold_evidence=ev,
        retrieved=[{"rank": r["rank"], "spans": [tuple(s) for s in r.get("spans", [])],
                    "tokens": r.get("tokens")} for r in ans.get("retrieved", [])],
        cited_spans=[tuple(s) for s in ans.get("cited_spans", [])],
        latency_ms=ans.get("latency_ms", 0), prompt_tokens=ans.get("prompt_tokens", 0),
        completion_tokens=ans.get("completion_tokens", 0),
        gold_sequence=q.get("gold_sequence"),
        pair_key=q.get("pair_key") or (q["id"] if q.get("qtype") == "order" else None),
        triple_key=q.get("triple_key"), triple_slot=q.get("triple_slot"),
        reversed_=bool(q.get("reversed")),
    )
    return rec, err


# ============================================================
# Run
# ============================================================
def run(doc_id: str, gold_path: Path, pipeline: str, *, repeats: int = 1,
        n_triples: int = 0, workers: int = 4, k_max: int | None = None,
        use_db: bool = True,
        split: Literal["all", "dev", "test"] = "all") -> str:
    k_max = k_max or settings.eval_k_max
    questions = filter_by_split(load_gold(gold_path), split)
    if not questions:
        raise SystemExit(f"no questions left after --split {split}")
    missing = [q.id for q in questions for e in q.evidence if e.char_start is None]
    if missing:
        raise SystemExit(f"gold has unverified evidence ({len(missing)}); run `verify` first")
    items = [q.model_dump() for q in questions]
    probes = make_probes(questions, n_triples)      # reversed twins always; triples if asked
    work = items + probes

    doc = docstore.load_document(doc_id)
    gold_sha1 = _sha1_file(gold_path)
    gold_set_id = Path(gold_path).stem
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{pipeline}_{uuid.uuid4().hex[:6]}"
    git = _git_info()
    params = {"k_max": k_max, "ks": list(KS), "budgets": list(BUDGETS), "repeats": repeats,
              "n_triples": n_triples, "n_questions": len(items), "n_probes": len(probes),
              "temperature": 0.0, "workers": workers, "split": split,
              "eval_dev_fraction": settings.eval_dev_fraction,
              "thresholds": {"gold_cover_frac": metrics.GOLD_COVER_FRAC,
                             "unit_inside_frac": metrics.UNIT_INSIDE_FRAC}}
    out_dir = settings.eval_path / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"run_id": run_id, "pipeline": pipeline, "doc_id": doc_id,
              "doc_text_sha1": docstore.text_sha1(doc.text), "gold_set_id": gold_set_id,
              "gold_sha1": gold_sha1, "git": git, "settings": settings.public_dict(),
              "params": params}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    if use_db:
        _db_start(config, questions)

    fn = _adapters()[pipeline]

    def one(job):
        q, rep = job
        try:
            ans = fn(doc_id, q["question"], k_max).model_dump()
            return q, rep, ans, None
        except Exception as exc:
            return q, rep, None, f"{type(exc).__name__}: {exc}"

    jobs = [(q, r) for r in range(repeats) for q in work]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, res in enumerate(pool.map(one, jobs), start=1):
            results.append(res)
            if n % 10 == 0 or n == len(jobs):
                print(f"[{run_id}] {n}/{len(jobs)} answered", flush=True)

    records_by_rep: dict[int, list[metrics.ItemRecord]] = {}
    n_errors = 0
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for q, rep, ans, err in results:
            rec, err = _to_record(q, ans, err)
            if err:
                n_errors += 1
            records_by_rep.setdefault(rep, []).append(rec)
            per_q = metrics.retrieval_metrics(rec.retrieved, rec.gold_evidence, KS, BUDGETS)
            row = {"question_id": q["id"], "repeat_idx": rep, "question": q["question"],
                   "gold_label": q.get("gold_label"), "stratum": q.get("stratum"),
                   "answer": (ans or {}).get("answer"), "pred_label": rec.pred_label,
                   "confidence": rec.confidence, "retrieved": (ans or {}).get("retrieved", []),
                   "cited_spans": (ans or {}).get("cited_spans", []),
                   "latency_ms": rec.latency_ms, "prompt_tokens": rec.prompt_tokens,
                   "completion_tokens": rec.completion_tokens, "metrics": per_q, "error": err}
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            if use_db:
                _db_item(run_id, row)

    per_repeat = {rep: metrics.aggregate(recs, KS, BUDGETS) for rep, recs in records_by_rep.items()}
    summary = {"repeats": per_repeat,
               "repeat_spread": _repeat_spread(per_repeat),
               "errors": n_errors,
               "llm_events": llm.drain_events()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if use_db:
        _db_finish(run_id, summary)
    _mlflow(config, per_repeat.get(0, {}), out_dir)
    print(f"done: {run_id}  ->  {out_dir}")
    return run_id


def _repeat_spread(per_repeat: dict[int, dict]) -> dict:
    """Mean and range of headline metrics across repeats (LLM nondeterminism)."""
    def pick(s: dict) -> dict:
        o = s.get("ordering", {})
        r = s.get("retrieval", {})
        return {
            "accuracy": (o.get("overall") or {}).get("accuracy"),
            "inverted_accuracy": (o.get("inverted") or {}).get("accuracy"),
            "macro_f1": (o.get("overall") or {}).get("macro_f1"),
            "recall@5": (r.get("recall@5") or {}).get("value"),
            "pair_recall@5": (r.get("pair_recall@5") or {}).get("value"),
            "mrr": (r.get("mrr") or {}).get("value"),
        }
    rows = [pick(s) for s in per_repeat.values()]
    out = {}
    for k in rows[0] if rows else []:
        vals = [r[k] for r in rows if r[k] is not None]
        if vals:
            out[k] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals),
                      "n": len(vals)}
    return out


# ============================================================
# Persistence
# ============================================================
def _db_start(config: dict, questions: list[GoldQuestion]) -> None:
    from .db import pg
    with pg() as cur:
        cur.execute(
            """INSERT INTO eval_gold_sets (gold_set_id, doc_id, content_sha1, n_questions)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT (gold_set_id) DO UPDATE SET content_sha1 = EXCLUDED.content_sha1,
                 n_questions = EXCLUDED.n_questions""",
            (config["gold_set_id"], config["doc_id"], config["gold_sha1"], len(questions)))
        cur.execute("DELETE FROM eval_questions WHERE gold_set_id = %s", (config["gold_set_id"],))
        cur.executemany(
            """INSERT INTO eval_questions (gold_set_id, question_id, qtype, stratum, question,
                 gold_label, gold_answer, gold_sequence, evidence, verified_by)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(config["gold_set_id"], q.id, q.qtype, q.stratum, q.question, q.gold_label,
              q.gold_answer, json.dumps(q.gold_sequence),
              json.dumps([e.model_dump() for e in q.evidence]), q.verified_by)
             for q in questions])
        s = config["settings"]
        cur.execute(
            """INSERT INTO eval_runs (run_id, pipeline, doc_id, doc_text_sha1, gold_set_id,
                 gold_sha1, git_commit, git_dirty, chat_deployment, embed_deployment,
                 api_version, settings, params)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (config["run_id"], config["pipeline"], config["doc_id"], config["doc_text_sha1"],
             config["gold_set_id"], config["gold_sha1"], config["git"]["commit"],
             config["git"]["dirty"], llm.model_id(None),
             llm.active_models()["embed_model"], s.get("azure_openai_api_version"),
             json.dumps(s, default=str), json.dumps(config["params"])))


def _db_item(run_id: str, row: dict) -> None:
    from .db import pg
    with pg() as cur:
        cur.execute(
            """INSERT INTO eval_items (run_id, question_id, repeat_idx, answer, pred_label,
                 confidence, retrieved, cited_spans, latency_ms, prompt_tokens,
                 completion_tokens, metrics, error)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (run_id, row["question_id"], row["repeat_idx"], row["answer"], row["pred_label"],
             row["confidence"], json.dumps(row["retrieved"], default=str),
             json.dumps(row["cited_spans"]), row["latency_ms"], row["prompt_tokens"],
             row["completion_tokens"], json.dumps(row["metrics"]), row["error"]))


def _db_finish(run_id: str, summary: dict) -> None:
    from .db import pg
    with pg() as cur:
        cur.execute("UPDATE eval_runs SET summary = %s, status = 'done' WHERE run_id = %s",
                    (json.dumps(summary, default=str), run_id))


def _mlflow(config: dict, summary: dict, out_dir: Path) -> None:
    """Optional: log to a local MLflow server if mlflow is installed."""
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_experiment(f"kaalkram_{config['gold_set_id']}")
    with mlflow.start_run(run_name=config["run_id"]):
        mlflow.log_params({"pipeline": config["pipeline"], "git": config["git"]["commit"],
                           **{f"s.{k}": v for k, v in config["settings"].items()},
                           **{f"p.{k}": v for k, v in config["params"].items()
                              if not isinstance(v, (dict, list))}})
        for k, v in (summary.get("retrieval") or {}).items():
            if v.get("value") is not None:
                mlflow.log_metric(k.replace("@", "_at_"), v["value"])
        for stratum, rep in (summary.get("ordering") or {}).items():
            if isinstance(rep, dict) and rep.get("accuracy") is not None:
                mlflow.log_metric(f"acc_{stratum}", rep["accuracy"])
        mlflow.log_artifacts(str(out_dir))


# ============================================================
# Compare / report
# ============================================================
def _load_records(run_id: str, rep: int = 0) -> list[metrics.ItemRecord]:
    d = settings.eval_path / run_id
    recs = []
    for line in (d / "items.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["repeat_idx"] != rep:
            continue
        recs.append(metrics.ItemRecord(
            question_id=row["question_id"], qtype="", stratum=row.get("stratum"),
            gold_label=row.get("gold_label"), pred_label=row.get("pred_label"),
            confidence=row.get("confidence"), gold_evidence=[], retrieved=[], cited_spans=[],
            latency_ms=row.get("latency_ms") or 0, prompt_tokens=row.get("prompt_tokens") or 0,
            completion_tokens=row.get("completion_tokens") or 0,
            reversed_="__" in row["question_id"]))       # probes: twins and triples
    return recs


def compare(run_a: str, run_b: str) -> dict:
    out = metrics.compare_runs(_load_records(run_a), _load_records(run_b))
    print(json.dumps(out, indent=2))
    return out


def report(run_ids: list[str]) -> Path:
    """One row per run with the headline numbers; writes report.md + report.csv."""
    cols = ["pipeline", "n", "recall@1", "recall@5", "recall@10", "recall@20", "precision@5",
            "hit@5", "mrr", "ndcg@10", "pair_recall@5", "pair_recall@10", "recall@2000tok",
            "acc", "acc_aligned", "acc_inverted", "acc_unordered", "macro_f1",
            "coverage", "coverage_aligned", "coverage_inverted", "coverage_unordered",
            "selective_acc", "selective_acc_aligned", "selective_acc_inverted",
            "selective_acc_unordered", "made_up_order_rate", "symmetry", "transitivity",
            "citation_precision", "citation_recall", "latency_p50_ms", "latency_p95_ms",
            "tokens_per_q"]
    rows = []
    for rid in run_ids:
        d = settings.eval_path / rid
        cfg = json.loads((d / "config.json").read_text())
        s = json.loads((d / "summary.json").read_text())["repeats"]["0"]
        r, o, c = s["retrieval"], s["ordering"], s["consistency"]

        def rv(k):
            v = (r.get(k) or {}).get("value")
            lo, hi = (r.get(k) or {}).get("lo"), (r.get(k) or {}).get("hi")
            return f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if v is not None else ""

        def ov(stratum, key="accuracy"):
            v = (o.get(stratum) or {}).get(key)
            ci = (o.get(stratum) or {}).get("accuracy_ci") if key == "accuracy" else None
            if v is None:
                return ""
            return f"{v:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]" if ci else f"{v:.3f}"

        rows.append({
            "pipeline": cfg["pipeline"], "n": s["n_items"],
            **{k: rv(k) for k in ("recall@1", "recall@5", "recall@10", "recall@20",
                                  "precision@5", "hit@5", "mrr", "ndcg@10", "pair_recall@5",
                                  "pair_recall@10", "recall@2000tok", "citation_precision",
                                  "citation_recall")},
            "acc": ov("overall"), "acc_aligned": ov("aligned"), "acc_inverted": ov("inverted"),
            "acc_unordered": ov("unordered"), "macro_f1": ov("overall", "macro_f1"),
            "coverage": ov("overall", "coverage"),
            "coverage_aligned": ov("aligned", "coverage"),
            "coverage_inverted": ov("inverted", "coverage"),
            "coverage_unordered": ov("unordered", "coverage"),
            "selective_acc": ov("overall", "selective_accuracy"),
            "selective_acc_aligned": ov("aligned", "selective_accuracy"),
            "selective_acc_inverted": ov("inverted", "selective_accuracy"),
            "selective_acc_unordered": ov("unordered", "selective_accuracy"),
            "made_up_order_rate": ov("overall", "made_up_order_rate"),
            "symmetry": f"{c['symmetry']:.3f}" if c.get("symmetry") is not None else "",
            "transitivity": (f"{c['transitivity']['consistency_all']:.3f}"
                             if c["transitivity"].get("consistency_all") is not None else ""),
            "latency_p50_ms": s["cost"]["latency_p50_ms"],
            "latency_p95_ms": s["cost"]["latency_p95_ms"],
            "tokens_per_q": round((s["cost"]["prompt_tokens_mean"] or 0)
                                  + (s["cost"]["completion_tokens_mean"] or 0)),
        })
    out = settings.eval_path / f"report_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "report.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| metric | " + " | ".join(r["pipeline"] for r in rows) + " |",
          "|---|" + "---|" * len(rows)]
    for c in cols[1:]:
        md.append(f"| {c} | " + " | ".join(str(r.get(c, "")) for r in rows) + " |")
    (out / "report.md").write_text("\n".join(md) + "\n\nValues: mean [95% bootstrap CI].\n")
    print((out / "report.md").read_text())
    return out


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(prog="python -m app.eval_runner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="locate gold quotes in the document; drop misquotes")
    v.add_argument("--doc", required=True)
    v.add_argument("--gold", required=True, type=Path)
    v.add_argument("--out", required=True, type=Path)

    r = sub.add_parser("run", help="run one pipeline over a verified gold set")
    r.add_argument("--doc", required=True)
    r.add_argument("--gold", required=True, type=Path)
    r.add_argument("--pipeline", required=True, choices=["naive", "kaalkram_v1", "kaalkram_v2"])
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--probes", type=int, default=0, help="number of consistency triples")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--k-max", type=int, default=None)
    r.add_argument("--split", choices=["all", "dev", "test"], default="all",
                   help="stable hash of question id; fraction eval_dev_fraction -> dev")
    r.add_argument("--no-db", action="store_true")

    c = sub.add_parser("compare", help="paired significance between two runs")
    c.add_argument("run_a")
    c.add_argument("run_b")

    rp = sub.add_parser("report", help="headline table across runs")
    rp.add_argument("run_ids", nargs="+")

    a = ap.parse_args()
    if a.cmd == "verify":
        kept, rejected = verify_gold(a.doc, load_gold(a.gold))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("".join(q.model_dump_json() + "\n" for q in kept), encoding="utf-8")
        print(f"kept {len(kept)}, rejected {len(rejected)} -> {a.out}")
        for rj in rejected:
            print("  rejected", rj)
    elif a.cmd == "run":
        run(a.doc, a.gold, a.pipeline, repeats=a.repeats, n_triples=a.probes,
            workers=a.workers, k_max=a.k_max, use_db=not a.no_db, split=a.split)
    elif a.cmd == "compare":
        compare(a.run_a, a.run_b)
    elif a.cmd == "report":
        report(a.run_ids)


if __name__ == "__main__":
    main()
