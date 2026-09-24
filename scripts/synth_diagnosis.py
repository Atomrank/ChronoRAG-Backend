"""Synth pipeline diagnosis — write counts to docs/results/synth_diagnosis.md.

  python -u scripts/synth_diagnosis.py
  python -u scripts/synth_diagnosis.py --ids docs/results/synth_s1/rerun_ids.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import buildlog, docstore  # noqa: E402
from app.db import neo4j, pg, pool  # noqa: E402
from app.ingest_v2 import _normalise_with_map  # noqa: E402
from app.v2 import query  # noqa: E402

DOC = "doc_b281c9bd70bedd34"
OUT = ROOT / "docs" / "results" / "synth_diagnosis.md"


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=Path, default=ROOT / "docs" / "results" / "synth_s1" / "rerun_ids.json",
                    help="JSON with v1 / v2_llm / v2_oracle run ids (from synth_rerun_step7)")
    ap.add_argument("--v1", default=None)
    ap.add_argument("--v2", default=None, help="v2 llm run id")
    ap.add_argument("--v2-oracle", default=None)
    return ap.parse_args()


ARGS = _parse_args()
V1 = ARGS.v1 or "run_20260924_171941_kaalkram_v1_9ccdc3"
V2 = ARGS.v2 or "run_20260924_172410_kaalkram_v2_709742"
V2_ORACLE = ARGS.v2_oracle
if ARGS.ids.is_file() and not (ARGS.v1 or ARGS.v2):
    ids = json.loads(ARGS.ids.read_text(encoding="utf-8"))
    V1 = ids.get("v1", V1)
    V2 = ids.get("v2_llm", V2)
    V2_ORACLE = ids.get("v2_oracle", V2_ORACLE)

RUNS = [("v1", V1), ("v2_llm", V2)]
if V2_ORACLE:
    RUNS.append(("v2_oracle", V2_ORACLE))

random.seed(1)
pool()
lines: list[str] = []


def P(s: str = "") -> None:
    lines.append(s)
    print(s, flush=True)


def load_items(rid: str) -> list[dict]:
    p = ROOT / "data" / "eval" / rid / "items.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_sum(rid: str) -> dict:
    return json.loads((ROOT / "data" / "eval" / rid / "summary.json").read_text())


def ov(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (hi - lo) / max(1, a[1] - a[0]) if hi > lo else 0.0


def flush_md(extra: str = "") -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + extra, encoding="utf-8")


P("# Synth diagnosis (seed 1)")
P()
P(f"doc_id: `{DOC}`")
for name, rid in RUNS:
    P(f"{name} run: `{rid}`")
P()

# ---- 1 ----
P("## 1. Errors")
P()
for name, rid in RUNS:
    s = load_sum(rid)
    items = load_items(rid)
    errs = [(it.get("error") or "") for it in items if it.get("error")]
    P(f"### {name}")
    P(f"- `summary.errors` = **{s.get('errors')}**")
    P(f"- items with `error` set: **{len(errs)} / {len(items)}**")
    if errs:
        for msg, n in Counter(errs).most_common(10):
            P(f"  - n={n}: `{msg[:180]}`")
    else:
        P("  - (no per-item errors — abstentions are successful `cannot_determine` answers)")
    P()

# ---- 1b finish_reason from buildlog / summary llm_events ----
P("## 1b. finish_reason histogram (build + eval llm_events)")
P()
fr_all: Counter = Counter()
for name, rid in RUNS:
    s = load_sum(rid)
    for ev in s.get("llm_events") or []:
        fr = (ev.get("finish_reason") or ev.get("stop_reason") or "?")
        fr_all[f"{name}:{fr}"] += 1
# also last buildlog summary for current doc
try:
    bl = buildlog.summary(DOC)
    for kind, n in (bl or {}).items():
        if "truncat" in str(kind).lower() or "length" in str(kind).lower():
            fr_all[f"buildlog:{kind}"] += int(n) if isinstance(n, int) else 1
except Exception as exc:
    P(f"- buildlog.summary skipped: `{exc}`")
if fr_all:
    for k, n in fr_all.most_common():
        P(f"- `{k}`: **{n}**")
else:
    P("- (no llm_events with finish_reason in summaries; check buildlog for extract truncations)")
P()

# ---- 2 ----
P("## 2. Relation / detail counts")
P()
for name, rid in RUNS:
    items = load_items(rid)
    labs = Counter((it.get("pred_label") if it.get("pred_label") is not None else "null")
                   for it in items)
    P(f"### {name} pred_label")
    for lab, n in labs.most_common():
        P(f"- `{lab}`: **{n}**")
    P()

# ---- 2b per-stratum from summary ----
P("## 2b. Per-stratum accuracy / coverage / selective accuracy")
P()
P("| run | stratum | n | accuracy | coverage | selective_accuracy | made_up_order_rate |")
P("|-----|---------|--:|---------:|---------:|-------------------:|-------------------:|")
for name, rid in RUNS:
    o = load_sum(rid)["repeats"]["0"]["ordering"]
    for st in ("overall", "aligned", "inverted", "unordered"):
        x = o.get(st) or {}
        if not x:
            continue
        def f(k):
            v = x.get(k)
            return f"{v:.3f}" if isinstance(v, (int, float)) else ""
        P(
            f"| {name} | {st} | {x.get('n', '')} | {f('accuracy')} | {f('coverage')} | "
            f"{f('selective_accuracy')} | {f('made_up_order_rate')} |"
        )
P()

# ---- 3 ----
P("## 3. v2 graph / build")
P()
with pg() as cur:
    cur.execute("SELECT count(*) AS n FROM v2_events WHERE doc_id = %s", (DOC,))
    n_ev = cur.fetchone()["n"]
    cur.execute(
        "SELECT mode, count(*) AS n FROM v2_mentions WHERE doc_id = %s GROUP BY mode",
        (DOC,),
    )
    by_mode = {r["mode"]: r["n"] for r in cur.fetchall()}
    cur.execute(
        "SELECT type, count(*) AS n FROM v2_frames WHERE doc_id = %s GROUP BY type",
        (DOC,),
    )
    by_frame = {r["type"]: r["n"] for r in cur.fetchall()}
    cur.execute(
        "SELECT count(*) AS n FROM v2_mentions WHERE doc_id = %s AND mode = 'occurs' "
        "AND coalesce(is_telling, false) = false",
        (DOC,),
    )
    occurs_nt = cur.fetchone()["n"]
    cur.execute(
        "SELECT graph, stats, prompt_version FROM v2_graphs WHERE doc_id = %s", (DOC,)
    )
    row = cur.fetchone()
    g = row["graph"] if isinstance(row["graph"], dict) else json.loads(row["graph"])
    edges = g.get("edges") or []
    rem = g.get("removed") or []
    by_src: Counter = Counter()
    for e in edges:
        for s in e.get("sources") or ["?"]:
            by_src[s] += 1
    cur.execute(
        "SELECT kind, count(*) AS n FROM build_events WHERE doc_id = %s "
        "GROUP BY kind ORDER BY n DESC",
        (DOC,),
    )
    be = {r["kind"]: r["n"] for r in cur.fetchall()}

P(f"- events: **{n_ev}**")
P(f"- mentions_by_mode: `{by_mode}`")
P(f"- occurs non-telling: **{occurs_nt}**")
P(f"- frames_by_type: `{by_frame}`")
P(f"- edges: **{len(edges)}** by_source=`{dict(by_src)}`")
P(f"- removed: **{len(rem)}**")
P(f"- prompt_version: `{row['prompt_version']}`")
P(f"- build_events: `{be}`")
if len(edges) < 5:
    P("- **FLAG: edges near zero**")
if occurs_nt < 10:
    P("- **FLAG: occurs mentions rare**")
P()

raw_llm = 0
for p in (ROOT / "data" / "cache").glob(f"{DOC}_v2_*_s0.json"):
    raw_llm += len(json.loads(p.read_text(encoding="utf-8")).get("mentions") or [])
P(f"- raw LLM mentions in extract cache (sum s0): **{raw_llm}**")
P()

# ---- 4 ----
P("## 4. Offsets + build-time gold recall")
P()
doc = docstore.load_document(DOC)
with pg() as cur:
    cur.execute(
        "SELECT id, quote, char_start, char_end FROM v2_mentions WHERE doc_id = %s",
        (DOC,),
    )
    ments = [dict(r) for r in cur.fetchall()]
sample_m = ments if len(ments) <= 30 else random.sample(ments, 30)
ok = 0
for m in sample_m:
    span = doc.text[m["char_start"] : m["char_end"]]
    ns, _ = _normalise_with_map(span)
    nq, _ = _normalise_with_map(m["quote"] or "")
    if nq.strip() and (nq.strip() == ns.strip() or nq.strip() in ns or ns.strip() in nq):
        ok += 1
P(f"- mention quote match rate: **{ok}/{len(sample_m)} = {ok / max(1, len(sample_m)):.3f}**")

gold = [
    json.loads(l)
    for l in (ROOT / "data" / "gold" / "synth_s1.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
]
spans = []
for q in gold:
    for e in q.get("evidence") or []:
        if e.get("char_start") is not None:
            spans.append((e["char_start"], e["char_end"]))
with pg() as cur:
    cur.execute(
        "SELECT char_start, char_end FROM v2_mentions WHERE doc_id = %s", (DOC,)
    )
    mspans = [(r["char_start"], r["char_end"]) for r in cur.fetchall()]
hit_spans = sum(1 for gs in spans if any(ov(gs, ms) >= 0.5 for ms in mspans))
P(
    f"- gold evidence spans overlapping a mention (>=50%): "
    f"**{hit_spans}/{len(spans)} = {hit_spans / max(1, len(spans)):.3f}**"
)
P()

flush_md("\n\n_(section 5 grounding in progress…)_\n")
print(f"PARTIAL WROTE {OUT}", flush=True)

# ---- 5 ----
P("## 5. Grounding (30 gold questions)")
P()
P("(items.jsonl does not store `trace`; detail counted from live `query.answer` traces.)")
P()

gold_order = [q for q in gold if q.get("stratum")]
sample30 = gold_order[:30]
detail_c: Counter = Counter()
rel30: Counter = Counter()
ground_rows = []
acc_hits = 0

for i, q in enumerate(sample30, 1):
    print(f"grounding {i}/30 {q['id']}...", flush=True)
    ans = query.answer(DOC, q["question"])
    rel30[ans.relation] += 1
    joined = " | ".join(ans.trace or [])
    det = None
    for t in ans.trace or []:
        if t.startswith("detail="):
            det = t.split("=", 1)[1].strip()
            break
    if "not_found" in joined:
        detail_c["not_found"] += 1
        det = det or "not_found"
    elif "ambiguous" in joined:
        detail_c["ambiguous"] += 1
        det = det or "ambiguous"
    elif det:
        detail_c[det] += 1
    for key in ("unordered", "during", "overlap"):
        if key in joined:
            detail_c[key] += 1

    gold_spans = [
        (e["char_start"], e["char_end"])
        for e in (q.get("evidence") or [])
        if e.get("char_start") is not None
    ]
    cited = [tuple(s) for s in (ans.cited_spans or [])]
    hit = False
    if gold_spans and cited:
        for cs in cited:
            for gs in gold_spans:
                if ov(gs, cs) >= 0.5 or ov(cs, gs) >= 0.5:
                    hit = True
                    break
            if hit:
                break
    if hit:
        acc_hits += 1

    # pull event_a/b from parse via trace if present
    ground_rows.append(
        {
            "id": q["id"],
            "stratum": q.get("stratum"),
            "gold": q.get("gold_label"),
            "pred": ans.relation,
            "detail": det,
            "trace": ans.trace or [],
            "hit": hit,
        }
    )

P(f"- live relation counts (30): `{dict(rel30)}`")
P(f"- detail counts (30): `{dict(detail_c)}`")
P(
    f"- grounding Acc@1 (cited overlaps gold >=50%): "
    f"**{acc_hits}/{len(sample30)} = {acc_hits / max(1, len(sample30)):.3f}**"
)
P()
P("| id | stratum | gold | pred | detail | hit | trace (abbrev) |")
P("|----|---------|------|------|--------|-----|----------------|")
for r in ground_rows:
    tr = " · ".join(r["trace"][:4]).replace("|", "/")
    P(
        f"| {r['id']} | {r['stratum']} | {r['gold']} | {r['pred']} | {r['detail']} | "
        f"{r['hit']} | {tr[:140]} |"
    )
P()

# ---- 6 ----
P("## 6. v1 build + sample answers")
P()
with pg() as cur:
    cur.execute("SELECT count(*) AS n FROM events WHERE doc_id = %s", (DOC,))
    P(f"- postgres `events` rows: **{cur.fetchone()['n']}**")
    cur.execute(
        "SELECT detail FROM jobs WHERE doc_id = %s AND kind = 'kaalkram' AND status = 'done' "
        "ORDER BY started_at DESC LIMIT 1",
        (DOC,),
    )
    jrow = cur.fetchone()
    if jrow:
        d = jrow["detail"] if isinstance(jrow["detail"], dict) else json.loads(jrow["detail"] or "{}")
        P(f"- last v1 job: events={d.get('events')} major={d.get('major')} windows={d.get('windows')}")

with neo4j().session() as sess:
    n = sess.run(
        "MATCH (e:Event) WHERE e.doc_id = $d RETURN count(e) AS n", d=DOC
    ).single()["n"]
    P(f"- Neo4j `:Event` nodes for doc: **{n}**")

P()
P("### 5 v1 answers")
P()
items_v1 = load_items(V1)
shown = 0
for it in items_v1:
    if not it.get("stratum"):
        continue
    P(
        f"**{it['question_id']}** stratum={it.get('stratum')} "
        f"gold=`{it.get('gold_label')}` pred=`{it.get('pred_label')}` conf={it.get('confidence')}"
    )
    P(f"- Q: {(it.get('question') or '')[:160]}")
    P(f"- A: {(it.get('answer') or '')[:220]}")
    P()
    shown += 1
    if shown >= 5:
        break

P("## Root cause (diagnosis)")
P()
# Numbers from this run — do not invent.
o_v2 = load_sum(V2)["repeats"]["0"]["ordering"]["overall"]
cov = o_v2.get("coverage")
recall = hit_spans / max(1, len(spans))
P(
    f"1. Errors: see §1. Pred-label nulls should be gone after mapping "
    f"`not_applicable` → `cannot_determine` (or counted as `unrecognised_relation`)."
)
P(
    f"2. Build-time gold span recall (v2 mentions vs gold evidence): "
    f"**{recall:.3f}** (acceptance ≥ 0.8 for LLM extractor)."
)
P(
    f"3. v2_llm overall coverage={cov}, accuracy={o_v2.get('accuracy')}, "
    f"events={n_ev}, occurs_non_telling={occurs_nt}, edges={len(edges)}."
)
if V2_ORACLE:
    o_or = load_sum(V2_ORACLE)["repeats"]["0"]["ordering"]["overall"]
    P(
        f"4. v2_oracle upper bound: coverage={o_or.get('coverage')}, "
        f"accuracy={o_or.get('accuracy')} (graph was overwritten by oracle build)."
    )
P()
flush_md()
print(f"WROTE {OUT}", flush=True)
