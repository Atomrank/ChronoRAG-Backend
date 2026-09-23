# Evaluation

Every reported number is computed by `app/metrics.py` from per-question rows that
`app/eval_runner.py` writes to Postgres (`eval_runs`, `eval_items`) **and** to
`data/eval/<run_id>/` (`config.json`, `items.jsonl`, `summary.json`).
Nothing is typed in by hand, and any metric can be recomputed for old runs.

## Workflow

```bash
# 1. upload the PDF through the API (stores clean text + char offsets)
# 2. write or generate the raw gold set -> data/gold/sabha_raw.jsonl
python -m app.eval_runner verify --doc DOC_ID --gold data/gold/sabha_raw.jsonl --out data/gold/sabha_v1.jsonl
# 3. run each pipeline (3 repeats = LLM-nondeterminism spread; 40 consistency triples)
python -m app.eval_runner run --doc DOC_ID --gold data/gold/sabha_v1.jsonl --pipeline naive       --repeats 3 --probes 40
python -m app.eval_runner run --doc DOC_ID --gold data/gold/sabha_v1.jsonl --pipeline kaalkram_v1 --repeats 3 --probes 40
# 4. headline table + paired significance
python -m app.eval_runner report  RUN_NAIVE RUN_V1
python -m app.eval_runner compare RUN_V1 RUN_NAIVE
```

`verify` locates every evidence quote in the document text (whitespace, quote
and dash differences tolerated) and **drops** any question whose quotes are not
found exactly once. A human only confirms the label of what survives.

## Gold format (one JSON object per line)

```json
{"id": "q001", "qtype": "order", "stratum": "inverted",
 "question": "Did <event A> happen before or after <event B>?",
 "gold_label": "before",
 "events": {"A": {"id": "E01", "desc": "<event A>"}, "B": {"id": "E02", "desc": "<event B>"}},
 "evidence": [{"group": "A", "quote": "<verbatim text>"}, {"group": "B", "quote": "<verbatim text>"}],
 "verified_by": "AT"}
```

`qtype`: order | sequence | before_after_x | state | factual.
`stratum` (order questions): aligned (book order = story order), inverted
(they differ), unordered (text does not settle it; gold_label = cannot_determine).
`gold_label` is event A relative to event B.
`events` ids let the runner build consistency probes automatically.

## Metrics

Relevance is judged on character spans: a gold span is *found* if one retrieved
unit covers ≥50% of it; a unit is *relevant* if it finds a gold span or ≥50% of
it lies inside gold spans. Both thresholds are stored with every run.

| metric | definition |
|---|---|
| recall@k | gold spans found in top k ÷ gold spans |
| precision@k | relevant units in top k ÷ k |
| hit@k | 1 if any relevant unit in top k |
| MRR | 1 ÷ rank of first relevant unit |
| nDCG@k | binary relevance, log2 discount, ideal = one unit per gold span |
| pair_recall@k | 1 if evidence for both A and B is in top k |
| recall@N tok | recall when units are taken in rank order until N tokens |
| accuracy, macro-F1 | over before / after / cannot_determine, per stratum |
| coverage | committed (non-cannot_determine) answers ÷ questions |
| selective accuracy | correct ÷ committed |
| made-up order rate | committed answers on unordered pairs ÷ unordered pairs |
| AURC | area under risk–coverage curve, using the model's confidence |
| symmetry | reversed question gets the inverse label |
| transitivity | pairwise answers on event triples contain no cycle |
| citation P/R | cited spans vs gold evidence spans |
| Kendall τ-b, Spearman ρ | global order vs gold order (when a gold order is given) |
| latency p50/p95, tokens | per question |

Every metric is reported with a 95% bootstrap CI (10,000 resamples).
Pipelines are compared with an exact McNemar test and a paired bootstrap on accuracy.
With `--repeats N`, the spread (mean, min, max) of the headline metrics is stored.

Pipeline errors count as abstentions in accuracy and are also counted in
`summary.errors`; check that number before reporting.
