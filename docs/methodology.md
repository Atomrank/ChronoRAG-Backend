# Kaalkram v2 methodology

Kaalkram is a timeline-aware RAG system for narrative PDFs. v2 keeps the honest
naive baseline and the v1 multi-pass pipeline, and adds an interval-point temporal
graph with narrator frames. Hyperparameters live in `app/config.py` (env-overridable);
reported numbers come only from `app/eval_runner.py`.

## Ingestion

`app/ingest_v2.py` builds one cleaned `Document`: full text, page/section/paragraph
char offsets, structure from TOC / font / caps. `windows(doc, max_chars, overlap_paras,
break_level)` yields paragraph-aligned windows. `locate_quote` maps evidence to unique
offsets (short or ambiguous quotes rejected). No silent truncation of LLM inputs.

## Frames

`app/v2/frames.py` — LLM proposes open/close ops; `FrameTracker` applies them.
Types: main, recollection, separate_tale, prediction, hypothetical. Stack carries
across the whole document (one unit, `break_level=0`).

## Extraction

`app/v2/extract.py` — per window: frame stack + section path + text → `WindowExtraction`
(`EXTRACT_SYSTEM`, temperature 0). Quotes resolved; context paragraphs rejected;
unresolved quotes logged via `buildlog`. Extra samples (setting `v2_extract_samples`)
vote only on relation consistency. Telling mentions created for non-main frames.
Cache under `data/cache/`. Content-filter → optional local LLM retry.

## Entities / coreference

`entities.py` — surfaces + contexts; groups by norm equality, aliases, embedding NN
(`v2_entity_*` settings); LLM clusters → `ent_n`. Same surface may split by context.
`coref.py` — mention candidates by shared entity + description cosine (`v2_coref_*`);
LLM decide; hard blocks for hyp↔non-hyp and two tellings; predicted/recounted may merge
with occurs.

## Constraints and solver

`constraints.py` — explicit/time/causal, succession (same frame, occurs), frame,
lifecycle, genealogy. Weights start as `SOURCE_PRIOR`, replaced by `calibrate_sources`
on DEV. `solver.py` — interval points; `A before B` is `A:e → B:s`; contradictions =
cycles with a strict edge, broken by removing lowest-weight edge (logged). No total
order: unconnected pairs → `cannot_determine` / `unordered`.

## Query

`query.py` — `QuestionParse` → ground events (entity surfaces ∪ embed top-k) →
`graph.relation` / predecessors / sequence / lifecycle state / factual hybrid →
`VERBALISE_SYSTEM` (computed relation kept if verbaliser differs). Adapter
`kaalkram_v2` in `eval_runner`.

## Evaluation

Gold JSONL verified by `locate_quote`. Metrics in `metrics.py` (do not change).
`--split {all,dev,test}`: stable hash of question id, `eval_dev_fraction` for DEV.
Silver gold via `gold_auto` (unanimous judges); fulltext judge required for
`cannot_determine` pairs.

## What each module cannot do

- Solver never invents order between unrelated events.
- Extraction cannot invent quotes outside the window.
- Grounding returns not_found / ambiguous rather than guessing.
- Naive stays allowed to abstain; not weakened for comparison.

## v1 limitations found

1. 14k truncation of LLM inputs (v2 raises `InputTooLongError` / windows instead).
2. Index-first truncation in Pass 2 merges.
3. Graph equal to the sort — ordering was not independent evidence.
4. Graph check not passed through to the model at answer time.
5. Rigged naive fallback that wrote answers in code (removed; naive always model-authored).
