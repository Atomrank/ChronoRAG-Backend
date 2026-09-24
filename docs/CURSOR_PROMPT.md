# Task: finish Kaalkram v2 (timeline-aware RAG) on top of the existing backend

Workspace layout: `ChronoRAG-Backend/` (FastAPI + Postgres/pgvector + Neo4j + Azure OpenAI, git branch
`v2-ingest-eval`) and `ChronoRAG-Frontend/` (Next.js). Paths below are relative to `ChronoRAG-Backend/`
unless they start with `ChronoRAG-Frontend/`. Three commits on the backend branch are done and tested
(ingestion, evaluation, v2 core). Your job: implement the tasks below in order, RUN everything, and
CHECK the results. Work in small commits (one task per commit), tests green after each.
Reference material (v1 methodology, seminar PPT text) is in `../docs/reference/`.

There is ONE point where you must stop and ask the user (marked **STOP**: missing Azure
credentials). Everything else you do yourself, including running the servers, builds, the
automatic gold set and the evaluations in the terminal. Nobody is available for manual labelling.

## 0. Read these first (do not skip)

- `docs/EVALUATION.md` — gold format, metric definitions, run/report commands.
- `app/ingest_v2.py` — clean text + page/section/paragraph char offsets; `windows()`; `locate_quote()`.
- `app/v2/solver.py` — interval-point temporal graph. `A before B` is the edge `A:e -> B:s`; only
  `before` edges go from an end point to a start point, so reachability is exact. Contradictions =
  cycles containing a strict edge; broken by removing the lowest-weight edge; every removal logged.
  No total order is imposed: unrelated events answer `cannot_determine` (detail `unordered`).
- `app/v2/frames.py` — narrator-frame stack. The LLM proposes open/close ops; code applies them.
- `app/v2/constraints.py` — generic constraint extractors (explicit/time/causal relations, succession
  within one frame, frame rules, lifecycle, genealogy) and `calibrate_sources()`.
- `app/v2/schemas.py`, `app/v2/prompts.py` — the strict-JSON contracts and system prompts for EVERY v2
  LLM call. Use them; do not invent new prompts without bumping `PROMPT_VERSION`.
- `app/metrics.py`, `app/eval_runner.py`, `app/gold_auto.py` (automatic silver gold), `app/gold_propose.py`.
- `tests/test_v2_core.py` — miniature scenarios (flashback, prophecy fulfilled later, parallel events,
  posthumous appearance). These define the intended semantics. Read them before touching v2 code.

## Non-negotiable rules

1. **No book-specific content anywhere in code, prompts, tests or config.** No character, place or
   book names; no alias lists; no phrase lists tuned to one book. Tests use invented names.
   Generic English connectives in prompts are fine. The same code must run unchanged on any book.
2. **No silent data loss.** Never truncate LLM input (`llm._sanitize_for_azure` raises
   `InputTooLongError` — split instead). Every dropped window, unresolved quote, content-filter hit,
   failed LLM call or rejected item is written with `buildlog.record(...)` and shows in build stats.
3. **All LLM calls go through `app/llm.py`** (`chat_structured` with the v2 schemas). Temperature 0 for
   primary extraction and all query-time calls.
4. **Do not change metric definitions or relevance thresholds** in `metrics.py`. Do not weaken the
   naive baseline in `naive_rag.py` (it must stay allowed to answer cannot_determine, full chunks,
   structured label, no code-written answers). Do not loosen `locate_quote` to admit more gold.
5. **Do not change the semantics of `solver.py`, `frames.py`, `constraints.py`** unless a failing test
   proves a bug; if you must, add the test first. Performance changes are fine if tests stay green.
6. **Every number comes from `eval_runner`.** No hand-computed metrics in docs or UI.
7. Keep v1 (`passes.py`, `graph.py`, `query_engine.py`) working as the `kaalkram_v1` baseline.
   Do not "improve" v1's design; only fix crashes.
8. If something in this spec is ambiguous or impossible, stop and write the question in
   `docs/OPEN_QUESTIONS.md` instead of guessing.

## Task 0 — bring up and verify ingestion on the real book

1. Check `ChronoRAG-Backend/.env` exists with real Azure values (copy from `.env.example`).
   If it does not, **STOP** and ask the user for: endpoint, API key, chat deployment, embedding
   deployment. A gold/full-text deployment (e.g. gpt-4.1) is optional; continue without it.
2. Create a venv, `pip install -r requirements.txt -r requirements-dev.txt && pytest -q` (all green).
3. `python scripts/get_books.py` -> `data/books/sabha_parva_ganguli.pdf`. The Old Man and the Sea is
   copyrighted: if `data/books/` has no PDF for it, continue without it and note that in the report.
4. `docker compose up -d`; start the API (`uvicorn app.main:app --port 8000`); upload the PDFs
   through `POST /api/documents` (new tables need a fresh upload).
3. Write `scripts/inspect_ingest.py DOC_ID` that prints: `structure_source`, stats, the first 40
   section titles with levels, 10 random paragraphs, paragraphs longer than 5,000 chars, and any
   page with zero text. Run it on the Sabha Parva PDF and on The Old Man and the Sea.
4. If headings are missed or running headers survive, fix `ingest_v2.py` **generically** (e.g. a new
   typographic signal) and add a synthetic-PDF test reproducing the failure. Commit the inspection
   output to `docs/ingest_report_<doc>.txt`.

## Task 1 — v2 storage (migration `infra/postgres/migrations/003_v2.sql`, idempotent)

Tables, all with `doc_id` FK + cascade: `v2_frames` (id, unit_id, type, parent, narrator, listener,
open_at, close_at, depth, summary, auto_closed), `v2_mentions` (id, event_id, frame_id, window_id,
char_start, char_end, para_id, quote, description, mode, event_type, subject_entity, participants
JSONB [{entity_id, surface, role}], location, time_expressions JSONB, posthumous, is_telling,
tells_frame, sample_idx), `v2_events` (id, description, event_type, first_offset, participants,
embedding vector(1536), partition_frame), `v2_entities` (id, canonical, surfaces JSONB),
`v2_relations` (mention_a, mention_b, rel, cue, consistency, quote, char_start, char_end),
`v2_graphs` (doc_id PK, graph JSONB = `TemporalGraph.to_dict()`, stats JSONB, weights JSONB,
prompt_version, created_at). HNSW index on `v2_events.embedding`.

## Task 2 — window extraction (`app/v2/extract.py`)

- The whole document is ONE unit processed **sequentially**: each window receives the current frame
  stack (`FrameTracker.state()`), and embedded narrations often span several sections, so the stack
  must carry across section boundaries. (Parallelism across units is a later optimisation: only
  behind a setting, only across level-1 sections of a document with >= 2 heading levels.)
- Windows from `ingest_v2.windows(doc, settings.v2_window_chars, settings.v2_window_overlap_paras,
  break_level=0)` so windows can cross section boundaries and keep context.
- User message = `FRAME STACK:` (JSON of `state()`) + `SECTION:` path + window text.
  Call `chat_structured(EXTRACT_SYSTEM, user, WindowExtraction, temperature=0)`.
- Resolve every quote to char offsets: `locate_quote` within the cited paragraph's span first, then
  within the window span; unresolved -> drop that item and `buildlog.record(..., "unresolved_quote")`.
  Reject mentions whose `para_id` is a context paragraph.
- Frame ops -> `FrameOp` (resolved offset; `frame_ref` `NEW:n` mapped to the id created for the n-th
  open in this window). Apply per window; `finish()` at unit end; persist `tracker.warnings` to
  build_events. Assign each mention's frame with `tracker.frame_at(mention.start)` AFTER `finish()`.
- For every non-main frame create a telling mention (`is_telling=True`, `tells_frame=frame.id`,
  mode `occurs`, participants narrator/listener as `agent`/`present`, span = 1st sentence at open_at,
  frame = parent frame, its own new event).
- Self-consistency: `settings.v2_extract_samples` (default 1 for development, 3 for reported runs).
  Sample 0 at temperature 0 is primary (defines mentions/frames). Samples 1..k-1 at temperature 0.4
  only vote on relations: a relation's `consistency` = fraction of samples containing a relation of
  the same type between mentions whose quote spans overlap the primary ones by >= 50%.
- Cache every window result to `data/cache/{doc_id}_v2_{window_id}_s{k}.json`; resume from cache.
- Content filter: on `ContentFilterError`, if `settings.local_llm_base_url` is set, retry the same
  window with the local OpenAI-compatible server (vLLM on the 4090, JSON-schema guided decoding,
  model name in `settings.local_llm_model`); record `buildlog` either way. Never skip silently.
- Tests: mock `chat_structured`; check quote resolution, NEW:n mapping, context-paragraph rejection,
  telling-mention creation, caching/resume, and that a window over budget raises instead of cutting.

## Task 3 — entity resolution (`app/v2/entities.py`)

- Collect all surface forms (participants, subjects, aliases, kinship names) with up to 3 short
  contexts each (±150 chars around mentions).
- Candidate groups: union of (a) case/punctuation-normalised equality, (b) alias statements from the
  text, (c) embedding nearest neighbours of `form + contexts` (top-5, cosine >= setting, report it).
  Split groups larger than 40 forms by embedding clustering before calling the LLM.
- `chat_structured(ENTITY_SYSTEM, ..., EntityResolution)` per group; union-find the result;
  canonical ids `ent_<n>`. Same surface form may map to different entities in different contexts
  only if the LLM puts the contexts in different clusters — support this by keying forms as
  `(surface, mention_id)` when a group is flagged ambiguous.
- Map `participants`, `subject`, kinship to entity ids. Tests with invented names including one
  deliberate name collision (two people, one name) and one person with three epithets.

## Task 4 — event coreference (`app/v2/coref.py`)

- Each primary mention starts as its own event. Candidates for mention m: earlier mentions sharing
  >= 1 canonical participant AND description-embedding cosine >= setting, top 10.
- Never merge: two mentions from the same window unless the LLM says so; a `hypothetical` with any
  non-hypothetical; two `is_telling` mentions. Allowed: `predicted` + `occurs` (prophecy fulfilled),
  `recounted` + `occurs` (retelling).
- `chat_structured(COREF_SYSTEM, both passages ±300 chars, CorefDecision)`; union-find; event
  description = description of the earliest `occurs`/`recounted` mention, else the first.
- Tests with mocked LLM for: retelling merged, prophecy+fulfilment merged, two similar battles kept apart.

## Task 5 — build job `run_kaalkram_v2` + route `POST /api/documents/{id}/build/kaalkram_v2`

extract -> entities -> coref -> `constraints.all_constraints(frames, mentions, rels, kin, weights)`
-> `TemporalGraph(event_ids, first_offset).add(...)` -> `repair()` -> persist (tables + `v2_graphs`)
-> embed events (`description + " | " + first quote`) -> done. Weights = latest calibrated weights
for this doc if present, else `SOURCE_PRIOR`. Progress stages in the job row. Completion detail:
windows, units, frames by type, mentions by mode, events, entities, constraints by source, removed
edges by source, removed weight, build_events summary, prompt_version, extract_samples.
Add `GET /api/documents/{id}/v2/graph` (events + edges + removed contradictions) and
`GET /api/documents/{id}/v2/timeline` (`linear_extension()` labelled "display order only").

## Task 6 — query engine (`app/v2/query.py`) + eval adapter `kaalkram_v2`

1. `QuestionParse` (temperature 0).
2. Ground each event description: candidates = events whose participants include entities matched
   from `parse.entities` (use the entity surfaces table) UNION embedding top-20 over v2_events;
   send candidate descriptions + first quotes to `GROUND_SYSTEM`. `-1` -> answer that the event
   was not found (relation cannot_determine, detail `not_found`). `ambiguous` -> list the options and
   return cannot_determine with detail `ambiguous`. Record grounding in the trace.
3. `order`: `graph.relation(a, b)`. `before_after_x`: predecessors/successors of X from the closure,
   keep those sharing an entity with X or within the top-10 by similarity, ordered by
   `linear_extension()` (say it is display order). `sequence`: pairwise relations; report pairs the
   graph cannot order. `state` (alive/present at X): lifecycle events of the entity vs X.
   `factual`: hybrid retrieval over mention passages (BM25 + embeddings) with the naive chunks as
   fallback — this is the no-regression control.
4. Fetch source passages by offset for every event in the chain (mention span ±200 chars) and call
   `VERBALISE_SYSTEM` with relation + detail + chain + passages. If the returned `relation` differs
   from the computed one, keep the computed one and log it.
5. Return `PipelineAnswer(pipeline="kaalkram_v2", relation, confidence = product of chain edge p
   (1.0 when grounded but unordered is not applicable -> use grounding confidence), cited_spans =
   mention spans used, retrieved = ranked units [grounded events, then other candidates, then chain
   events], each with `spans` = all its mention spans and `tokens`, trace)`.
   Register it in `eval_runner._adapters()`.
6. Tests with a tiny in-memory graph and mocked LLM calls for every question type.

## Task 7 — calibration and splits

- Add `--split {all,dev,test}` to `eval_runner run`; split = stable hash of question id, 30% dev.
- `scripts/calibrate.py DOC_ID GOLD`: map gold events to v2 events (mention span overlapping the
  evidence span by >= 50%), call `calibrate_sources` on DEV pairs only, store weights in
  `v2_graphs.weights`, rebuild the graph (no re-extraction), print the table. Report numbers on TEST.

## Task 8 — synthetic benchmark (`app/synth.py`)

Seeded generator: invented entities and places; a story timeline with births/deaths, parallel
branches, and events linked by explicit connectives; rendered in discourse order with controlled
rates of recollection frames (with a named narrator), nested frames, separate tales, prophecies
fulfilled later, hypotheticals, and parallel events told sequentially. Output a PDF (via PyMuPDF,
with headings) + a verified gold JSONL (exact quotes, strata from positions). Parameters and seed
saved beside the output. CLI: `python -m app.synth --seed 1 --events 150 --flashback-rate 0.2
--nesting 2 --out data/synth/s1`. Tests: gold quotes all locate; strata counts match parameters.

## Task 9 — frontend (`ChronoRAG-Frontend`)

- Remove the hardcoded presets and the regex order parser (`lib/order.ts`); use `relation` from the
  API. Add `GET /api/documents/{id}/gold` in the backend (questions from the latest gold set) and
  load presets from it.
- Compare page: three columns (Naive, v1, v2). v2 shows the evidence chain (each step: relation,
  source, weight, quoted passage with page) and an explicit "The text does not settle this" state
  with the detail (unordered / during / overlap / ambiguous / not found).
- Metrics page: list `eval-runs`, show the report table with 95% CIs, recall@k curves (k = 1..20),
  per-stratum accuracy bars, consistency, cost; show build health (`build_events` by kind) and
  removed contradictions for v2.

## Task 10 — docs

Rewrite `methodology.md` for v2 (ingestion, frames, extraction, entities, coref, constraints,
solver, query, evaluation), stating hyperparameters and what each module can and cannot do. Keep a
short "v1 limitations found" section: 14k truncation, index-first truncation in Pass 2, graph equal
to the sort, graph check not passed to the model, rigged naive fallback.

## Task 11 — end-to-end runner and the check sequence

Write `scripts/e2e.py` (uses the HTTP API; `httpx` is fine to add): upload a PDF, build naive,
kaalkram_v1 and kaalkram_v2, poll jobs, print build stats and `build_events` summary. Then run the
checks in THIS order and fix problems before moving on:

1. **Synthetic set** (Task 8, seed 1, ~150 events): build all three pipelines, run `eval_runner`
   (`--repeats 1 --probes 20`). Expected: v2 clearly above naive on the `inverted` and `unordered`
   strata, symmetry and transitivity close to 1 for v2. If not, debug extraction/frames/grounding
   on this set first — it has exact answers and needs no human.
2. **Sabha Parva**: build all three pipelines. Inspect: frames by type (expect a handful of
   recollections and separate tales, not hundreds), 10 random mentions with their quotes, the
   removed contradictions, entity clusters containing several epithets. Write findings to
   `docs/results/sabha_build_check.md`.
3. **Automatic silver gold — no human labelling** (the user works alone):
   `python -m app.gold_auto --doc <sabha id> --n 150 --out data/gold/sabha_silver.jsonl`.
   It proposes events/pairs, verifies every quote, and keeps a pair only if every judge in
   `GOLD_JUDGES` agrees in both A/B orders (see the module docstring). Before running it:
   - If `LOCAL_LLM_BASE_URL` is set (vLLM on the 4090), add a local judge from a different model
     family, e.g. `GOLD_JUDGES=gpt-4o,local:Qwen/Qwen2.5-14B-Instruct`. Two model families are much
     better than one. If no local server exists, run with `GOLD_JUDGES=gpt-4o` and say so.
   - `cannot_determine` pairs are only kept when `GOLD_FULLTEXT_JUDGE` (a model that can read the
     whole parva, e.g. a gpt-4.1 deployment) confirms them. Without it, the Sabha silver set has
     before/after pairs only; then the `unordered` stratum and the made-up-order rate are reported
     from the synthetic set only. Do not invent unordered gold any other way.
   - Read `sabha_silver.report.json`: pairs kept, drop reasons, judge kappa. If fewer than 60 pairs
     are kept, rerun with a larger `--n` (up to 400). Report `same_model_as_pipeline` honestly.
   The human path (`app.gold_propose` + CSV) stays available but is NOT used in this run.
4. `eval_runner verify` on the silver file, calibrate on dev (Task 7), then `eval_runner run` for
   naive, kaalkram_v1, kaalkram_v2 with `--repeats 3 --probes 40 --split test`, then `report` and
   `compare` (v2 vs naive, v2 vs v1). Label every Sabha number "silver gold (unanimous model
   judges, not human-verified)" in the summary.
5. Same for The Old Man and the Sea if its PDF is present (silver gold via `app.gold_auto`).
6. Write `docs/results/SUMMARY.md`: the report tables (copied from `report.md`, not retyped), the
   significance results, build health, the three biggest failure modes seen in `items.jsonl` with
   2 examples each, and what to fix next. No claims the numbers do not support.

## Definition of done

- `pytest -q` green; each new module has tests; `pyflakes app` clean.
- On the Sabha Parva: `gold_auto` (silver gold) -> `verify`; then
  `eval_runner run` for naive, kaalkram_v1, kaalkram_v2 (`--repeats 3 --probes 40 --split test`),
  `report` and `compare`. Same on The Old Man and the Sea and one synthetic set.
- Commit the report folders' `report.md` files under `docs/results/`. Do not edit numbers by hand.
