# Day report — 24 Sep 2026

Kaalkram / ChronoRAG synth seed-1 recovery: extract under-recall → fixes → re-eval → naive vs v2 head-to-head.

**Doc under test:** `doc_b281c9bd70bedd34` (synthetic seed 1, ~150 events, ~12k chars).  
**Gold:** `data/gold/synth_s1.jsonl` (154 order questions + reversed probes → 308 eval items).  
**Rule:** no book-specific hardcoding in pipeline code/prompts/config; numbers below are from `eval_runner` `summary.json` only.

---

## 1. Starting point (broken)

Earlier seed-1 scoreboard (pre-fix):

| metric | naive | kaalkram_v1 | kaalkram_v2 |
|--------|------:|------------:|------------:|
| order accuracy | 0.318 | 0.065 | 0.065 |
| coverage | 0.506 | 0.000 | 0.000 |
| hit@5 | 0.812 | 0.162 | 0.058 |

**Diagnosis** (`docs/results/synth_diagnosis.md`, earlier revision): not exception-driven abstention. Root cause was **extraction under-recall** — v1 ~7 events; v2 raw extract cache ~13 mentions; build-time gold span recall **0.039**. Offsets were fine (quote match 1.000). Graph/answer logic was not the primary bug.

---

## 2. What we changed (7-step protocol)

### Step 1 — Instrument extraction
- Log per LLM extract call: window, input chars, `max_tokens`, `finish_reason`, completion tokens, parse salvage.
- Hard-fail on `finish_reason=length` via `OutputTruncatedError` (no silent truncation).
- Script: `scripts/extract_instrumentation.py` → `docs/results/extract_instrumentation_synth.md`.

**Finding:** v1 was truncating (whole-doc + `max_tokens=1600`).

### Step 2 — Shared windowing + output budget
- Both v1 pass1 and v2 use shared extract windows: **1500 chars / 12 paras / 1 overlap**.
- Extract `max_tokens` ≈ `2 × max_paras × extract_tokens_per_event` (floor via settings).

### Step 3 — Exhaustive extraction mode
- Config `extraction_mode`: `exhaustive` (default) | `salient`.
- Prompts no longer ask for “major milestones only”; `PROMPT_VERSION` → **v2.2**.

### Step 4 — Oracle extractor ablation
- `settings.extractor`: `llm` | `oracle`.
- `app/v2/oracle.py`: deterministic regex parse of synth `[E####]` templates (upper bound on extract→graph→answer).

### Step 5 — Synth question cleanup
- `synth.question_phrase()`: gerund NP, no discourse prefix, no `[E####]` leak in questions.
- `textutil.scrub_llm_text()` strips E-tags from LLM-facing answer/ground text.
- Gold regenerated under `data/gold/synth_s1.jsonl`.

### Step 6 — Metrics
- **nDCG:** ideal DCG over **deduplicated relevant retrieval units** (capped at k); DCG credits each unique span-set once (keeps ndcg ≤ 1).
- Report columns: coverage / selective accuracy **by stratum**.
- `not_applicable` → eval label `cannot_determine`; unrecognised relations → counted **error** (no silent `pred_label=null`).

### Step 7 — Rerun
- Rebuild + eval: v1, v2 (llm), naive (same cleaned gold).
- Oracle build was started for upper bound; if incomplete at doc time, note in §5.

Supporting infra shipped earlier the same effort: v2 storage (`003_v2.sql`), extract/entities/coref/persist/query, eval adapter, frontend Compare/Metrics, `methodology.md`.

---

## 3. Build-time acceptance (LLM extractor)

After exhaustive + window fixes (v2 job `job_4928648fddd5`):

| check | value |
|-------|------:|
| v2 events | 149 |
| v2 mentions | 162 (`occurs` 139 + frames) |
| windows | 14 |
| **gold span recall** (≥50% overlap) | **0.812** (250/308) |
| length-truncated extract calls | 0 (hard-fail path; build completed) |

Acceptance target was gold span recall ≥ 0.8 — **met**.

v1 rebuild produced **118** postgres events (vs ~7 before).

---

## 4. Eval results (cleaned gold, 2026-09-24 evening)

### Run ids

| pipeline | run_id |
|----------|--------|
| naive | `run_20260924_232553_naive_765a1c` |
| kaalkram_v1 | `run_20260924_225755_kaalkram_v1_5afe52` |
| kaalkram_v2 (llm) | `run_20260924_231311_kaalkram_v2_a51196` |

### Headlines

| metric | naive | v1 | v2_llm |
|--------|------:|---:|-------:|
| overall order acc | 0.461 | 0.338 | **0.565** |
| coverage | 0.695 | 0.558 | 0.526 |
| selective accuracy | 0.607 | 0.547 | **0.963** |
| aligned acc | 0.478 | 0.331 | **0.574** |
| inverted acc (n=8) | 0.000 | 0.250 | 0.000 |
| unordered acc (n=10) | 0.600 | 0.500 | **0.900** |
| hit@5 | **0.779** | 0.903 | 0.253 |
| MRR | **0.560** | 0.773 | 0.167 |
| errors | 0 | 0 | 0 |

### Naive vs v2 (paired, same gold)

Full table: `docs/results/synth_s1/naive_vs_v2.md`.

- Accuracy diff (v2 − naive): **+0.104** (bootstrap one-sided p ≈ 0.027; McNemar two-sided p ≈ 0.064).
- v2 wins **overall** and **unordered**; commits less often but is usually correct when it does (selective acc 0.963).
- Naive still wins **passage retrieval** (hit@5 / MRR). v2 answers are carried more by the **temporal graph** than by top-k chunks.
- **Inverted** remains unsolved for both (tiny stratum, n=8; v2 coverage 0 on inverted).

---

## 5. Open / in progress at end of day

- **Oracle extractor** full rebuild+eval (upper bound) — kicked off after v2-llm eval; confirm `rerun_ids.json` / scoreboard when job finishes.
- Regenerate `scripts/synth_diagnosis.py` output against the new run ids (finish_reason histogram, grounding Acc@1, stratum table).
- **Inverted stratum** — still broken (always abstain / never correct on this tiny set).
- v2 **retrieval metrics** weak vs naive — expected if citation/retrieval units are graph-oriented; revisit if hit@k is used as a primary KPI.
- Sabha Parva silver gold / full TASK §11 sequence still ahead after synth acceptance.

---

## 6. Files touched (backend)

**Core:** `app/v2/*` (extract, entities, coref, persist, query, oracle, prompts), `app/synth.py`, `app/textutil.py`, `app/llm.py`, `app/config.py`, `app/passes.py`, `app/jobs.py`, `app/metrics.py`, `app/eval_runner.py`, `app/naive_rag.py`, `app/query_engine.py`, migration `003_v2.sql`.

**Tests:** `tests/test_v2_*.py`, `test_oracle_extract.py`, `test_extraction_mode.py`, `test_synth_and_split.py`, metrics/eval_runner updates.

**Docs / scripts:** this report, `docs/results/synth_s1/naive_vs_v2.md`, `docs/methodology.md`, `docs/EVALUATION.md` (nDCG wording), `scripts/synth_rerun_step7.py`, `scripts/naive_vs_v2.py`, `scripts/synth_diagnosis.py`, `scripts/extract_instrumentation.py`.

**Frontend (separate repo):** Compare / Metrics pages wired to v2 API (no hardcoded book presets).

---

## 7. Bottom line

1. Synth failure mode was **extract under-recall**, not metrics or offsets.  
2. Exhaustive windowed extract restored recall (**0.812** gold-span).  
3. On cleaned gold, **kaalkram_v2 (llm) beats naive on order accuracy** (0.565 vs 0.461) with much higher selective accuracy; retrieval remains naive’s strength; inverted still open.
