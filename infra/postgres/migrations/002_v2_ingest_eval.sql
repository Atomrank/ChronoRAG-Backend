-- ============================================================
-- 002: character-offset ingestion + evaluation recording
-- Idempotent: safe to run on every startup (db.ensure_schema runs it).
-- ============================================================

-- ---------- v2 ingestion: one clean text per document + offset maps
CREATE TABLE IF NOT EXISTS doc_text (
    doc_id            TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    text              TEXT  NOT NULL,
    text_sha1         TEXT  NOT NULL,
    structure_source  TEXT  NOT NULL,            -- toc | font | caps | none
    stats             JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS doc_pages (
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no     INT  NOT NULL,
    char_start  INT  NOT NULL,
    char_end    INT  NOT NULL,
    PRIMARY KEY (doc_id, page_no)
);

CREATE TABLE IF NOT EXISTS doc_sections (
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id  TEXT NOT NULL,
    level       INT  NOT NULL,
    title       TEXT NOT NULL,
    char_start  INT  NOT NULL,
    char_end    INT  NOT NULL,
    parent_id   TEXT,
    PRIMARY KEY (doc_id, section_id)
);

CREATE TABLE IF NOT EXISTS doc_paragraphs (
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    para_id     TEXT NOT NULL,
    char_start  INT  NOT NULL,
    char_end    INT  NOT NULL,
    page_start  INT  NOT NULL,
    page_end    INT  NOT NULL,
    section_id  TEXT,
    is_heading  BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (doc_id, para_id)
);
CREATE INDEX IF NOT EXISTS doc_paragraphs_span_idx ON doc_paragraphs (doc_id, char_start);

-- naive chunks now carry exact offsets so recall@k can be scored on spans
ALTER TABLE naive_chunks ADD COLUMN IF NOT EXISTS char_start INT;
ALTER TABLE naive_chunks ADD COLUMN IF NOT EXISTS char_end   INT;

-- ---------- gold sets
CREATE TABLE IF NOT EXISTS eval_gold_sets (
    gold_set_id   TEXT PRIMARY KEY,               -- e.g. sabha_v1
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content_sha1  TEXT NOT NULL,                  -- hash of the JSONL file
    n_questions   INT  NOT NULL,
    notes         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_questions (
    gold_set_id   TEXT NOT NULL REFERENCES eval_gold_sets(gold_set_id) ON DELETE CASCADE,
    question_id   TEXT NOT NULL,
    qtype         TEXT NOT NULL,                  -- order|sequence|before_after_x|state|factual
    stratum       TEXT,                           -- aligned|inverted|unordered
    question      TEXT NOT NULL,
    gold_label    TEXT,                           -- before|after|cannot_determine
    gold_answer   TEXT,
    gold_sequence JSONB,
    evidence      JSONB NOT NULL,                 -- [{group, quote, char_start, char_end}]
    pair_key      TEXT,
    triple_key    TEXT,
    triple_slot   TEXT,
    is_reversed   BOOLEAN NOT NULL DEFAULT false,
    verified_by   TEXT,                           -- initials of the person who confirmed
    PRIMARY KEY (gold_set_id, question_id)
);

-- ---------- runs: everything needed to reproduce a number
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline        TEXT NOT NULL,                -- naive | kaalkram_v1 | kaalkram_v2
    doc_id          TEXT NOT NULL,
    doc_text_sha1   TEXT,
    gold_set_id     TEXT NOT NULL,
    gold_sha1       TEXT NOT NULL,
    git_commit      TEXT,
    git_dirty       BOOLEAN,
    chat_deployment TEXT,
    embed_deployment TEXT,
    api_version     TEXT,
    settings        JSONB NOT NULL,               -- every pipeline setting, secrets removed
    params          JSONB NOT NULL,               -- k_max, ks, budgets, temperature, seed, repeat
    summary         JSONB,                        -- output of metrics.aggregate()
    status          TEXT NOT NULL DEFAULT 'running',
    error           TEXT
);

CREATE TABLE IF NOT EXISTS eval_items (
    run_id            TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    question_id       TEXT NOT NULL,
    repeat_idx        INT  NOT NULL DEFAULT 0,
    answer            TEXT,
    pred_label        TEXT,
    confidence        REAL,
    pred_sequence     JSONB,
    retrieved         JSONB NOT NULL,             -- [{rank, unit_id, spans, tokens, score}]
    cited_spans       JSONB NOT NULL,
    latency_ms        INT,
    prompt_tokens     INT,
    completion_tokens INT,
    metrics           JSONB,                      -- per-question retrieval metrics
    error             TEXT,
    PRIMARY KEY (run_id, question_id, repeat_idx)
);

-- ---------- build-health log (content filter hits, truncations, failed windows)
CREATE TABLE IF NOT EXISTS build_events (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    job_id      TEXT,
    kind        TEXT NOT NULL,                    -- content_filter | truncated | failed_window | ...
    ref         TEXT,                             -- window id, batch id, ...
    detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS build_events_doc_idx ON build_events (doc_id, kind);
