-- ============================================================
-- 003: Kaalkram v2 storage (frames, mentions, events, entities,
--      relations, temporal graphs). Idempotent.
-- ============================================================

CREATE TABLE IF NOT EXISTS v2_frames (
    id            TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    unit_id       TEXT NOT NULL,
    type          TEXT NOT NULL,
    parent        TEXT,
    narrator      TEXT NOT NULL DEFAULT '',
    listener      TEXT NOT NULL DEFAULT '',
    open_at       INT  NOT NULL,
    close_at      INT,
    depth         INT  NOT NULL DEFAULT 0,
    summary       TEXT NOT NULL DEFAULT '',
    auto_closed   BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS v2_frames_doc_idx ON v2_frames (doc_id);

CREATE TABLE IF NOT EXISTS v2_entities (
    id            TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    canonical     TEXT NOT NULL,
    surfaces      JSONB NOT NULL DEFAULT '[]'::jsonb
);
CREATE INDEX IF NOT EXISTS v2_entities_doc_idx ON v2_entities (doc_id);

CREATE TABLE IF NOT EXISTS v2_events (
    id                TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    description       TEXT NOT NULL,
    event_type        TEXT NOT NULL DEFAULT 'other',
    first_offset      INT  NOT NULL DEFAULT 0,
    participants      JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding         vector(1536),
    partition_frame   TEXT
);
CREATE INDEX IF NOT EXISTS v2_events_doc_idx ON v2_events (doc_id);
CREATE INDEX IF NOT EXISTS v2_events_embedding_hnsw
    ON v2_events USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS v2_mentions (
    id                TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    event_id          TEXT REFERENCES v2_events(id) ON DELETE SET NULL,
    frame_id          TEXT,
    window_id         TEXT,
    char_start        INT  NOT NULL,
    char_end          INT  NOT NULL,
    para_id           TEXT,
    quote             TEXT NOT NULL DEFAULT '',
    description       TEXT NOT NULL DEFAULT '',
    mode              TEXT NOT NULL DEFAULT 'occurs',
    event_type        TEXT NOT NULL DEFAULT 'other',
    subject_entity    TEXT,
    participants      JSONB NOT NULL DEFAULT '[]'::jsonb,
    location          TEXT NOT NULL DEFAULT '',
    time_expressions  JSONB NOT NULL DEFAULT '[]'::jsonb,
    posthumous        BOOLEAN NOT NULL DEFAULT false,
    is_telling        BOOLEAN NOT NULL DEFAULT false,
    tells_frame       TEXT,
    sample_idx        INT  NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS v2_mentions_doc_idx ON v2_mentions (doc_id);
CREATE INDEX IF NOT EXISTS v2_mentions_event_idx ON v2_mentions (event_id);
CREATE INDEX IF NOT EXISTS v2_mentions_span_idx ON v2_mentions (doc_id, char_start);

CREATE TABLE IF NOT EXISTS v2_relations (
    id            BIGSERIAL PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    mention_a     TEXT NOT NULL,
    mention_b     TEXT NOT NULL,
    rel           TEXT NOT NULL,
    cue           TEXT NOT NULL,
    consistency   REAL NOT NULL DEFAULT 1.0,
    quote         TEXT NOT NULL DEFAULT '',
    char_start    INT,
    char_end      INT
);
CREATE INDEX IF NOT EXISTS v2_relations_doc_idx ON v2_relations (doc_id);

CREATE TABLE IF NOT EXISTS v2_graphs (
    doc_id          TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    graph           JSONB NOT NULL,
    stats           JSONB NOT NULL DEFAULT '{}'::jsonb,
    weights         JSONB NOT NULL DEFAULT '{}'::jsonb,
    prompt_version  TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
