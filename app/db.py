from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from neo4j import GraphDatabase

from .config import settings, ROOT

_pool: ConnectionPool | None = None
_neo4j = None


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.pg_dsn,
            min_size=1,
            max_size=settings.concurrency + 4,
            configure=_configure,
            open=True,
        )
    return _pool


@contextmanager
def pg():
    """Yields a dict-row cursor inside a transaction."""
    with pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            yield cur


def neo4j():
    global _neo4j
    if _neo4j is None:
        _neo4j = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _neo4j


def ensure_schema() -> None:
    """Idempotent migrations for already-provisioned Postgres volumes.
    Runs every infra/postgres/migrations/*.sql file in name order; each file
    must be safe to re-run (IF NOT EXISTS everywhere)."""
    with pg() as cur:
        cur.execute(
            """ALTER TABLE documents
               ADD COLUMN IF NOT EXISTS taxonomy JSONB NOT NULL DEFAULT '[]'::jsonb"""
        )
    mig_dir = ROOT / "infra" / "postgres" / "migrations"
    for f in sorted(mig_dir.glob("*.sql")):
        with pg() as cur:
            cur.execute(f.read_text(encoding="utf-8"))


def init_neo4j() -> None:
    """Apply constraints/indexes. Safe to call on every startup."""
    path = ROOT / "infra" / "neo4j" / "init.cypher"
    statements = [s.strip() for s in path.read_text().split(";") if s.strip()]
    with neo4j().session() as sess:
        for stmt in statements:
            sess.run(stmt)


def close() -> None:
    global _pool, _neo4j
    if _pool is not None:
        _pool.close()
        _pool = None
    if _neo4j is not None:
        _neo4j.close()
        _neo4j = None
