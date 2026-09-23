"""Persist build-health events so dropped or degraded input is always visible."""
import json

from . import llm
from .db import pg


def record(doc_id: str, kind: str, ref: str | None = None, detail: dict | None = None,
           job_id: str | None = None) -> None:
    try:
        with pg() as cur:
            cur.execute(
                """INSERT INTO build_events (doc_id, job_id, kind, ref, detail)
                   VALUES (%s,%s,%s,%s,%s)""",
                (doc_id, job_id, kind, ref, json.dumps(detail or {})),
            )
    except Exception:
        pass  # logging must never break a build


def flush_llm_events(doc_id: str, ref: str | None = None, job_id: str | None = None) -> None:
    for ev in llm.drain_events():
        kind = ev.pop("kind", "llm_event")
        record(doc_id, kind, ref, ev, job_id)


def summary(doc_id: str) -> dict:
    with pg() as cur:
        cur.execute(
            """SELECT kind, count(*) AS n FROM build_events
               WHERE doc_id = %s GROUP BY kind""", (doc_id,))
        return {r["kind"]: r["n"] for r in cur.fetchall()}
