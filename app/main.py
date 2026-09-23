import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import buildlog, db, docstore, evaluation, graph, jobs, llm, naive_rag, passes, query_engine
from .config import settings, DEFAULT_TAXONOMY
from .ingest_v2 import doc_id_for, extract_document
from .schemas import CompareResponse, DocumentOut, EventOut, JobOut, TaxonomyStageOut


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.pool()
    db.ensure_schema()
    db.init_neo4j()
    yield
    db.close()


app = FastAPI(title="Kaalkram API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _taxonomy_out(raw) -> list[TaxonomyStageOut]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    out: list[TaxonomyStageOut] = []
    for item in raw or []:
        if isinstance(item, str):
            out.append(TaxonomyStageOut(name=item))
        elif isinstance(item, dict) and item.get("name"):
            out.append(TaxonomyStageOut(
                name=item["name"],
                description=item.get("description") or "",
                is_framing=bool(item.get("is_framing", False)),
            ))
    return out


# ------------------------------------------------------------
def doc_pages(doc_id: str) -> list[str]:
    with db.pg() as cur:
        cur.execute(
            "SELECT content FROM pages WHERE doc_id = %s ORDER BY page_no", (doc_id,)
        )
        rows = cur.fetchall()
    if not rows:
        raise HTTPException(404, "document not found")
    return [r["content"] for r in rows]


# ------------------------------------------------------------
# Documents
# ------------------------------------------------------------
@app.post("/api/documents", response_model=DocumentOut)
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf is supported")

    dest = settings.upload_path / file.filename
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    doc_id = doc_id_for(dest)
    doc = extract_document(dest)
    if not doc.text.strip():
        raise HTTPException(
            400, "no extractable text — this looks like a scanned PDF; "
                 "upload a version with an OCR text layer")
    # v1 still reads per-page text: give it slices of the same cleaned text
    # so every pipeline reads identical input.
    n_pages = doc.stats.get("pages", len(doc.pages))
    span = {p.no: (p.start, p.end) for p in doc.pages}
    pages = [doc.text[span[i][0]:span[i][1]] if i in span else "" for i in range(1, n_pages + 1)]

    title = Path(file.filename).stem.replace("_", " ").title()
    with db.pg() as cur:
        cur.execute(
            """INSERT INTO documents (id, title, filename, page_count)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title""",
            (doc_id, title, file.filename, len(pages)),
        )
        cur.execute("DELETE FROM pages WHERE doc_id = %s", (doc_id,))
        cur.executemany(
            "INSERT INTO pages (doc_id, page_no, content) VALUES (%s,%s,%s)",
            [(doc_id, i, txt) for i, txt in enumerate(pages, start=1)],
        )
    docstore.save_document(doc_id, doc)

    return DocumentOut(id=doc_id, title=title, filename=file.filename,
                       page_count=len(pages))


@app.get("/api/documents", response_model=list[DocumentOut])
async def list_documents():
    with db.pg() as cur:
        cur.execute(
            """SELECT d.id, d.title, d.filename, d.page_count, d.taxonomy,
                      (SELECT count(*) FROM naive_chunks n WHERE n.doc_id = d.id) AS chunks,
                      (SELECT count(*) FROM events e WHERE e.doc_id = d.id) AS events
               FROM documents d ORDER BY d.uploaded_at DESC"""
        )
        rows = cur.fetchall()
    return [
        DocumentOut(
            id=r["id"], title=r["title"], filename=r["filename"],
            page_count=r["page_count"],
            naive_ready=r["chunks"] > 0,
            kaalkram_ready=r["events"] > 0,
            event_count=r["events"],
            taxonomy=_taxonomy_out(r["taxonomy"]),
        )
        for r in rows
    ]


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    with db.pg() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    with db.neo4j().session() as sess:
        sess.run("MATCH (n) WHERE n.doc_id = $d DETACH DELETE n", d=doc_id)
    return {"deleted": doc_id}


# ------------------------------------------------------------
# Build jobs
# ------------------------------------------------------------
@app.post("/api/documents/{doc_id}/build/{kind}", response_model=JobOut)
async def build(doc_id: str, kind: str, bg: BackgroundTasks):
    if kind not in ("naive", "kaalkram"):
        raise HTTPException(400, "kind must be 'naive' or 'kaalkram'")
    doc_pages(doc_id)                      # 404s if unknown
    job_id = jobs.create(doc_id, kind)
    bg.add_task(jobs.run_naive if kind == "naive" else jobs.run_kaalkram, job_id, doc_id)
    return JobOut(**{**jobs.get(job_id), "detail": {}})


@app.get("/api/jobs/{job_id}", response_model=JobOut)
async def job_status(job_id: str):
    row = jobs.get(job_id)
    if not row:
        raise HTTPException(404, "job not found")
    return JobOut(id=row["id"], doc_id=row["doc_id"], kind=row["kind"],
                  status=row["status"], stage=row["stage"],
                  progress=row["progress"], detail=row["detail"],
                  error=row["error"])


@app.get("/api/documents/{doc_id}/jobs")
async def doc_jobs(doc_id: str):
    return {k: jobs.latest(doc_id, k) for k in ("naive", "kaalkram")}


# ------------------------------------------------------------
# Query
# ------------------------------------------------------------
class Ask(BaseModel):
    question: str


@app.post("/api/documents/{doc_id}/ask/naive")
async def ask_naive(doc_id: str, body: Ask):
    try:
        return naive_rag.answer(doc_id, body.question)
    except llm.ContentFilterError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/documents/{doc_id}/ask/kaalkram")
async def ask_kaalkram(doc_id: str, body: Ask):
    try:
        return query_engine.answer(doc_id, body.question)
    except llm.ContentFilterError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/documents/{doc_id}/compare", response_model=CompareResponse)
async def compare(doc_id: str, body: Ask):
    try:
        naive = naive_rag.answer(doc_id, body.question)
        kaal = query_engine.answer(doc_id, body.question)
    except llm.ContentFilterError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
    with db.pg() as cur:
        cur.execute(
            """INSERT INTO query_runs
               (doc_id, question, naive_answer, naive_ms, naive_tokens,
                kaal_answer, kaal_ms, kaal_tokens)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (doc_id, body.question, naive.answer, naive.latency_ms,
             naive.prompt_tokens + naive.completion_tokens,
             kaal.answer, kaal.latency_ms,
             kaal.prompt_tokens + kaal.completion_tokens),
        )
    return CompareResponse(question=body.question, naive=naive, kaalkram=kaal)


# ------------------------------------------------------------
# Timeline / graph / metrics
# ------------------------------------------------------------
@app.get("/api/documents/{doc_id}/events", response_model=list[EventOut])
async def events(doc_id: str):
    return [EventOut(**e) for e in passes.load_events(doc_id)]


@app.get("/api/documents/{doc_id}/graph")
async def event_graph(doc_id: str):
    return graph.fetch_graph(doc_id)


@app.get("/api/documents/{doc_id}/observations")
async def observations(doc_id: str, limit: int = 20):
    with db.pg() as cur:
        cur.execute(
            """SELECT window_id, page_start, page_end, raw_text
               FROM observations WHERE doc_id = %s
               ORDER BY page_start LIMIT %s""",
            (doc_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]


@app.get("/api/documents/{doc_id}/metrics")
async def metrics(doc_id: str):
    out = evaluation.summarise(doc_id)
    out["build_events"] = buildlog.summary(doc_id)
    return out


@app.get("/api/documents/{doc_id}/eval-runs")
async def eval_runs(doc_id: str):
    with db.pg() as cur:
        cur.execute(
            """SELECT run_id, created_at, pipeline, gold_set_id, git_commit, status, summary
               FROM eval_runs WHERE doc_id = %s ORDER BY created_at DESC""", (doc_id,))
        return [dict(r) for r in cur.fetchall()]


@app.get("/api/documents/{doc_id}/taxonomy")
async def document_taxonomy(doc_id: str):
    doc_pages(doc_id)  # 404 if missing
    stages = passes.load_taxonomy(doc_id)
    return {"doc_id": doc_id, "stages": _taxonomy_out(stages)}


@app.get("/api/taxonomy")
async def taxonomy():
    """Fallback default stages (not book-specific). Prefer /api/documents/{id}/taxonomy."""
    return {
        "stages": DEFAULT_TAXONOMY,
        "note": "global fallback only; use GET /api/documents/{doc_id}/taxonomy after Build Kaalkram",
    }


@app.get("/api/health")
async def health():
    ok_pg = ok_neo = True
    try:
        with db.pg() as cur:
            cur.execute("SELECT 1")
    except Exception:
        ok_pg = False
    try:
        with db.neo4j().session() as s:
            s.run("RETURN 1").consume()
    except Exception:
        ok_neo = False
    return {"postgres": ok_pg, "neo4j": ok_neo,
            "chat_deployment": settings.azure_chat_deployment}
