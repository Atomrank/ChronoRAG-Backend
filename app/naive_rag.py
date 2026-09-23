"""
Naive RAG baseline — fixed-size character chunks, cosine top-k, similarity order.

Fairness rules (so any win for Kaalkram is real):
  * the model may say the passages do not settle the question;
  * chunks go into the prompt whole, not cut to a preview;
  * no fallback text is ever written by code — the answer is always the model's;
  * it returns the same structured label (before/after/cannot_determine) as Kaalkram.
The only intended weakness is the design itself: chunk retrieval by similarity,
passages presented in similarity order.
"""
import time

from pgvector.psycopg import Vector

from . import llm
from .config import settings
from .db import pg
from .ingest_v2 import Document, naive_chunks
from .schemas import Citation, NaiveAnswer, PipelineAnswer

SYSTEM = """You answer questions about a book using ONLY the numbered passages provided.

Rules:
- Ground every statement in the passages. Do not use outside knowledge of the book.
- The passages are ranked by similarity to the question, NOT in the order they appear
  in the book and NOT in story order.
- If the passages do not settle the answer, say so plainly and set relation to
  cannot_determine.
- Keep the answer to 2-5 sentences. Treat the text as literature under discussion."""


def build(doc_id: str, doc: Document, on_progress=None) -> int:
    """Chunk the cleaned v2 text, embed and store with character offsets."""
    chunks = naive_chunks(doc, settings.naive_chunk_chars, settings.naive_chunk_overlap)
    total = len(chunks)
    with pg() as cur:
        cur.execute("DELETE FROM naive_chunks WHERE doc_id = %s", (doc_id,))
    for i in range(0, total, 64):
        part = chunks[i:i + 64]
        vectors = llm.embed([c["content"] for c in part])
        with pg() as cur:
            cur.executemany(
                """INSERT INTO naive_chunks
                   (doc_id, chunk_index, page_start, page_end, content, char_start,
                    char_end, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                [(doc_id, c["chunk_index"], c["page_start"], c["page_end"], c["content"],
                  c["char_start"], c["char_end"], Vector(v)) for c, v in zip(part, vectors)],
            )
        if on_progress:
            on_progress(min(1.0, (i + len(part)) / max(1, total)),
                        f"embedded {min(i + len(part), total)}/{total} chunks")
    return total


def retrieve(doc_id: str, question: str, k: int) -> list[dict]:
    qvec = llm.embed([question])[0]
    with pg() as cur:
        cur.execute(
            """SELECT chunk_index, page_start, page_end, char_start, char_end, content,
                      1 - (embedding <=> %s) AS score
               FROM naive_chunks WHERE doc_id = %s
               ORDER BY embedding <=> %s LIMIT %s""",
            (Vector(qvec), doc_id, Vector(qvec), k),
        )
        return [dict(r) for r in cur.fetchall()]


def answer(doc_id: str, question: str, k_ctx: int | None = None,
           k_retrieve: int | None = None) -> PipelineAnswer:
    """k_ctx passages go to the model; k_retrieve (>= k_ctx) are returned for
    recall@k scoring. Retrieval is done once, then cut."""
    k_ctx = k_ctx or settings.naive_top_k
    k_retrieve = max(k_ctx, k_retrieve or k_ctx)
    t0 = time.perf_counter()
    before = llm.usage_snapshot()

    hits = retrieve(doc_id, question, k_retrieve)
    ctx = hits[:k_ctx]
    context = "\n\n---\n\n".join(
        f"[Passage {i + 1} | similarity {h['score']:.3f}]\n{h['content']}"
        for i, h in enumerate(ctx)
    )
    user = f"PASSAGES (similarity order, not story order):\n\n{context}\n\nQUESTION: {question}"
    result = llm.chat_structured(SYSTEM, user, NaiveAnswer, temperature=0.0,
                                 max_tokens=800, degrade_on_filter=True)
    after = llm.usage_snapshot()

    used = [n for n in result.used_passages if 1 <= n <= len(ctx)]
    return PipelineAnswer(
        pipeline="naive",
        answer=result.answer,
        relation=result.relation,
        confidence=max(0.0, min(1.0, result.confidence)),
        cited_spans=[[ctx[n - 1]["char_start"], ctx[n - 1]["char_end"]] for n in used],
        latency_ms=int((time.perf_counter() - t0) * 1000),
        prompt_tokens=after["prompt"] - before["prompt"],
        completion_tokens=after["completion"] - before["completion"],
        citations=[Citation(label=f"Passage {n}",
                            pages=list(range(ctx[n - 1]["page_start"], ctx[n - 1]["page_end"] + 1)))
                   for n in used],
        retrieved=[
            {"rank": i + 1, "unit_id": f"chunk_{h['chunk_index']}",
             "score": round(float(h["score"]), 4), "in_context": i < k_ctx,
             "spans": [[h["char_start"], h["char_end"]]],
             "tokens": max(1, len(h["content"]) // 4),
             "pages": [h["page_start"], h["page_end"]], "preview": h["content"][:280]}
            for i, h in enumerate(hits)
        ],
        trace=[
            f"Embedded the question ({settings.embed_dim}-dim).",
            f"Retrieved top {len(hits)} chunks by cosine similarity; sent the top "
            f"{len(ctx)} to the model in similarity order (not story order).",
        ],
    )
