import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from pgvector.psycopg import Vector

from . import buildlog, llm
from .config import settings, stage_index, stage_names, DEFAULT_TAXONOMY
from .db import pg
from .schemas import BatchMergeResponse, TaxonomyProposal

# ============================================================
# PASS 1 — parallel skim
# ============================================================
PASS1_SYSTEM_EXHAUSTIVE = """You are a rapid-scanning literary agent. Read the text slice provided.
The text contains inline [PAGE N] markers showing exactly which PDF page each passage comes from.

List EVERY narrated event in this slice — every action, happening, or state change — as a
bullet. Include events inside flashbacks/recollections, prophecies/foretellings, counterfactuals
("suppose …"), separate tales, and "meanwhile" / "at the same hour" / "in another part of …"
stretches. Do not skip an event because it seems minor. Do not summarise several actions into
one bullet.

IMPORTANT: For every bullet point, record the [PAGE N] number(s) where that event occurs.
Format each bullet exactly as: [PAGE N] <observation>
If an event spans multiple pages: [PAGE N, M] <observation>
Do not output anything except the bullet list."""

PASS1_SYSTEM_SALIENT = """You are a rapid-scanning literary agent. Skim the text slice provided.
The text contains inline [PAGE N] markers showing exactly which PDF page each passage comes from.

Write a concise, chronological bulleted list of the plot-significant events and character
actions in this slice. Prefer milestones over atmosphere.

IMPORTANT: For every bullet point, record the [PAGE N] number(s) where that event occurs.
Format each bullet exactly as: [PAGE N] <observation>
If an event spans multiple pages: [PAGE N, M] <observation>
Do not output anything except the bullet list."""


def pass1_system() -> str:
    if (settings.extraction_mode or "exhaustive").lower() == "salient":
        return PASS1_SYSTEM_SALIENT
    return PASS1_SYSTEM_EXHAUSTIVE


# Back-compat alias (resolved at import; prefer pass1_system() at call time).
PASS1_SYSTEM = PASS1_SYSTEM_EXHAUSTIVE


def _cache_file(doc_id: str, name: str):
    return settings.cache_path / f"{doc_id}_{name}.json"


def _load_cache(doc_id: str, name: str) -> dict:
    f = _cache_file(doc_id, name)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(doc_id: str, name: str, data: dict) -> None:
    _cache_file(doc_id, name).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_pass1(doc_id: str, windows: list[dict], on_progress=None) -> list[dict]:
    """Skim every window concurrently. Checkpointed and resumable."""
    cache = _load_cache(doc_id, "pass1")
    cache_lock = threading.Lock()
    pending = [w for w in windows if w["id"] not in cache]
    done = len(windows) - len(pending)

    if on_progress and done:
        on_progress(done / len(windows), f"resumed: {done}/{len(windows)} windows cached")

    def work(w: dict) -> tuple[str, str | None]:
        user = f"[Text Window: Pages {w['start']} to {w['end']}]\n\n{w['text']}"
        # window id is typically "win_0", "win_1", … — parse index when possible
        try:
            widx = int(str(w["id"]).rsplit("_", 1)[-1])
        except ValueError:
            widx = w["id"]
        call_meta = {
            "pipeline": "kaalkram_v1",
            "phase": "pass1",
            "doc_id": doc_id,
            "window_id": w["id"],
            "window_index": widx,
            "window_start": w.get("char_start", w["start"]),
            "window_end": w.get("char_end", w["end"]),
            "window_chars": len(w.get("text") or ""),
            "max_tokens_setting": settings.extraction_max_tokens(),
        }
        try:
            return w["id"], llm.chat(pass1_system(), user,
                                     max_tokens=settings.extraction_max_tokens(),
                                     call_meta=call_meta)
        except (llm.ContentFilterError, llm.InputTooLongError,
                llm.OutputTruncatedError) as exc:
            # Recorded, not hidden: shows up in build_events and the metrics screen.
            kind = ("extract_output_truncated"
                    if isinstance(exc, llm.OutputTruncatedError) else "failed_window")
            buildlog.record(doc_id, kind, w["id"],
                            {"pass": 1, "error": type(exc).__name__,
                             "pages": [w["start"], w["end"]], **call_meta})
            return w["id"], None

    if pending:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            futures = {pool.submit(work, w): w for w in pending}
            for fut in as_completed(futures):
                w = futures[fut]
                wid, text = fut.result()
                if text is None:
                    continue
                with cache_lock:
                    cache[wid] = {"start": w["start"], "end": w["end"], "content": text}
                    _save_cache(doc_id, "pass1", cache)
                    done += 1
                if on_progress:
                    on_progress(done / len(windows),
                                f"pass 1: {done}/{len(windows)} windows read")

    # Persist observations for the audit panel
    with pg() as cur:
        cur.execute("DELETE FROM observations WHERE doc_id = %s", (doc_id,))
        cur.executemany(
            """INSERT INTO observations (doc_id, window_id, page_start, page_end, raw_text)
               VALUES (%s,%s,%s,%s,%s)""",
            [(doc_id, w["id"], cache[w["id"]]["start"], cache[w["id"]]["end"],
              cache[w["id"]]["content"]) for w in windows if w["id"] in cache],
        )

    return [
        {"window_id": w["id"], "start": w["start"], "end": w["end"],
         "content": cache[w["id"]]["content"]}
        for w in windows if w["id"] in cache
    ]


# ============================================================
# PASS 0 — invent a book-specific stage taxonomy
# ============================================================
PASS0_SYSTEM = """You design a coarse STORY-WORLD timeline taxonomy for ONE narrative book.

Goal: 5 to 7 ordered stages that partition the plot by when events happen in the story world,
NOT by print/chapter order. Flashbacks, framing prologues, and epilogues must sit at their
story-time position (often the last stage if they are aftermath/frame).

Rules:
- Stages must be specific to THIS book (characters, settings, plot beats) — not generic
  "rising action / climax" labels unless the book truly has no clearer beats.
- Each stage name: 2–6 words, Title Case, optionally "A / B" form.
- Descriptions: concrete cues an annotator can match (who/what belongs here).
- Mark is_framing=true only for stages that collect out-of-order printed framing
  (epilogue, tourists seeing the aftermath, hospital frame, "years later", etc.).
- Cover the whole narrative arc; no huge gaps; no overlapping stages.
- Return stages in story-world chronological order (earliest → latest)."""


def _normalize_stages(stages: list) -> list[dict]:
    out: list[dict] = []
    for s in stages:
        if isinstance(s, dict):
            name = str(s.get("name") or "").strip()
            if not name:
                continue
            out.append({
                "name": name,
                "description": str(s.get("description") or "").strip(),
                "is_framing": bool(s.get("is_framing", False)),
            })
        else:
            name = str(getattr(s, "name", "") or "").strip()
            if not name:
                continue
            out.append({
                "name": name,
                "description": str(getattr(s, "description", "") or "").strip(),
                "is_framing": bool(getattr(s, "is_framing", False)),
            })
    # Enforce 5–7 by trimming or padding with defaults (rare).
    if len(out) > 7:
        out = out[:7]
    if len(out) < 5:
        for name in DEFAULT_TAXONOMY:
            if len(out) >= 5:
                break
            if name not in {s["name"] for s in out}:
                out.append({"name": name, "description": "", "is_framing": name == DEFAULT_TAXONOMY[-1]})
    if out and not any(s["is_framing"] for s in out):
        out[-1]["is_framing"] = True
    return out


def _observation_digest(observations: list[dict], *, max_chars: int = 12000) -> str:
    """Spread sample across the book so Pass 0 sees beginning, middle, and end."""
    if not observations:
        return "(no observations)"
    n = len(observations)
    if n <= 8:
        picks = observations
    else:
        idxs = sorted({0, 1, n // 4, n // 2, (3 * n) // 4, n - 2, n - 1})
        picks = [observations[i] for i in idxs if 0 <= i < n]
    parts: list[str] = []
    budget = max_chars
    for o in picks:
        block = f"--- pages {o['start']}-{o['end']} ---\n{o['content']}"
        if len(block) > budget:
            block = block[: max(0, budget - 20)] + "\n…"
        parts.append(block)
        budget -= len(block)
        if budget <= 200:
            break
    return "\n\n".join(parts)


def save_taxonomy(doc_id: str, stages: list[dict]) -> None:
    with pg() as cur:
        cur.execute(
            "UPDATE documents SET taxonomy = %s::jsonb WHERE id = %s",
            (json.dumps(stages), doc_id),
        )


def load_taxonomy(doc_id: str) -> list[dict]:
    with pg() as cur:
        cur.execute("SELECT taxonomy FROM documents WHERE id = %s", (doc_id,))
        row = cur.fetchone()
    raw = (row or {}).get("taxonomy") if row else None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    if not raw:
        return [
            {"name": n, "description": "", "is_framing": i == len(DEFAULT_TAXONOMY) - 1}
            for i, n in enumerate(DEFAULT_TAXONOMY)
        ]
    return _normalize_stages(list(raw))


def run_pass0(
    doc_id: str,
    title: str,
    observations: list[dict],
    on_progress=None,
) -> list[dict]:
    """
    Infer a 5–7 stage taxonomy for this document from Pass-1 observations.
    Cached and written to documents.taxonomy.
    """
    cache = _load_cache(doc_id, "pass0")
    if cache.get("stages"):
        stages = _normalize_stages(cache["stages"])
        save_taxonomy(doc_id, stages)
        if on_progress:
            on_progress(1.0, f"pass 0: reused {len(stages)} cached stages")
        return stages

    if on_progress:
        on_progress(0.1, "pass 0: inventing story-stage taxonomy")

    digest = _observation_digest(observations)
    user = (
        f"BOOK TITLE: {title or doc_id}\n"
        f"OBSERVATION WINDOWS: {len(observations)}\n\n"
        f"PASS-1 SKIM NOTES (sampled across the book):\n{digest}\n\n"
        "Propose the taxonomy for THIS book only."
    )
    proposal = llm.chat_structured(PASS0_SYSTEM, user, TaxonomyProposal, max_tokens=2000)
    stages = _normalize_stages(proposal.stages)
    _save_cache(doc_id, "pass0", {"stages": stages, "title": title})
    save_taxonomy(doc_id, stages)
    if on_progress:
        on_progress(1.0, f"pass 0: {len(stages)} stages — " + ", ".join(s["name"] for s in stages))
    return stages


# ============================================================
# PASS 2 — stateful merge and deduplication
# ============================================================
def _pass2_system(taxonomy: list[dict]) -> str:
    lines = []
    for i, s in enumerate(taxonomy, start=1):
        desc = s.get("description") or "events that belong in this story phase"
        frame = " [FRAMING/OUT-OF-PRINT-ORDER]" if s.get("is_framing") else ""
        lines.append(f'{i}. "{s["name"]}"{frame} — {desc}')
    last = taxonomy[-1]["name"] if taxonomy else DEFAULT_TAXONOMY[-1]
    stage_block = "\n".join(lines)
    exhaustive = (settings.extraction_mode or "exhaustive").lower() != "salient"
    if exhaustive:
        category_block = """CATEGORY (exhaustive mode):
- Emit ONE instruction for EVERY observation. Do not drop or skip any observation.
- "minor": the default for ordinary narrated events (most observations).
- "major": only clear plot milestones (optional label; never discard an observation to
  keep the major count low). Prefer "minor" when unsure.
There is NO target count of majors. Completeness matters more than salience."""
    else:
        category_block = """CATEGORY (salient mode):
- "major": core structural milestones that drive the main plot - inciting incidents, turning
  points, climax, resolution, irreversible decisions. A novel has roughly 8-15 of these. Be strict.
- "minor": subplots, backstory reveals, secondary interactions, atmosphere, comic relief."""
    return f"""You are a structural data engineer managing a narrative timeline database.
You receive (a) a light index of events ALREADY in the database, and (b) a new set of raw
observations. Produce one instruction per observation describing how to integrate it.

STRICT TIMELINE ANCHOR TAXONOMY - timeline_anchor must be EXACTLY one of:
{stage_block}

Assign the anchor by WHERE THE EVENT SITS IN STORY-WORLD TIME, never by where it is printed.
Framing or aftermath scenes printed early (or late) still belong in their story-time stage
(usually "{last}" if marked FRAMING).

{category_block}

DUPLICATE DETECTION:
Compare each observation against the baseline index. If it describes the SAME event frame as an
existing entry (same participants, same action, adjacent pages), set is_duplicate=true and put the
EXACT baseline event_name in matched_event_name. Otherwise is_duplicate=false and
matched_event_name must be an empty string.

LOCATION FORMAT: specific-to-general, e.g. "skiff, Gulf Stream off Cuba". Never let a room name
replace the setting.

CAUSALITY: antecedent_cause = what triggered this event. consequent_effect = what it leads to.
Both must be grounded in the text, not invented.

SOURCE PAGES: copy the [PAGE N] numbers from the observation bullet verbatim."""


def _light_index(events: list[dict]) -> list[dict]:
    return [
        {"event_name": e["event_name"], "category": e["category"],
         "timeline_anchor": e["timeline_anchor"], "source_pages": e["source_pages"]}
        for e in events
    ]


def _native_merge(events: list[dict], resp: BatchMergeResponse) -> tuple[int, int]:
    """Apply LLM decisions in plain Python. Returns (added, merged)."""
    by_name = {e["event_name"]: e for e in events}
    added = merged = 0

    for ins in resp.instructions:
        inf = ins.inferred_event
        pages = sorted(set(int(p) for p in inf.source_pages if p and p > 0))

        if ins.is_duplicate and ins.matched_event_name:
            target = by_name.get(ins.matched_event_name)
            if target is not None:
                target["source_pages"] = sorted(set(target["source_pages"] + pages))
                if inf.antecedent_cause and inf.antecedent_cause not in target["antecedent_cause"]:
                    target["antecedent_cause"] = (
                        f"{target['antecedent_cause']} | {inf.antecedent_cause}".strip(" |")
                    )
                if inf.consequent_effect and inf.consequent_effect not in target["consequent_effect"]:
                    target["consequent_effect"] = (
                        f"{target['consequent_effect']} | {inf.consequent_effect}".strip(" |")
                    )
                for c in inf.characters_present:
                    if c not in target["characters"]:
                        target["characters"].append(c)
                target["merge_count"] += 1
                merged += 1
                continue
            # matched name not found -> fall through and add as new

        record = {
            "id": f"ev_{uuid.uuid4().hex[:12]}",
            "event_name": inf.event_name,
            "category": ins.category,
            "timeline_anchor": inf.timeline_anchor,
            "location": inf.location,
            "characters": list(inf.characters_present),
            "core_event": inf.core_event,
            "antecedent_cause": inf.antecedent_cause,
            "consequent_effect": inf.consequent_effect,
            "source_pages": pages,
            "merge_count": 1,
        }
        events.append(record)
        by_name[record["event_name"]] = record
        added += 1

    return added, merged


def run_pass2(
    doc_id: str,
    observations: list[dict],
    taxonomy: list[dict] | None = None,
    on_progress=None,
) -> list[dict]:
    """
    Parallel fetch, sequential commit.

    Several batches are sent to the model at once against a FROZEN snapshot of the
    timeline index, but their decisions are applied one at a time. That keeps the
    merge deterministic and free of race conditions.
    """
    tax = _normalize_stages(taxonomy) if taxonomy else load_taxonomy(doc_id)
    system = _pass2_system(tax)

    bs = settings.pass2_batch_size
    batches = [
        "\n\n".join(
            f"--- Observations (pages {o['start']}-{o['end']}) ---\n{o['content']}"
            for o in observations[i:i + bs]
        )
        for i in range(0, len(observations), bs)
    ]
    total = len(batches)
    events: list[dict] = []
    stats = {"added": 0, "merged": 0, "failed": 0}

    def fetch(idx: int, index_snapshot: list[dict], text: str):
        user = (
            f"BASELINE TIMELINE INDEX ({len(index_snapshot)} events):\n"
            f"{json.dumps(index_snapshot, indent=2, ensure_ascii=False)}\n\n"
            f"NEW OBSERVATIONS TO RECONCILE:\n{text}"
        )
        call_meta = {
            "pipeline": "kaalkram_v1",
            "phase": "pass2",
            "doc_id": doc_id,
            "window_id": f"pass2_batch_{idx}",
            "window_index": idx,
            "window_start": None,
            "window_end": None,
            "window_chars": len(text),
            "max_tokens_setting": settings.extraction_max_tokens(),
        }
        try:
            return idx, llm.chat_structured(
                system, user, BatchMergeResponse,
                max_tokens=settings.extraction_max_tokens(),
                call_meta=call_meta)
        except Exception as exc:
            kind = ("extract_output_truncated"
                    if isinstance(exc, llm.OutputTruncatedError) else "failed_window")
            buildlog.record(doc_id, kind, f"pass2_batch_{idx}",
                            {"pass": 2, "error": type(exc).__name__,
                             "msg": str(exc)[:300], **call_meta})
            return idx, None

    processed = 0
    for start in range(0, total, settings.concurrency):
        group = list(enumerate(batches[start:start + settings.concurrency], start=start))
        snapshot = _light_index(events)      # frozen for this whole group

        results: dict[int, BatchMergeResponse | None] = {}
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            futures = [pool.submit(fetch, i, snapshot, t) for i, t in group]
            for fut in as_completed(futures):
                i, parsed = fut.result()
                results[i] = parsed

        for i, _ in group:                   # commit in deterministic order
            parsed = results.get(i)
            if parsed is None:
                stats["failed"] += 1
            else:
                a, m = _native_merge(events, parsed)
                stats["added"] += a
                stats["merged"] += m
            processed += 1
            if on_progress:
                on_progress(processed / total,
                            f"pass 2: batch {processed}/{total} "
                            f"({len(events)} events, {stats['merged']} merges)")

        _save_cache(doc_id, "pass2", {"events": events, "stats": stats, "taxonomy": tax})

    return events


# ============================================================
# PASS 3 — deterministic chronology
# ============================================================
# Book-agnostic cues that an event is framing / out-of-print-order aftermath.
GENERIC_FRAME_HINTS = (
    "years later", "epilogue", "prologue frame", "frame narrative",
    "framing device", "looking back", "afterwards", "aftermath",
)


def run_pass3(events: list[dict], taxonomy: list[dict] | None = None) -> list[dict]:
    """
    Two deterministic steps, no LLM:
      1. Assign a stage_order from the taxonomy, overriding obvious framing devices.
      2. Sort by (stage_order, first source page).
    """
    tax = _normalize_stages(taxonomy) if taxonomy else [
        {"name": n, "description": "", "is_framing": False} for n in DEFAULT_TAXONOMY
    ]
    names = stage_names(tax)
    framing_names = [s["name"] for s in tax if s.get("is_framing")]
    frame_target = framing_names[-1] if framing_names else names[-1]

    for e in events:
        e["source_pages"] = sorted(set(e.get("source_pages") or []))
        e["first_page"] = e["source_pages"][0] if e["source_pages"] else 0

        blob = f"{e['event_name']} {e['core_event']}".lower()
        if any(h in blob for h in GENERIC_FRAME_HINTS):
            e["timeline_anchor"] = frame_target
        e["stage_order"] = stage_index(e["timeline_anchor"], tax)

    events.sort(key=lambda e: (e["stage_order"], e["first_page"], e["event_name"]))
    return events


# ============================================================
# Persistence
# ============================================================
def persist_events(doc_id: str, events: list[dict], on_progress=None) -> None:
    texts = [
        f"{e['event_name']}. {e['core_event']} "
        f"Characters: {', '.join(e['characters'])}. Location: {e['location']}."
        for e in events
    ]
    vectors: list = []
    for i in range(0, len(texts), 64):
        vectors.extend(llm.embed(texts[i:i + 64]))
        if on_progress:
            on_progress(min(1.0, (i + 64) / max(1, len(texts))),
                        f"embedding events {min(i + 64, len(texts))}/{len(texts)}")

    with pg() as cur:
        cur.execute("DELETE FROM events WHERE doc_id = %s", (doc_id,))
        cur.executemany(
            """INSERT INTO events
               (id, doc_id, event_name, category, timeline_anchor, stage_order,
                location, characters, core_event, antecedent_cause, consequent_effect,
                source_pages, first_page, merge_count, embedding)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [
                (e["id"], doc_id, e["event_name"], e["category"], e["timeline_anchor"],
                 e["stage_order"], e["location"], e["characters"], e["core_event"],
                 e["antecedent_cause"], e["consequent_effect"], e["source_pages"],
                 e["first_page"], e["merge_count"], Vector(v))
                for e, v in zip(events, vectors)
            ],
        )


def load_events(doc_id: str) -> list[dict]:
    with pg() as cur:
        cur.execute(
            """SELECT id, event_name, category, timeline_anchor, stage_order, location,
                      characters, core_event, antecedent_cause, consequent_effect,
                      source_pages, first_page, merge_count
               FROM events WHERE doc_id = %s
               ORDER BY stage_order, first_page, event_name""",
            (doc_id,),
        )
        return [dict(r) for r in cur.fetchall()]
