"""
Seeded synthetic narrative benchmark for Kaalkram v2.

Produces a PDF (headings + body) and a verified gold JSONL with exact quotes.
All entity/place names are invented; no book-specific content.

  python -m app.synth --seed 1 --events 150 --flashback-rate 0.2 --nesting 2 \\
                      --out data/synth/s1
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from .config import settings
from .eval_runner import ORDER_TEMPLATE, GoldEvidence, GoldEvent, GoldQuestion
from .ingest_v2 import locate_quote
from .textutil import strip_discourse_prefix, strip_event_ids

# Invented name pools only (tests/synth).
_GIVEN = (
    "Mira", "Tovin", "Selka", "Brann", "Lira", "Corin", "Nessa", "Perrin",
    "Yara", "Dune", "Fenn", "Osa", "Kale", "Rin", "Vesper", "Thane",
    "Elowen", "Joric", "Sable", "Wynn",
)
_FAMILY = (
    "Harth", "Vellum", "Ashford", "Quill", "Marrow", "Cinder", "Pelt",
    "Rook", "Silt", "Bramble", "Wold", "Kestrel", "Tarn", "Gossamer",
)
_PLACES = (
    "Riverbend", "Ashmoor", "Glassport", "Highfen", "Copperford", "Mistfall",
    "Stoneferry", "Willowreach", "Northspire", "Saltmarsh", "Elmholt",
    "Brightmere", "Ironvale", "Duskwatch", "Fairhollow",
)
_VERBS = (
    "opened the gate", "signed the pact", "crossed the bridge", "lit the beacon",
    "called the council", "sealed the letter", "raised the banner", "left the harbor",
    "mended the roof", "planted the orchard", "found the token", "spoke the vow",
    "broke the silence", "shared the bread", "drew the map", "sounded the horn",
)
_CONNECTIVES_BEFORE = (
    "Before that,", "Earlier,", "Prior to this,", "First,",
)
_CONNECTIVES_AFTER = (
    "After that,", "Afterwards,", "Later,", "Then,", "Next,",
)

# Past-tense verb head → gerund for question phrasing (synth verbs only).
_PAST_TO_GERUND = {
    "opened": "opening", "signed": "signing", "crossed": "crossing",
    "lit": "lighting", "called": "calling", "sealed": "sealing",
    "raised": "raising", "left": "leaving", "mended": "mending",
    "planted": "planting", "found": "finding", "spoke": "speaking",
    "broke": "breaking", "shared": "sharing", "drew": "drawing",
    "sounded": "sounding",
}


def _gerund_np(verb_phrase: str) -> str:
    """'shared the bread' → 'sharing of the bread'; 'crossed the bridge' → 'crossing of the bridge'."""
    parts = verb_phrase.strip().split(None, 1)
    if not parts:
        return verb_phrase
    head, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
    gerund = _PAST_TO_GERUND.get(head)
    if gerund is None:
        gerund = head[:-1] + "ing" if head.endswith("e") else head + "ing"
    if rest.lower().startswith("the "):
        return f"{gerund} of {rest}"
    return f"{gerund} {rest}".strip()


def _verb_phrase_from_event(e: SynthEvent) -> str:
    """Extract the bare action phrase (no name/place/connective/E-tag)."""
    if e.event_type == "birth":
        return "was born"
    if e.event_type == "death":
        return "died"
    who = e.actors[0] if e.actors else ""
    q = strip_discourse_prefix(strip_event_ids(e.quote).rstrip("."))
    # Meanwhile form already stripped prefix → "Name shared the bread"
    # Same-hour / standard → "Name crossed the bridge at Place"
    if who and q.startswith(who):
        rest = q[len(who):].strip()
        if e.place and rest.endswith(f"at {e.place}"):
            rest = rest[: -len(f"at {e.place}")].strip()
        return rest
    # Fallback: last known desc without connective
    d = strip_discourse_prefix(strip_event_ids(e.desc).rstrip("."))
    if who and d.startswith(who):
        return d[len(who):].strip()
    return d


def question_phrase(e: SynthEvent) -> str:
    """Clean event description for questions: no discourse prefix, no [E####], gerund NP.

    Example: \"Brann Rook's sharing of the bread at Fairhollow\"
    """
    who = e.actors[0] if e.actors else "someone"
    place = e.place or ""
    if e.event_type == "birth":
        return f"{who}'s birth in {place}" if place else f"{who}'s birth"
    if e.event_type == "death":
        return f"{who}'s death at {place}" if place else f"{who}'s death"
    verb = _verb_phrase_from_event(e)
    gerund = _gerund_np(verb)
    if place:
        return f"{who}'s {gerund} at {place}"
    return f"{who}'s {gerund}"


@dataclass
class SynthEvent:
    id: str
    desc: str
    story_time: float
    actors: list[str]
    place: str
    event_type: str = "other"          # birth | death | other
    parallel_group: str | None = None
    quote: str = ""                    # exact sentence placed in the PDF
    discourse_order: int = -1          # filled at render time
    char_start: int | None = None
    char_end: int | None = None
    frame_kind: str = "main"           # main | recollection | separate_tale | prediction | hypothetical
    narrator: str = ""
    fulfilled_of: str | None = None    # prophecy event id this fulfills


@dataclass
class SynthParams:
    seed: int
    events: int
    flashback_rate: float
    nesting: int
    prophecy_rate: float
    hypothetical_rate: float
    separate_tale_rate: float
    parallel_rate: float


@dataclass
class SynthResult:
    params: SynthParams
    events: list[SynthEvent]
    text: str
    gold: list[GoldQuestion]
    strata_counts: dict[str, int] = field(default_factory=dict)
    frame_counts: dict[str, int] = field(default_factory=dict)


def _names(rng: random.Random, n: int) -> list[str]:
    used: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        name = f"{rng.choice(_GIVEN)} {rng.choice(_FAMILY)}"
        if name not in used:
            used.add(name)
            out.append(name)
    return out


def _places(rng: random.Random, n: int) -> list[str]:
    pool = list(_PLACES)
    rng.shuffle(pool)
    while len(pool) < n:
        pool.append(f"Place{len(pool)}")
    return pool[:n]


def _sentence(who: str, verb: str, place: str, *, tag: str, connective: str = "") -> str:
    # tag (event id) keeps locate_quote unique across similar actions
    core = f"{who} {verb} at {place} [{tag}]."
    return f"{connective} {core}".strip() if connective else core


def build_timeline(params: SynthParams) -> list[SynthEvent]:
    rng = random.Random(params.seed)
    n_people = max(8, params.events // 8)
    people = _names(rng, n_people)
    places = _places(rng, max(5, n_people // 2))

    events: list[SynthEvent] = []
    t = 0.0
    # Births for a subset
    for i, person in enumerate(people[: max(3, n_people // 3)]):
        place = rng.choice(places)
        eid = f"E{len(events) + 1:04d}"
        quote = f"{person} was born in {place} [{eid}]."
        events.append(SynthEvent(
            id=eid, desc=f"birth of {person}",
            story_time=t, actors=[person], place=place, event_type="birth",
            quote=quote,
        ))
        t += 1.0

    # Main chronological actions
    while len(events) < params.events - max(2, int(params.events * 0.05)):
        who = rng.choice(people)
        place = rng.choice(places)
        verb = rng.choice(_VERBS)
        eid = f"E{len(events) + 1:04d}"
        quote = _sentence(who, verb, place, tag=eid)
        events.append(SynthEvent(
            id=eid,
            desc=f"{who} {verb}",
            story_time=t, actors=[who], place=place, quote=quote,
        ))
        t += 1.0

    # Deaths near the end of story time
    for person in people[: max(2, n_people // 5)]:
        if len(events) >= params.events:
            break
        place = rng.choice(places)
        eid = f"E{len(events) + 1:04d}"
        quote = f"{person} died at {place} [{eid}]."
        events.append(SynthEvent(
            id=eid, desc=f"death of {person}",
            story_time=t, actors=[person], place=place, event_type="death",
            quote=quote,
        ))
        t += 1.0

    # Trim / pad to exact count
    events = events[: params.events]
    while len(events) < params.events:
        who = rng.choice(people)
        place = rng.choice(places)
        verb = rng.choice(_VERBS)
        eid = f"E{len(events) + 1:04d}"
        events.append(SynthEvent(
            id=eid, desc=f"{who} {verb}",
            story_time=t, actors=[who], place=place,
            quote=_sentence(who, verb, place, tag=eid),
        ))
        t += 1.0

    # Parallel groups: same story_time band, told sequentially later
    n_parallel = max(0, int(round(params.events * params.parallel_rate)))
    cand = [e for e in events if e.event_type == "other"]
    rng.shuffle(cand)
    gid = 0
    i = 0
    while i + 1 < len(cand) and gid * 2 < n_parallel:
        a, b = cand[i], cand[i + 1]
        g = f"P{gid}"
        a.parallel_group = g
        b.parallel_group = g
        mid = (a.story_time + b.story_time) / 2.0
        a.story_time = mid
        b.story_time = mid
        a.quote = (
            f"Meanwhile, in another part of {a.place}, {a.actors[0]} "
            f"{rng.choice(_VERBS)} [{a.id}]."
        )
        b.quote = (
            f"At the same hour, {b.actors[0]} {rng.choice(_VERBS)} at {b.place} [{b.id}]."
        )
        a.desc = a.quote.rstrip(".")
        b.desc = b.quote.rstrip(".")
        gid += 1
        i += 2

    return events


def _assign_frames(events: list[SynthEvent], params: SynthParams) -> None:
    rng = random.Random(params.seed + 17)
    # Parallel pairs keep frame_kind=main so they are emitted once, together.
    chron = sorted(
        (e for e in events if not e.parallel_group),
        key=lambda e: (e.story_time, e.id),
    )
    n = len(chron)
    if n == 0:
        return
    n_fb = max(0, int(round(n * params.flashback_rate)))
    n_prop = max(0, int(round(n * params.prophecy_rate)))
    n_hyp = max(0, int(round(n * params.hypothetical_rate)))
    n_tale = max(0, int(round(n * params.separate_tale_rate)))

    # Flashbacks: pick early story events to narrate late (recollection)
    early = chron[: max(1, n // 2)]
    late_hosts = chron[n // 2 :]
    fb_targets = early[:n_fb] if len(early) >= n_fb else early
    for e in fb_targets:
        host = rng.choice(late_hosts) if late_hosts else chron[-1]
        e.frame_kind = "recollection"
        e.narrator = host.actors[0] if host.actors else "a traveler"
        depth = rng.randint(1, max(1, params.nesting))
        if depth > 1 and host.actors:
            e.narrator = " > ".join(
                [e.narrator] + [rng.choice(host.actors or e.actors) for _ in range(depth - 1)]
            )

    prop_targets = [e for e in chron[- max(1, n // 3) :] if e.frame_kind == "main"][:n_prop]
    for e in prop_targets:
        e.frame_kind = "prediction"
        prophet = rng.choice(chron[: max(1, n // 3)]).actors[0]
        e.narrator = prophet
        e.fulfilled_of = e.id

    hyp = [e for e in chron if e.frame_kind == "main"][:n_hyp]
    for e in hyp:
        e.frame_kind = "hypothetical"
        e.narrator = e.actors[0] if e.actors else "someone"

    tales = [e for e in chron if e.frame_kind == "main"][:n_tale]
    for e in tales:
        e.frame_kind = "separate_tale"
        e.narrator = rng.choice(chron).actors[0] if chron and chron[0].actors else "a stranger"


def render_document(events: list[SynthEvent], params: SynthParams) -> str:
    """Render discourse-ordered text; set quote offsets on each event."""
    rng = random.Random(params.seed + 99)
    chron = sorted(events, key=lambda e: (e.story_time, e.id))
    # Discourse plan: mainline chronological, but insert flashbacks when we hit
    # a "host" after the flashback's story time; prophecies told early.
    mainline = [e for e in chron if e.frame_kind == "main"]
    flashbacks = [e for e in chron if e.frame_kind == "recollection"]
    prophecies = [e for e in chron if e.frame_kind == "prediction"]
    hyps = [e for e in chron if e.frame_kind == "hypothetical"]
    tales = [e for e in chron if e.frame_kind == "separate_tale"]

    # Parallel pairs: keep together sequentially in discourse
    parallel_done: set[str] = set()

    parts: list[str] = []
    discourse_i = 0

    def emit(heading: str | None, body_paras: list[str]) -> None:
        if heading:
            parts.append(heading)
            parts.append("")
        for p in body_paras:
            parts.append(p)
            parts.append("")

    # Chapter 1: prophecies (early discourse, future story) — paraphrase only,
    # so the fulfilment quote in chapter 5 is the unique locate_quote hit.
    if prophecies:
        paras = []
        for e in prophecies:
            who = e.actors[0] if e.actors else "someone"
            paras.append(
                f"{e.narrator} foresaw that {who} would one day act at {e.place}."
            )
        emit("CHAPTER 1 THE FORETELLINGS", paras)

    # Chapter 2: main chronicle with flashbacks nested
    chapter_paras: list[str] = []
    fb_queue = list(flashbacks)
    for e in mainline:
        if e.id in parallel_done:
            continue
        # Possibly insert a flashback before this main event
        while fb_queue and rng.random() < 0.45:
            fb = fb_queue.pop(0)
            depth = min(params.nesting, max(1, fb.narrator.count(">") + 1))
            narrators = [x.strip() for x in fb.narrator.split(">")] or [fb.narrator or "A traveler"]
            if depth > 1 and len(narrators) > 1:
                block = (
                    f"{narrators[0]} began a recollection. "
                    f"Within it, {narrators[1]} spoke of still older times. "
                    f"{fb.quote}"
                )
            else:
                block = f"{narrators[0]} paused and recalled earlier days. {fb.quote}"
            chapter_paras.append(block)
            fb.discourse_order = discourse_i
            discourse_i += 1

        # Parallel: emit both together (each quote once)
        if e.parallel_group:
            group = [x for x in events if x.parallel_group == e.parallel_group]
            group = sorted(group, key=lambda x: x.id)
            for g in group:
                if g.discourse_order >= 0:
                    continue
                if g is group[0]:
                    chapter_paras.append(g.quote)
                else:
                    conn = rng.choice(_CONNECTIVES_AFTER)
                    chapter_paras.append(f"{conn} {g.quote}")
                g.discourse_order = discourse_i
                discourse_i += 1
                parallel_done.add(g.id)
            continue

        conn = rng.choice(_CONNECTIVES_AFTER + ("",)) if discourse_i else ""
        if conn:
            chapter_paras.append(f"{conn} {e.quote}")
        else:
            chapter_paras.append(e.quote)
        e.discourse_order = discourse_i
        discourse_i += 1

    # Drain remaining flashbacks
    for fb in fb_queue:
        nar = (fb.narrator.split(">")[0].strip() if fb.narrator else "A traveler")
        chapter_paras.append(f"{nar} looked back. {fb.quote}")
        fb.discourse_order = discourse_i
        discourse_i += 1

    emit("CHAPTER 2 THE CHRONICLE", chapter_paras)

    # Chapter 3: separate tales
    if tales:
        paras = []
        for e in tales:
            nar = e.narrator or "A stranger"
            paras.append(f"{nar} told a tale from another land. {e.quote}")
            e.discourse_order = discourse_i
            discourse_i += 1
        emit("CHAPTER 3 OTHER TALES", paras)

    # Chapter 4: hypotheticals
    if hyps:
        paras = []
        for e in hyps:
            paras.append(f"Suppose things had gone otherwise. {e.quote}")
            e.discourse_order = discourse_i
            discourse_i += 1
        emit("CHAPTER 4 SUPPOSINGS", paras)

    # Fulfilment: unique occurrence of prophecy quotes
    if prophecies:
        paras = []
        for e in prophecies:
            paras.append(f"In time the foretelling came to pass. {e.quote}")
            e.discourse_order = discourse_i
            discourse_i += 1
        emit("CHAPTER 5 WHAT CAME TO PASS", paras)

    # Ensure every event got a discourse slot (and exactly one quote emission)
    for e in events:
        if e.discourse_order < 0:
            parts.append(e.quote)
            parts.append("")
            e.discourse_order = discourse_i
            discourse_i += 1

    text = "\n".join(parts).strip() + "\n"

    # Locate each event's quote in the rendered text (must be unique)
    for e in events:
        loc = locate_quote(text, e.quote)
        if loc is None:
            raise RuntimeError(f"synth quote not found/unique for {e.id}: {e.quote!r}")
        e.char_start, e.char_end = loc

    return text


def build_gold(events: list[SynthEvent], *, max_pairs: int | None = None) -> list[GoldQuestion]:
    """Order questions with strata from story vs discourse positions."""
    gold: list[GoldQuestion] = []
    chron = sorted(events, key=lambda e: (e.story_time, e.id))
    # Prefer non-hypothetical for reliable order gold
    usable = [e for e in chron if e.frame_kind != "hypothetical" and e.char_start is not None]

    def stratum(a: SynthEvent, b: SynthEvent) -> tuple[str, str]:
        if a.parallel_group and a.parallel_group == b.parallel_group:
            return "unordered", "cannot_determine"
        story_before = (a.story_time, a.id) < (b.story_time, b.id)
        disc_before = a.discourse_order < b.discourse_order
        if a.story_time == b.story_time and a.parallel_group:
            return "unordered", "cannot_determine"
        label = "before" if story_before else "after"
        if story_before == disc_before:
            return "aligned", label
        return "inverted", label

    pairs: list[tuple[SynthEvent, SynthEvent]] = []
    # Parallel unordered pairs
    seen_g: set[str] = set()
    for e in usable:
        if e.parallel_group and e.parallel_group not in seen_g:
            group = [x for x in usable if x.parallel_group == e.parallel_group]
            if len(group) >= 2:
                pairs.append((group[0], group[1]))
                seen_g.add(e.parallel_group)

    # Systematic neighbors + some long-range
    for i in range(len(usable) - 1):
        pairs.append((usable[i], usable[i + 1]))
    step = max(2, len(usable) // 10)
    for i in range(0, len(usable) - step, step):
        pairs.append((usable[i], usable[i + step]))

    # Dedup
    seen: set[tuple[str, str]] = set()
    uniq: list[tuple[SynthEvent, SynthEvent]] = []
    for a, b in pairs:
        key = tuple(sorted((a.id, b.id)))
        if key in seen or a.id == b.id:
            continue
        seen.add(key)
        uniq.append((a, b))

    if max_pairs is not None:
        uniq = uniq[:max_pairs]

    for i, (a, b) in enumerate(uniq, start=1):
        st, lab = stratum(a, b)
        qid = f"sq{i:04d}"
        pa, pb = question_phrase(a), question_phrase(b)
        gold.append(GoldQuestion(
            id=qid, qtype="order", stratum=st,  # type: ignore[arg-type]
            question=ORDER_TEMPLATE.format(a=pa, b=pb),
            gold_label=lab,  # type: ignore[arg-type]
            # events[].id keeps the internal E-key; desc is the clean phrase.
            events={"A": GoldEvent(id=a.id, desc=pa),
                    "B": GoldEvent(id=b.id, desc=pb)},
            evidence=[
                # Evidence quotes stay verbatim (with E-tags) so locate_quote works.
                GoldEvidence(group="A", quote=a.quote,
                             char_start=a.char_start, char_end=a.char_end),
                GoldEvidence(group="B", quote=b.quote,
                             char_start=b.char_start, char_end=b.char_end),
            ],
            verified_by="synth",
        ))
    return gold


def write_pdf(text: str, path: Path) -> None:
    """Write a multi-page PDF with larger-font ALL-CAPS chapter headings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    width, height = 595, 842  # A4
    margin = 54
    body_size = 11
    head_size = 16
    y = margin
    page = doc.new_page(width=width, height=height)

    def new_page():
        nonlocal page, y
        page = doc.new_page(width=width, height=height)
        y = margin

    for raw in text.split("\n"):
        line = raw.rstrip()
        is_head = bool(line) and line == line.upper() and line.startswith("CHAPTER")
        size = head_size if is_head else body_size
        font = "helv"
        # wrap
        if not line:
            y += size * 0.8
            if y > height - margin:
                new_page()
            continue
        words = line.split()
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            tw = fitz.get_text_length(trial, fontname=font, fontsize=size)
            if tw > width - 2 * margin and cur:
                if y + size * 1.4 > height - margin:
                    new_page()
                page.insert_text((margin, y), cur, fontname=font, fontsize=size)
                y += size * 1.35
                cur = w
            else:
                cur = trial
        if cur:
            if y + size * 1.4 > height - margin:
                new_page()
            page.insert_text((margin, y), cur, fontname=font, fontsize=size)
            y += size * (1.6 if is_head else 1.35)
    doc.save(path)
    doc.close()


def generate(params: SynthParams) -> SynthResult:
    events = build_timeline(params)
    _assign_frames(events, params)
    text = render_document(events, params)
    gold = build_gold(events)
    strata: dict[str, int] = {}
    for q in gold:
        strata[q.stratum or ""] = strata.get(q.stratum or "", 0) + 1
    frames: dict[str, int] = {}
    for e in events:
        frames[e.frame_kind] = frames.get(e.frame_kind, 0) + 1
    return SynthResult(params=params, events=events, text=text, gold=gold,
                       strata_counts=strata, frame_counts=frames)


def write_outputs(result: SynthResult, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "synth.pdf"
    gold_path = out_dir / "gold.jsonl"
    params_path = out_dir / "params.json"
    text_path = out_dir / "text.txt"

    write_pdf(result.text, pdf_path)
    text_path.write_text(result.text, encoding="utf-8")
    gold_path.write_text(
        "".join(q.model_dump_json() + "\n" for q in result.gold), encoding="utf-8"
    )
    meta = {
        **asdict(result.params),
        "n_events": len(result.events),
        "n_gold": len(result.gold),
        "strata_counts": result.strata_counts,
        "frame_counts": result.frame_counts,
        "pdf": str(pdf_path),
        "gold": str(gold_path),
    }
    params_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def verify_gold_quotes(text: str, gold: list[GoldQuestion]) -> list[str]:
    """Return list of question ids whose quotes fail locate_quote."""
    bad = []
    for q in gold:
        for ev in q.evidence:
            if locate_quote(text, ev.quote) is None:
                bad.append(q.id)
                break
    return bad


def expected_frame_counts(params: SynthParams) -> dict[str, int]:
    """Expected counts from rates. Frame rates apply to the non-parallel pool
    (parallel events stay main so their quotes remain unique)."""
    n = params.events
    n_parallel = int(round(n * params.parallel_rate))
    n_parallel -= n_parallel % 2  # pairs only
    pool = max(0, n - n_parallel)
    return {
        "recollection": int(round(pool * params.flashback_rate)),
        "prediction": int(round(pool * params.prophecy_rate)),
        "hypothetical": int(round(pool * params.hypothetical_rate)),
        "separate_tale": int(round(pool * params.separate_tale_rate)),
        "parallel_events": n_parallel,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m app.synth")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--events", type=int, default=settings.synth_default_events)
    ap.add_argument("--flashback-rate", type=float,
                    default=settings.synth_default_flashback_rate)
    ap.add_argument("--nesting", type=int, default=settings.synth_default_nesting)
    ap.add_argument("--prophecy-rate", type=float,
                    default=settings.synth_default_prophecy_rate)
    ap.add_argument("--hypothetical-rate", type=float,
                    default=settings.synth_default_hypothetical_rate)
    ap.add_argument("--separate-tale-rate", type=float,
                    default=settings.synth_default_separate_tale_rate)
    ap.add_argument("--parallel-rate", type=float,
                    default=settings.synth_default_parallel_rate)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    params = SynthParams(
        seed=args.seed, events=args.events, flashback_rate=args.flashback_rate,
        nesting=args.nesting, prophecy_rate=args.prophecy_rate,
        hypothetical_rate=args.hypothetical_rate,
        separate_tale_rate=args.separate_tale_rate,
        parallel_rate=args.parallel_rate,
    )
    result = generate(params)
    bad = verify_gold_quotes(result.text, result.gold)
    if bad:
        raise SystemExit(f"gold quotes failed locate for {len(bad)} questions: {bad[:5]}")
    meta = write_outputs(result, args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
