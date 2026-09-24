"""System prompts for v2 LLM calls. Generic: no book, character or place names.
Any edit here changes results — bump PROMPT_VERSION and record it with runs."""

from ..config import settings

PROMPT_VERSION = "v2.2"

# ---------------------------------------------------------------------------
# Window extraction — exhaustive (default) vs salient
# ---------------------------------------------------------------------------
_EXTRACT_SHARED_HEAD = """You extract story events and their time relations from one window of a book.

INPUT
- The window is split into paragraphs marked [p17] etc. Paragraphs marked
  "(context, already processed)" are for reading only: do NOT extract from them.
- FRAME STACK lists the embedded narrations that are open when this window starts,
  outermost first. The first entry is always the main narration.

1. FRAME OPERATIONS
Open a frame when a character begins NARRATING a sequence of events (not a single line
of dialogue). Types:
- recollection: earlier events of the same story world (a character recounts the past)
- separate_tale: a story from another world or time, told as an illustration or history
- prediction: prophecy, vow, curse, oath, plan, or order about the future
- hypothetical: dream, counterfactual, conditional, "suppose", simile, example
Also note discourse shifts such as "meanwhile", "in another part of …", "at the same
hour": keep events as mode=occurs in the current frame (do not drop them).
Close a frame where that narration ends (the text returns to the outer narrator or
the same speaker stops narrating). Give the paragraph id and 3-12 verbatim words at
the exact point. If a frame stays open past this window, do not close it.
"""

_EXTRACT_MENTIONS_EXHAUSTIVE = """
2. EVENT MENTIONS (from owned paragraphs only) — EXHAUSTIVE MODE
Extract EVERY narrated event: an action, happening or change of state at a point in
story time — including short one-sentence chronicle lines, births, arrivals, speech-acts
that change state (vows, seals, summons), and events inside recollections, predictions,
counterfactuals ("Suppose things had gone otherwise"), separate tales, and
meanwhile / at-the-same-hour stretches. Prefer recall: if unsure whether a sentence
narrates an event, extract it. Do NOT extract descriptions of places or objects alone,
general truths, teachings, lists of names, or speech that narrates nothing. When many
events appear in sequence, emit one mention per event (do not summarise a paragraph of
several actions into a single mention). Never skip an event because it seems minor.
"""

_EXTRACT_MENTIONS_SALIENT = """
2. EVENT MENTIONS (from owned paragraphs only) — SALIENT MODE
Extract plot-significant events: actions and state-changes that matter to the story arc.
You may omit pure atmosphere, repeated greetings, and decorative detail. Still extract
events inside recollections/predictions when they are plot-relevant. Prefer precision
over dumping every clause. When several actions share one beat, one mention is enough.
"""

_EXTRACT_SHARED_TAIL = """
- mode, relative to the frame the event is narrated in:
  occurs       narrated as happening at this point of the narration
  recounted    narrated as having happened EARLIER than the surrounding narration
               (e.g. "had done", "long ago", "before this")
  predicted    foretold, vowed, planned, ordered, feared, not yet happened
  hypothetical imagined, conditional, counterfactual, dreamed
  habitual     repeated or customary ("used to", "every day")
- quote: 5-25 words copied exactly from the text.
- participants: names or epithets exactly as written. role agent/patient/present for
  people who take part; mentioned for people only talked about.
- event_type birth/death/marriage with subject when the event is one of those.

3. RELATIONS between events in this window
Only when the text supports them. cue:
- explicit_connective: "after", "then", "before", "when", "while", "meanwhile", "at the
  same time", "having done", "on hearing this"
- time_expression: dated or counted time ("on the tenth day", "twelve years later")
- causal: one event is stated to cause or enable the other (cause is before effect)
- narrative_order: only for clear sequential narration within one frame
Always relate each "recounted" event to the event it is recounted during.
Use overlap/simultaneous for events the text says happen together or at the same time,
even if they are described one after another.

4. ALIASES AND KINSHIP
aliases: only where the text itself equates two names/epithets for one individual.
kinship: only stated parent-child relations.

Copy quotes exactly. Never use knowledge of the book from outside the window."""


def extract_system() -> str:
    """Active extract system prompt for the configured extraction_mode."""
    body = (_EXTRACT_MENTIONS_EXHAUSTIVE
            if (settings.extraction_mode or "exhaustive").lower() != "salient"
            else _EXTRACT_MENTIONS_SALIENT)
    return _EXTRACT_SHARED_HEAD + body + _EXTRACT_SHARED_TAIL


# Prefer extract_system() at call time; alias for older imports.
EXTRACT_SYSTEM = extract_system()

ENTITY_SYSTEM = """You group surface forms (names, epithets, patronymics) that refer to the same
individual in one book. You get each form with short contexts and the alias statements the
text itself makes. Put two forms together only if the contexts show they are one person.
Two people can share a name; keep them apart when contexts show different people.
Every form must appear in exactly one cluster."""

COREF_SYSTEM = """Decide whether two passages from the same book refer to the SAME event
occurrence in the story world. A prophecy or vow and its fulfilment are the same event only
if the passage describes the fulfilment itself. A retelling of an earlier event by a
character IS the same event. Two similar events (two battles, two journeys) are not the
same unless participants, place and circumstances match."""

QUESTION_SYSTEM = """Parse a question about the order of events in a book. Extract the events
it asks about as short descriptions, in the order they appear in the question.

qtype rules:
- order: the question asks whether event A happened before or after event B
  (e.g. "Did A happen before or after B?", "which came first, A or B?").
  Set event_a = A, event_b = B, direction = none.
- before_after_x: the question asks what happened before OR after a single event X
  (e.g. "What happened before X?", "what came after X?"). Set event_a = X,
  event_b empty, direction = before|after.
- during: asks what was happening while X occurred.
- factual: asks a non-ordering fact.

Strip discourse prefixes ("Meanwhile,", "At the same hour,") and bracket tags like
[E0013] from event descriptions — keep clean actor/action/place phrasing only."""

GROUND_SYSTEM = """Pick the candidate event that the description refers to. Candidates come
from the book with their text. Answer -1 if none of them is the described event. Mark
ambiguous if two or more fit equally well."""

VERBALISE_SYSTEM = """You explain an ordering result that has ALREADY been computed from the
book's text. Do not change or second-guess the relation you are given. Explain it using the
evidence chain and the quoted passages, citing pages as (p. N). If the relation is
cannot_determine, say plainly that the text does not settle the order and, if a detail
is given (during, overlap, simultaneous), say what the text does establish."""
