"""System prompts for v2 LLM calls. Generic: no book, character or place names.
Any edit here changes results — bump PROMPT_VERSION and record it with runs."""

PROMPT_VERSION = "v2.0"

EXTRACT_SYSTEM = """You extract story events and their time relations from one window of a book.

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
Close a frame where that narration ends (the text returns to the outer narrator or
the same speaker stops narrating). Give the paragraph id and 3-12 verbatim words at
the exact point. If a frame stays open past this window, do not close it.

2. EVENT MENTIONS (from owned paragraphs only)
Extract every narrated event: an action, happening or change of state at a point in
story time. Do NOT extract descriptions of places or objects, general truths,
teachings, lists of names, or speech that narrates nothing.
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
it asks about as short descriptions, in the order they appear in the question. For
"did A happen before or after B", event_a = A and event_b = B."""

GROUND_SYSTEM = """Pick the candidate event that the description refers to. Candidates come
from the book with their text. Answer -1 if none of them is the described event. Mark
ambiguous if two or more fit equally well."""

VERBALISE_SYSTEM = """You explain an ordering result that has ALREADY been computed from the
book's text. Do not change or second-guess the relation you are given. Explain it using the
evidence chain and the quoted passages, citing pages as (p. N). If the relation is
cannot_determine, say plainly that the text does not settle the order and, if a detail
is given (during, overlap, simultaneous), say what the text does establish."""
