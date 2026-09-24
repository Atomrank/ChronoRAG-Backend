"""Strict-JSON contracts for every v2 LLM call (used with llm.chat_structured)."""
from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["occurs", "recounted", "predicted", "hypothetical", "habitual"]
Role = Literal["agent", "patient", "present", "mentioned"]
RelKind = Literal["before", "after", "during", "contains", "simultaneous", "overlap"]
Cue = Literal["explicit_connective", "time_expression", "causal", "narrative_order"]


# ---------------- window extraction ----------------
class FrameOpOut(BaseModel):
    op: Literal["open", "close"]
    para_id: str = Field(description="Paragraph id like p17 where the frame opens/closes")
    quote: str = Field(description="3-12 verbatim words at the exact point it opens/closes")
    frame_type: Literal["recollection", "separate_tale", "prediction", "hypothetical", "none"] = Field(
        description="For open: the kind of embedded narration. For close: 'none'")
    narrator: str = Field(description="Who narrates the embedded content (as written); '' for close")
    listener: str = Field(description="Who is addressed (as written); '' if unknown or close")
    frame_ref: str = Field(description="For close: id of the open frame being closed, from the "
                                       "stack you were given or 'NEW:<n>' for one opened in this "
                                       "window (n = its position among your opens, from 1). "
                                       "For open: ''")
    summary: str = Field(description="For open: one line on what the embedded narration is about")


class ParticipantOut(BaseModel):
    name: str = Field(description="Name or epithet exactly as written in the text")
    role: Role


class MentionOut(BaseModel):
    local_id: str = Field(description="m1, m2, ... unique within this window")
    para_id: str
    quote: str = Field(description="5-25 verbatim words that state this event")
    description: str = Field(description="One plain sentence: who does what to whom")
    event_type: Literal["birth", "death", "marriage", "other"]
    subject: str = Field(description="For birth/death/marriage: the person born/who dies/"
                                     "who marries, as written. Otherwise ''")
    participants: list[ParticipantOut]
    location: str
    mode: Mode
    posthumous: bool = Field(description="True only if a person acts after their own death "
                                         "(ghost, summoned dead, afterlife)")
    time_expressions: list[str] = Field(description="Verbatim time phrases attached to this event")


class RelationOut(BaseModel):
    a: str = Field(description="local_id of the first event")
    b: str = Field(description="local_id of the second event")
    relation: RelKind = Field(description="a <relation> b")
    cue: Cue
    quote: str = Field(description="Verbatim words that justify the relation")


class AliasOut(BaseModel):
    name_a: str
    name_b: str
    quote: str = Field(description="Verbatim words where the text equates the two")


class KinshipOut(BaseModel):
    parent: str
    child: str
    quote: str


class WindowExtraction(BaseModel):
    frame_ops: list[FrameOpOut]
    mentions: list[MentionOut]
    relations: list[RelationOut]
    aliases: list[AliasOut]
    kinship: list[KinshipOut]


# ---------------- entity resolution ----------------
class EntityCluster(BaseModel):
    canonical: str = Field(description="The most common full name in the group")
    members: list[str] = Field(description="Surface forms that refer to this one individual")


class EntityResolution(BaseModel):
    clusters: list[EntityCluster]


# ---------------- event coreference ----------------
class CorefDecision(BaseModel):
    same_event: bool = Field(description="True only if both passages refer to the SAME "
                                         "occurrence in the story world")
    reason: str


# ---------------- query time ----------------
class QuestionParse(BaseModel):
    qtype: Literal["order", "sequence", "before_after_x", "state", "factual"]
    event_a: str = Field(description="Description of the first event mentioned ('' if none)")
    event_b: str = Field(description="Description of the second event mentioned ('' if none)")
    events_list: list[str] = Field(description="For sequence questions: the events to order")
    entities: list[str] = Field(description="People/places named in the question, as written")
    direction: Literal["before", "after", "none"] = Field(
        description="For before_after_x: whether the question asks what came before or after X")


class GroundingChoice(BaseModel):
    choice: int = Field(description="Index of the matching candidate, or -1 if none matches")
    ambiguous: bool = Field(description="True if two or more candidates match equally well")
    reason: str


class VerbalisedAnswer(BaseModel):
    answer: str = Field(description="2-5 sentences. State the given relation; cite (p. N)")
    relation: Literal["before", "after", "cannot_determine", "not_applicable"] = Field(
        description="Must equal the relation you were given")
