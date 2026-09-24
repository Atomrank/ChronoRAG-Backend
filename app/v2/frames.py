"""
Narrator-frame tracking (the "deictic reference stack").

The LLM only proposes operations per window ("an embedded narration opens
here", "the frame on top closes here"). This module applies them to a stack in
code, so the frame structure is deterministic and auditable, and assigns every
character offset to exactly one innermost frame.

Frame types (generic narratology, no book knowledge):
  main           the top-level narration of a unit
  recollection   a character narrates earlier events of the same story world
  separate_tale  a character tells a story from another story world/timeline
  prediction     prophecy, vow, curse, plan, command about the future
  hypothetical   dream, counterfactual, conditional, simile, illustrative example
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

FrameType = Literal["main", "recollection", "separate_tale", "prediction", "hypothetical"]
PAST_FRAMES = ("recollection", "separate_tale")


@dataclass
class Frame:
    id: str
    unit_id: str
    type: str
    parent: str | None
    narrator: str
    listener: str
    open_at: int
    close_at: int | None = None
    depth: int = 0
    summary: str = ""
    auto_closed: bool = False

    def contains(self, offset: int) -> bool:
        return self.open_at <= offset < (self.close_at if self.close_at is not None else 1 << 62)


@dataclass
class FrameOp:
    op: Literal["open", "close"]
    at: int                                   # resolved character offset
    frame_type: str = "recollection"
    narrator: str = ""
    listener: str = ""
    frame_ref: str | None = None              # for close: id from the stack shown to the LLM
    summary: str = ""


@dataclass
class FrameTracker:
    unit_id: str
    unit_start: int
    unit_end: int
    frames: list[Frame] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.frames:
            root = Frame(id=f"{self.unit_id}/F0", unit_id=self.unit_id, type="main",
                         parent=None, narrator="", listener="", open_at=self.unit_start)
            self.frames.append(root)
            self.stack.append(root.id)

    def _get(self, fid: str) -> Frame:
        return next(f for f in self.frames if f.id == fid)

    def state(self) -> list[dict]:
        """The open stack, outermost first — shown to the LLM for the next window."""
        return [{"id": f.id, "type": f.type, "narrator": f.narrator, "listener": f.listener,
                 "summary": f.summary} for f in (self._get(i) for i in self.stack)]

    def apply(self, ops: list[FrameOp]) -> None:
        for op in sorted(ops, key=lambda o: (o.at, 0 if o.op == "close" else 1)):
            if not (self.unit_start <= op.at <= self.unit_end):
                self.warnings.append({"warning": "op_outside_unit", "op": asdict(op)})
                continue
            if op.op == "open":
                if op.frame_type not in ("recollection", "separate_tale", "prediction",
                                         "hypothetical"):
                    self.warnings.append({"warning": "bad_frame_type", "op": asdict(op)})
                    continue
                parent = self._get(self.stack[-1])
                if op.at < parent.open_at:
                    self.warnings.append({"warning": "open_before_parent", "op": asdict(op)})
                    continue
                f = Frame(id=f"{self.unit_id}/F{len(self.frames)}", unit_id=self.unit_id,
                          type=op.frame_type, parent=parent.id, narrator=op.narrator,
                          listener=op.listener, open_at=op.at, depth=parent.depth + 1,
                          summary=op.summary)
                self.frames.append(f)
                self.stack.append(f.id)
            else:
                target = op.frame_ref if op.frame_ref in self.stack else None
                if len(self.stack) == 1:
                    self.warnings.append({"warning": "close_with_empty_stack", "op": asdict(op)})
                    continue
                if target is None:
                    target = self.stack[-1]
                    if op.frame_ref:
                        self.warnings.append({"warning": "unknown_frame_ref_closed_top",
                                              "op": asdict(op)})
                if target == self.stack[0]:
                    self.warnings.append({"warning": "tried_to_close_main", "op": asdict(op)})
                    continue
                # closing an outer frame closes everything opened inside it
                while self.stack:
                    fid = self.stack.pop()
                    fr = self._get(fid)
                    fr.close_at = max(op.at, fr.open_at)
                    if fid != target:
                        fr.auto_closed = True
                        self.warnings.append({"warning": "closed_inner_frame_implicitly",
                                              "frame": fid})
                    if fid == target:
                        break

    def finish(self) -> list[Frame]:
        """Close anything still open at the end of the unit."""
        while len(self.stack) > 1:
            fr = self._get(self.stack.pop())
            fr.close_at = self.unit_end
            fr.auto_closed = True
            self.warnings.append({"warning": "auto_closed_at_unit_end", "frame": fr.id})
        root = self._get(self.stack[0])
        root.close_at = self.unit_end
        return self.frames

    def frame_at(self, offset: int) -> Frame:
        """Innermost frame containing the offset."""
        best = None
        for f in self.frames:
            if f.contains(offset) and (best is None or f.depth > best.depth):
                best = f
        return best or self.frames[0]
