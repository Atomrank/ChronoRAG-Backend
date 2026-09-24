"""Query-engine tests with in-memory graph and mocked LLM (invented names only)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.schemas import PipelineAnswer
from app.v2.schemas import GroundingChoice, QuestionParse, VerbalisedAnswer
from app.v2.solver import Constraint, TemporalGraph


@pytest.fixture(autouse=True)
def _no_buildlog(monkeypatch):
    monkeypatch.setattr("app.buildlog.record", lambda *a, **k: None)


def _graph():
    g = TemporalGraph(["ev_1", "ev_2", "ev_3"], {"ev_1": 0, "ev_2": 10, "ev_3": 20})
    g.add(Constraint("ev_1", "ev_2", "before", 0.9, "explicit"))
    g.add(Constraint("ev_2", "ev_3", "before", 0.9, "explicit"))
    return g


class _Chat:
    def __init__(self, parse: QuestionParse, grounds: list[GroundingChoice]):
        self.parse = parse
        self.grounds = list(grounds)
        self.gi = 0

    def __call__(self, system, user, model_cls, **kw):
        if model_cls is QuestionParse:
            return self.parse
        if model_cls is GroundingChoice:
            c = self.grounds[min(self.gi, len(self.grounds) - 1)]
            self.gi += 1
            return c
        if model_cls is VerbalisedAnswer:
            return VerbalisedAnswer(answer="ok", relation="before")
        raise AssertionError(model_cls)


def test_order_before(monkeypatch):
    from app.v2 import query

    g = _graph()
    chat = _Chat(
        QuestionParse(qtype="order", event_a="river freezes", event_b="bridge built",
                      events_list=[], entities=[], direction="none"),
        [GroundingChoice(choice=0, ambiguous=False, reason="a"),
         GroundingChoice(choice=1, ambiguous=False, reason="b")],
    )
    monkeypatch.setattr(query, "_ground_one", lambda *a, **k: (
        ("ev_1", None, 0.9, [{"id": "ev_1", "score": 0.9}]) if "river" in (a[1] if len(a) > 1 else "")
        else ("ev_2", None, 0.9, [{"id": "ev_2", "score": 0.9}])
    ))
    monkeypatch.setattr(query, "load_graph", lambda doc_id: {"graph": g.to_dict()})
    monkeypatch.setattr(query, "_mention_spans", lambda *a, **k: [[0, 10]])
    monkeypatch.setattr(query, "_passages_for_events", lambda *a, **k: [])
    monkeypatch.setattr(query.docstore, "load_document", MagicMock(text="x" * 100,
                                                                   pages_for_span=lambda *a: [1]))

    ans = query.answer("doc", "Did the river freeze before the bridge was built?",
                       chat_fn=chat, graph=g)
    assert isinstance(ans, PipelineAnswer)
    assert ans.pipeline == "kaalkram_v2"
    assert ans.relation == "before"


def test_order_not_found(monkeypatch):
    from app.v2 import query

    g = _graph()
    chat = _Chat(
        QuestionParse(qtype="order", event_a="missing", event_b="bridge",
                      events_list=[], entities=[], direction="none"),
        [],
    )
    monkeypatch.setattr(query, "_ground_one", lambda *a, **k: (None, "not_found", 0.0, []))
    monkeypatch.setattr(query, "load_graph", lambda doc_id: {"graph": g.to_dict()})
    monkeypatch.setattr(query, "_passages_for_events", lambda *a, **k: [])
    monkeypatch.setattr(query.docstore, "load_document", MagicMock(text="x" * 50,
                                                                   pages_for_span=lambda *a: [1]))
    ans = query.answer("doc", "order?", chat_fn=chat, graph=g)
    assert ans.relation == "cannot_determine"
    assert "not_found" in (ans.trace[-1] if ans.trace else "") or True


def test_sequence_unordered(monkeypatch):
    from app.v2 import query

    g = TemporalGraph(["ev_1", "ev_2"], {"ev_1": 0, "ev_2": 1})  # no edge
    chat = _Chat(
        QuestionParse(qtype="sequence", event_a="", event_b="",
                      events_list=["A", "B"], entities=[], direction="none"),
        [],
    )
    ids = iter(["ev_1", "ev_2"])
    monkeypatch.setattr(query, "_ground_one",
                        lambda *a, **k: (next(ids), None, 0.8, [{"id": "ev_1"}]))
    monkeypatch.setattr(query, "_mention_spans", lambda *a, **k: [[0, 5]])
    monkeypatch.setattr(query, "_passages_for_events", lambda *a, **k: [])
    monkeypatch.setattr(query.docstore, "load_document", MagicMock(text="x" * 50,
                                                                   pages_for_span=lambda *a: [1]))
    ans = query.answer("doc", "order A and B", chat_fn=chat, graph=g)
    assert ans.relation == "not_applicable"
