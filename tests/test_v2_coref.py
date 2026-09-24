"""Tests for app.v2.coref — mocked LLM/embeddings, invented names only."""
from __future__ import annotations

import numpy as np
import pytest

from app.v2.coref import resolve
from app.v2.extract import ExtractedMention
from app.v2.schemas import CorefDecision


@pytest.fixture(autouse=True)
def _no_buildlog(monkeypatch):
    monkeypatch.setattr("app.v2.coref.buildlog.record", lambda *a, **k: None)


def _m(mid, start, desc, mode, entity="ent_1", window="win_1", *,
       is_telling=False, **kw):
    return ExtractedMention(
        id=mid, local_id=mid, window_id=window, frame_id="U0/F0",
        start=start, end=start + max(10, len(desc)),
        para_id="p1", quote=desc[:20], description=desc, mode=mode,
        event_type="other",
        participants=[{"surface": "Kael", "role": "agent", "entity_id": entity}],
        is_telling=is_telling, **kw,
    )


def _unit(dim=8, *axes):
    v = np.zeros(dim)
    for a in axes:
        v[a % dim] = 1.0
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else v.tolist()


def test_retelling_merged():
    text = ("Kael broke the blue seal at dawn. " * 3
            + "Later someone said Kael had broken the blue seal. ")
    mentions = [
        _m("m1", 0, "Kael breaks the blue seal", "occurs"),
        _m("m2", 120, "Kael breaks the blue seal", "recounted", window="win_2"),
    ]
    # Identical descriptions -> cosine 1.
    def embed_fn(texts):
        return [_unit(8, 0) for _ in texts]

    def chat_fn(system, user, model_cls):
        return CorefDecision(same_event=True, reason="retelling of the seal breaking")

    result = resolve(mentions, text, doc_id="c_retell",
                     embed_fn=embed_fn, chat_fn=chat_fn)
    assert len(result.events) == 1
    assert set(result.events[0].mention_ids) == {"m1", "m2"}
    assert result.events[0].description == "Kael breaks the blue seal"
    assert {m.event_id for m in result.mentions} == {"ev_1"}


def test_prophecy_and_fulfilment_merged():
    text = ("A seer vowed that Kael would take the white gate. "
            "Years later Kael took the white gate.")
    mentions = [
        _m("m1", 0, "Kael takes the white gate", "predicted"),
        _m("m2", 55, "Kael takes the white gate", "occurs", window="win_2"),
    ]

    def embed_fn(texts):
        return [_unit(8, 1) for _ in texts]

    def chat_fn(system, user, model_cls):
        return CorefDecision(same_event=True, reason="prophecy fulfilled")

    result = resolve(mentions, text, doc_id="c_prophecy",
                     embed_fn=embed_fn, chat_fn=chat_fn)
    assert len(result.events) == 1
    # Description prefers earliest occurs/recounted over predicted.
    assert result.events[0].description == "Kael takes the white gate"
    assert result.mentions[0].event_id == result.mentions[1].event_id


def test_two_similar_battles_kept_apart():
    text = ("Kael fought at the north ford. " * 5
            + "Much later Kael fought at the south ford. ")
    mentions = [
        _m("m1", 0, "Kael fights at the north ford", "occurs"),
        _m("m2", 200, "Kael fights at the south ford", "occurs", window="win_2"),
    ]

    def embed_fn(texts):
        # High similarity but distinct vectors still above threshold if identical —
        # use the same vector so candidates form; LLM refuses merge.
        return [_unit(8, 2) for _ in texts]

    def chat_fn(system, user, model_cls):
        return CorefDecision(same_event=False, reason="two different fords")

    result = resolve(mentions, text, doc_id="c_battles",
                     embed_fn=embed_fn, chat_fn=chat_fn)
    assert len(result.events) == 2
    assert result.mentions[0].event_id != result.mentions[1].event_id


def test_hypothetical_hard_block():
    mentions = [
        _m("m1", 0, "Kael opens the vault", "occurs"),
        _m("m2", 50, "Kael opens the vault", "hypothetical", window="win_2"),
    ]
    calls = {"n": 0}

    def embed_fn(texts):
        return [_unit(8, 3) for _ in texts]

    def chat_fn(system, user, model_cls):
        calls["n"] += 1
        return CorefDecision(same_event=True, reason="should not be asked")

    result = resolve(mentions, "x" * 200, doc_id="c_hyp",
                     embed_fn=embed_fn, chat_fn=chat_fn)
    assert calls["n"] == 0
    assert len(result.events) == 2
