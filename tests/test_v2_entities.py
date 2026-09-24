"""Tests for app.v2.entities — mocked LLM/embeddings, invented names only."""
from __future__ import annotations

import numpy as np
import pytest

from app.v2.entities import resolve
from app.v2.extract import ExtractedAlias, ExtractedMention
from app.v2.schemas import EntityCluster, EntityResolution


@pytest.fixture(autouse=True)
def _no_buildlog(monkeypatch):
    monkeypatch.setattr("app.v2.entities.buildlog.record", lambda *a, **k: None)


def _mention(mid, start, end, participants, subject="", **kw):
    return ExtractedMention(
        id=mid, local_id=mid, window_id="win_1", frame_id="U0/F0",
        start=start, end=end, para_id="p1", quote=kw.get("quote", "quote"),
        description=kw.get("description", "event"), mode="occurs",
        event_type=kw.get("event_type", "other"), subject=subject,
        participants=[{"surface": n, "role": r} for n, r in participants],
    )


def _vec(*axis_ones: int, dim: int = 8) -> list[float]:
    v = np.zeros(dim)
    for a in axis_ones:
        v[a % dim] = 1.0
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else v.tolist()


def test_three_epithets_one_person():
    """One individual with three epithets, linked by alias + LLM clustering."""
    text = (
        "Orin the Red crossed the ford. "
        "Later the Flame-Hand raised a banner. "
        "At dusk Red-Orin sealed the pact."
    )
    mentions = [
        _mention("m1", 0, 28, [("Orin the Red", "agent")],
                 quote="Orin the Red crossed", description="Orin crosses"),
        _mention("m2", 30, 68, [("the Flame-Hand", "agent")],
                 quote="Flame-Hand raised a banner", description="banner raised"),
        _mention("m3", 70, 104, [("Red-Orin", "agent")],
                 quote="Red-Orin sealed the pact", description="pact sealed"),
    ]
    aliases = [
        ExtractedAlias("Orin the Red", "the Flame-Hand",
                       "Orin the Red, called the Flame-Hand", 0, 40),
        ExtractedAlias("the Flame-Hand", "Red-Orin",
                       "the Flame-Hand, known as Red-Orin", 30, 70),
    ]

    # Embeddings: all three forms near each other on axis 0.
    def embed_fn(texts):
        out = []
        for t in texts:
            if "Flame" in t or "Red-Orin" in t or "Orin the Red" in t:
                out.append(_vec(0))
            else:
                out.append(_vec(7))
        return out

    def chat_fn(system, user, model_cls):
        return EntityResolution(clusters=[
            EntityCluster(canonical="Orin the Red",
                          members=["Orin the Red", "the Flame-Hand", "Red-Orin"]),
        ])

    result = resolve(text, mentions, aliases, [], doc_id="t_epithet",
                     embed_fn=embed_fn, chat_fn=chat_fn)
    assert len(result.entities) == 1
    ent = result.entities[0]
    assert ent.id == "ent_1"
    assert set(ent.surfaces) >= {"Orin the Red", "the Flame-Hand", "Red-Orin"}
    eids = {p["entity_id"] for m in result.mentions for p in m.participants}
    assert eids == {"ent_1"}


def test_name_collision_two_people():
    """Two people share the surface 'Mara'; LLM splits tagged forms."""
    text = (
        "Mara of the North lit the beacon. "
        "Far away Mara of the Docks sold fish."
    )
    mentions = [
        _mention("m1", 0, 34, [("Mara", "agent")],
                 quote="Mara of the North lit", description="north Mara lights"),
        _mention("m2", 36, 76, [("Mara", "agent")],
                 quote="Mara of the Docks sold", description="dock Mara sells"),
    ]

    def embed_fn(texts):
        # Same surface -> candidate-grouped; contexts differ.
        return [_vec(0) for _ in texts]

    def chat_fn(system, user, model_cls):
        assert "Mara#m1" in user and "Mara#m2" in user
        return EntityResolution(clusters=[
            EntityCluster(canonical="Mara of the North", members=["Mara#m1"]),
            EntityCluster(canonical="Mara of the Docks", members=["Mara#m2"]),
        ])

    result = resolve(text, mentions, [], [], doc_id="t_collision",
                     embed_fn=embed_fn, chat_fn=chat_fn)
    assert len(result.entities) == 2
    e1 = result.mentions[0].participants[0]["entity_id"]
    e2 = result.mentions[1].participants[0]["entity_id"]
    assert e1 != e2
    assert "Mara" not in result.surface_to_entity  # ambiguous surface
    assert result.form_to_entity[("Mara", "m1")] == e1
    assert result.form_to_entity[("Mara", "m2")] == e2
