"""Tests for eval split hashing and synthetic benchmark quotes."""
from __future__ import annotations

import json
from collections import Counter

from app import eval_runner as E
from app import synth
from app.config import settings
from app.ingest_v2 import locate_quote
from app.textutil import scrub_llm_text


def test_question_split_stable_and_fraction(monkeypatch):
    monkeypatch.setattr(settings, "eval_dev_fraction", 0.3)
    ids = [f"q{i:04d}" for i in range(1000)]
    labels = [E.question_split(i) for i in ids]
    # Stability
    assert all(E.question_split(i) == lab for i, lab in zip(ids, labels))
    # Only two labels
    assert set(labels) <= {"dev", "test"}
    # Fraction roughly 0.3 (binomial: allow wide band for 1000 samples)
    n_dev = sum(1 for lab in labels if lab == "dev")
    assert 220 <= n_dev <= 380, n_dev
    # Explicit fraction override
    assert E.question_split("anything", fraction=0.0) == "test"
    assert E.question_split("anything", fraction=1.0) == "dev"


def test_filter_by_split_partitions():
    qs = [
        E.GoldQuestion(id="a", qtype="order", question="q", gold_label="before"),
        E.GoldQuestion(id="b", qtype="order", question="q", gold_label="after"),
        E.GoldQuestion(id="c", qtype="order", question="q", gold_label="before"),
    ]
    all_ = E.filter_by_split(qs, "all")
    assert len(all_) == 3
    dev = E.filter_by_split(qs, "dev")
    test = E.filter_by_split(qs, "test")
    assert sorted(q.id for q in dev + test) == sorted(q.id for q in qs)
    assert not (set(q.id for q in dev) & set(q.id for q in test))


def test_synth_question_phrases_are_clean():
    """Questions must not embed discourse prefixes or [E####] tags."""
    params = synth.SynthParams(
        seed=1, events=40, flashback_rate=0.2, nesting=2,
        prophecy_rate=0.05, hypothetical_rate=0.05,
        separate_tale_rate=0.05, parallel_rate=0.15,
    )
    result = synth.generate(params)
    for q in result.gold:
        assert "[E" not in q.question, q.question
        assert "Meanwhile" not in q.question, q.question
        assert "At the same hour" not in q.question, q.question
        assert "'s " in q.question  # gerund NP form
        for key, ev in (q.events or {}).items():
            assert "[E" not in ev.desc
            assert ev.id.startswith("E")  # id stays internal gold key
        # Evidence quotes still verbatim for locate_quote
        for ev in q.evidence:
            assert "[E" in ev.quote


def test_scrub_llm_text_strips_e_tags():
    assert scrub_llm_text("Brann Rook shared the bread [E0013].") == \
        "Brann Rook shared the bread."


def test_synth_quotes_locate_and_strata(tmp_path):
    params = synth.SynthParams(
        seed=1, events=40, flashback_rate=0.2, nesting=2,
        prophecy_rate=0.05, hypothetical_rate=0.05,
        separate_tale_rate=0.05, parallel_rate=0.15,
    )
    result = synth.generate(params)
    assert len(result.events) == 40

    bad = synth.verify_gold_quotes(result.text, result.gold)
    assert bad == [], bad

    # Every gold quote locates exactly (offsets match locate_quote)
    for q in result.gold:
        for ev in q.evidence:
            loc = locate_quote(result.text, ev.quote)
            assert loc is not None
            assert ev.char_start == loc[0] and ev.char_end == loc[1]

    # Strata present; parallel rate drives unordered pairs
    strata = Counter(q.stratum for q in result.gold)
    assert strata.get("aligned", 0) >= 1
    assert strata.get("inverted", 0) >= 1
    assert strata.get("unordered", 0) >= 1

    expected = synth.expected_frame_counts(params)
    # Frame counts should match rounded rate * events (assignment is exact round)
    assert result.frame_counts.get("recollection", 0) == expected["recollection"]
    assert result.frame_counts.get("prediction", 0) == expected["prediction"]
    assert result.frame_counts.get("hypothetical", 0) == expected["hypothetical"]
    assert result.frame_counts.get("separate_tale", 0) == expected["separate_tale"]
    n_parallel = sum(1 for e in result.events if e.parallel_group)
    assert n_parallel == expected["parallel_events"] or n_parallel == expected["parallel_events"] - (
        expected["parallel_events"] % 2
    )

    meta = synth.write_outputs(result, tmp_path / "s1")
    assert (tmp_path / "s1" / "synth.pdf").is_file()
    assert (tmp_path / "s1" / "gold.jsonl").is_file()
    assert (tmp_path / "s1" / "params.json").is_file()
    saved = json.loads((tmp_path / "s1" / "params.json").read_text())
    assert saved["seed"] == 1 and saved["events"] == 40
    assert meta["n_gold"] == len(result.gold)
