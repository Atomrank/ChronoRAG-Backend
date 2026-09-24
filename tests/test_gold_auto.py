import json

from app import gold_auto as GA
from app.config import settings
from app.eval_runner import load_gold
from app.gold_propose import GoldProposal, PEvent, PFactual, PQuestion
from tests.test_eval_runner import _doc

EVENTS = [PEvent(id="E1", desc="the promise", quote="the old king made a promise to the river folk"),
          PEvent(id="E2", desc="the council", quote="the council met in the great hall"),
          PEvent(id="E3", desc="the demand", quote="recalled the promise and demanded the tribute"),
          PEvent(id="E4", desc="bad", quote="not a quote from this text at all anywhere")]
TRUTH = {"the promise": 0, "the council": 1, "the demand": 2}


def _setup(tmp_path, monkeypatch, judges, fulltext="", doc_fits=True, flaky=None):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "doc_x_document.json").write_text(json.dumps(_doc().to_dict()))
    monkeypatch.setattr(settings, "gold_judges", judges)
    monkeypatch.setattr(settings, "gold_fulltext_judge", fulltext)
    monkeypatch.setattr(settings, "gold_max_input_chars", 10_000 if doc_fits else 50)

    def fake(system, user, model, **k):
        if model is GoldProposal:
            return GoldProposal(events=EVENTS, questions=[
                PQuestion(a="E2", b="E1", label="after", justification=""),
                PQuestion(a="E1", b="E3", label="before", justification=""),
                PQuestion(a="E1", b="E4", label="before", justification=""),
                PQuestion(a="E2", b="E3", label="before", justification="")],
                factual=[PFactual(question="Who promised?", answer="the old king",
                                  quote="the old king made a promise to the river folk")])
        if model is GA.ChunkEvents:
            return GA.ChunkEvents(events=EVENTS, factual=[])
        if model is GA.PairList:
            return GA.PairList(pairs=[GA.Pair(a="E2", b="E1", intent="told_out_of_order"),
                                      GA.Pair(a="E2", b="E3", intent="told_in_order")])
        # judge: parse the two event descriptions
        lines = user.splitlines()
        a = lines[0].split(": ", 1)[1]
        b = next(l for l in lines if l.startswith("SECOND EVENT")).split(": ", 1)[1]
        if flaky and k.get("deployment") == flaky and a == "the council":
            return GA.Judgement(label="cannot_determine", reason="")
        lab = "before" if TRUTH[a] < TRUTH[b] else "after"
        return GA.Judgement(label=lab, reason="")
    monkeypatch.setattr(GA.llm, "chat_structured", fake)


def test_auto_single_call(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, "gpt-4o, local:qwen")
    out = tmp_path / "silver.jsonl"
    rep = GA.build("doc_x", 4, out)
    gold = load_gold(out)
    order = [g for g in gold if g.qtype == "order"]
    assert rep["mode"] == "single_call" and rep["events_rejected_quote"] == 1
    assert sorted((g.events["A"].desc, g.events["B"].desc, g.gold_label) for g in order) == [
        ("the council", "the demand", "before"), ("the council", "the promise", "after"),
        ("the promise", "the demand", "before")]
    assert {g.stratum for g in order} == {"aligned"}          # text order = story order here
    assert rep["judge_kappa"]["gpt-4o~local:qwen"] == 1.0
    assert (tmp_path / "silver_spot_check.csv").exists()
    assert any(g.qtype == "factual" for g in gold)


def test_disagreement_drops_pair(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, "gpt-4o,local:qwen", flaky="local:qwen")
    rep = GA.build("doc_x", 4, tmp_path / "s.jsonl")
    assert rep["dropped"].get("judges_disagree", 0) + rep["dropped"].get(
        "position_inconsistent", 0) >= 1


def test_two_pass_mode(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, "gpt-4o", doc_fits=False)
    rep = GA.build("doc_x", 2, tmp_path / "s.jsonl")
    assert rep["mode"] == "two_pass" and rep["pairs_kept"] == 2


def test_consensus_rules():
    assert GA.consensus({"a": ("before", "before"), "b": ("before", "before")}) == ("before", "agreed")
    assert GA.consensus({"a": ("before", "after")})[1] == "position_inconsistent"
    assert GA.consensus({"a": ("before", "before"), "b": ("after", "after")})[1] == "judges_disagree"
    assert GA.consensus({"a": ("invalid", "before")})[1] == "invalid"
    assert GA.consensus({"a": ("error:X", "before")})[1] == "judge_error"
    assert GA.cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
