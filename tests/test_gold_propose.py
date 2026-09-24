import csv
import json

from app import gold_propose as GP
from app.config import settings
from app.eval_runner import load_gold
from tests.test_eval_runner import TEXT, _doc


def test_stratum_rules():
    assert GP._stratum("before", 10, 50) == "aligned"
    assert GP._stratum("before", 50, 10) == "inverted"
    assert GP._stratum("after", 50, 10) == "aligned"
    assert GP._stratum("cannot_determine", 1, 2) == "unordered"


def test_propose_and_import(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "doc_x_document.json").write_text(json.dumps(_doc().to_dict()))
    fake = GP.GoldProposal(
        events=[GP.PEvent(id="E1", desc="the promise", quote="the old king made a promise to the river folk"),
                GP.PEvent(id="E2", desc="the council", quote="the council met in the great hall"),
                GP.PEvent(id="E3", desc="bad", quote="words that are not in the text at all")],
        questions=[GP.PQuestion(a="E2", b="E1", label="after", justification="x"),
                   GP.PQuestion(a="E1", b="E3", label="before", justification="x")],
        factual=[GP.PFactual(question="Who made a promise?", answer="the old king",
                             quote="the old king made a promise to the river folk")])
    monkeypatch.setattr(GP.llm, "chat_structured", lambda *a, **k: fake)
    out = tmp_path / "p.csv"
    rep = GP.propose("doc_x", 2, out)
    assert rep["exported_rows"] == 2 and rep["strata"]["aligned"] == 1
    rows = list(csv.DictReader(out.open()))
    rows[0]["confirm"], rows[0]["verified_by"] = "Y", "AT"
    rows[1]["confirm"] = "y"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    GP.import_confirmed("doc_x", out, tmp_path / "gold.jsonl")
    gold = load_gold(tmp_path / "gold.jsonl")
    assert [g.qtype for g in gold] == ["order", "factual"]
    assert gold[0].gold_label == "after" and gold[0].evidence[0].char_start is not None
    assert TEXT[gold[0].evidence[1].char_start:gold[0].evidence[1].char_end].startswith("the old king")
