import json

from app import eval_runner as E
from app.config import settings
from app.ingest_v2 import Document, Page, Paragraph
from app.schemas import PipelineAnswer

TEXT = ("Long before the council, the old king made a promise to the river folk. "
        "Many years later the council met in the great hall. "
        "At the council the envoy recalled the promise and demanded the tribute.")


def _doc():
    return Document(text=TEXT, pages=[Page(1, 0, len(TEXT))], sections=[],
                    paragraphs=[Paragraph("p1", 0, len(TEXT), 1, 1, None)],
                    structure_source="none")


def _gold(path):
    rows = [
        {"id": "q1", "qtype": "order", "stratum": "inverted",
         "question": "Did the promise happen before or after the council?",
         "gold_label": "before",
         "events": {"A": {"id": "E1", "desc": "the promise"}, "B": {"id": "E2", "desc": "the council"}},
         "evidence": [{"group": "A", "quote": "the old king made a promise to the river folk"},
                      {"group": "B", "quote": "the council met in the great hall"}]},
        {"id": "q2", "qtype": "order", "stratum": "aligned",
         "question": "Did the council happen before or after the demand?",
         "gold_label": "before",
         "events": {"A": {"id": "E2", "desc": "the council"}, "B": {"id": "E3", "desc": "the demand"}},
         "evidence": [{"group": "A", "quote": "the council met in the great hall"},
                      {"group": "B", "quote": "recalled the promise and demanded the tribute"}]},
        {"id": "q3", "qtype": "order", "stratum": "inverted", "question": "misquoted",
         "gold_label": "after", "evidence": [{"group": "A", "quote": "this text does not exist anywhere"}]},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_verify_run_report_compare(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "eval_dir", str(tmp_path / "eval"))
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "doc_x_document.json").write_text(json.dumps(_doc().to_dict()))

    raw, ver = tmp_path / "raw.jsonl", tmp_path / "gold_v1.jsonl"
    _gold(raw)
    kept, rejected = E.verify_gold("doc_x", E.load_gold(raw))
    assert [q.id for q in kept] == ["q1", "q2"] and rejected[0]["id"] == "q3"
    ver.write_text("".join(q.model_dump_json() + "\n" for q in kept))

    def fake(label_for):
        def fn(doc_id, q, k):
            rel = label_for(q)
            return PipelineAnswer(pipeline="fake", answer="x", relation=rel, confidence=0.8,
                                  cited_spans=[[0, 60]], latency_ms=10, prompt_tokens=5,
                                  completion_tokens=1,
                                  retrieved=[{"rank": 1, "spans": [[0, 75]], "tokens": 20},
                                             {"rank": 2, "spans": [[75, 130]], "tokens": 20},
                                             {"rank": 3, "spans": [[130, 300]], "tokens": 20}])
        return fn
    order = {"the promise": 0, "the council": 1, "the demand": 2}

    def truth(q):
        import re
        a, b = re.match(r"Did (.+) happen before or after (.+)\?", q).groups()
        return "before" if order[a] < order[b] else "after"
    good = fake(truth)
    bad = fake(lambda q: "after")
    monkeypatch.setattr(E, "_adapters", lambda: {"naive": bad, "kaalkram_v1": good})

    r_bad = E.run("doc_x", ver, "naive", n_triples=1, workers=2, use_db=False)
    r_good = E.run("doc_x", ver, "kaalkram_v1", n_triples=1, workers=2, use_db=False)
    s = json.loads((tmp_path / "eval" / r_good / "summary.json").read_text())["repeats"]["0"]
    assert s["ordering"]["overall"]["accuracy"] == 1.0
    assert s["retrieval"]["recall@3"]["value"] == 1.0
    assert s["consistency"]["symmetry"] == 1.0
    s_bad = json.loads((tmp_path / "eval" / r_bad / "summary.json").read_text())["repeats"]["0"]
    assert s_bad["ordering"]["overall"]["accuracy"] == 0.0
    assert s_bad["consistency"]["symmetry"] == 0.0          # says "after" both ways
    cfg = json.loads((tmp_path / "eval" / r_good / "config.json").read_text())
    assert cfg["settings"]["azure_openai_api_key"] == "***"
    assert cfg["gold_sha1"] and cfg["doc_text_sha1"]

    out = E.report([r_bad, r_good])
    assert "acc_inverted" in (out / "report.md").read_text()
    cmp_ = E.compare(r_good, r_bad)
    assert cmp_["n"] == 2 and cmp_["mcnemar"]["a_only"] == 2
