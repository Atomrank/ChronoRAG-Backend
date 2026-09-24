"""Scenario tests: the situations the Sabha Parva contains, in miniature."""
from app.v2.constraints import (Kinship, LocalRelation, Mention, all_constraints,
                                calibrate_sources, lifecycle, succession)
from app.v2.frames import FrameOp, FrameTracker
from app.v2.solver import Constraint, TemporalGraph


def G(events, cons):
    g = TemporalGraph(events, {e: i for i, e in enumerate(events)})
    for c in cons:
        g.add(c)
    g.repair()
    return g


def C(a, b, rel="before", p=0.9, src="explicit"):
    return Constraint(a, b, rel, p, src, [{"span": [0, 1]}])


# ---------------- solver ----------------
def test_transitive_before_and_unordered():
    g = G(list("ABCD"), [C("A", "B"), C("B", "C")])
    assert g.relation("A", "C")["label"] == "before"
    assert g.relation("C", "A")["label"] == "after"
    assert g.relation("A", "D") == {"label": "cannot_determine", "detail": "unordered", "chain": []}
    assert len(g.relation("A", "C")["chain"]) == 2


def test_no_total_order_is_invented():
    # two branches after A: B and C are not ordered with respect to each other
    g = G(list("ABC"), [C("A", "B"), C("A", "C")])
    assert g.relation("B", "C")["detail"] == "unordered"
    assert set(g.linear_extension()) == {"A", "B", "C"}


def test_during_propagates():
    # X during B, A before B  =>  A before X
    g = G(["A", "B", "X"], [C("A", "B"), C("X", "B", "during")])
    assert g.relation("A", "X")["label"] == "before"
    assert g.relation("X", "B")["detail"] == "during"


def test_simultaneous_and_overlap_are_not_before():
    g = G(list("ABC"), [C("A", "B", "simultaneous"), C("B", "C")])
    assert g.relation("A", "B")["detail"] == "simultaneous"
    assert g.relation("A", "C")["label"] == "before"
    g2 = G(list("AB"), [C("A", "B", "overlap")])
    assert g2.relation("A", "B")["detail"] == "overlap"


def test_contradiction_drops_weakest_and_is_logged():
    g = G(list("ABC"), [C("A", "B", p=0.9), C("B", "C", p=0.9),
                        C("C", "A", p=0.3, src="succession")])
    st = g.stats()
    assert st["removed_edges"] == 1 and st["removed_by_source"] == {"succession": 1}
    assert g.relation("A", "C")["label"] == "before"
    assert g.removed[0]["cycle"]


def test_weak_cycles_are_equality_not_contradiction():
    g = G(list("AB"), [C("A", "B", "simultaneous"), C("B", "A", "simultaneous")])
    assert g.stats()["removed_edges"] == 0


def test_duplicate_evidence_combines():
    g = G(list("AB"), [C("A", "B", p=0.6), C("A", "B", p=0.6, src="frame")])
    e = [e for e in g.edges.values() if e.removable][0]
    assert abs(e.p - 0.84) < 1e-9 and set(e.sources) == {"explicit", "frame"}


def test_roundtrip():
    g = G(list("ABC"), [C("A", "B"), C("B", "C")])
    g2 = TemporalGraph.from_dict(g.to_dict())
    assert g2.relation("A", "C")["label"] == "before"


def test_scale():
    n = 3000
    ev = [f"e{i}" for i in range(n)]
    g = G(ev, [C(ev[i], ev[i + 1], p=0.6, src="succession") for i in range(n - 1)]
          + [C(ev[2500], ev[10], p=0.3, src="causal")])            # one contradiction
    assert g.stats()["removed_edges"] == 1
    assert g.relation("e0", "e2999")["label"] == "before"          # no 80-hop limit


# ---------------- frames ----------------
def test_frame_stack():
    t = FrameTracker("u1", 0, 1000)
    t.apply([FrameOp("open", 100, "recollection", "Bhishma?", "", summary="old story")])
    assert [f["type"] for f in t.state()] == ["main", "recollection"]
    t.apply([FrameOp("open", 150, "separate_tale", "N", "L")])
    t.apply([FrameOp("close", 400, frame_ref=t.state()[1]["id"])])   # closes both
    t.finish()
    assert t.frame_at(120).type == "recollection"
    assert t.frame_at(200).type == "separate_tale"
    assert t.frame_at(500).type == "main"
    assert any(w["warning"] == "closed_inner_frame_implicitly" for w in t.warnings)


def test_frame_guards():
    t = FrameTracker("u1", 0, 100)
    t.apply([FrameOp("close", 10), FrameOp("open", 500, "recollection")])
    t.finish()
    kinds = {w["warning"] for w in t.warnings}
    assert {"close_with_empty_stack", "op_outside_unit"} <= kinds


# ---------------- end to end: flashback, prophecy, parallel campaigns ----------------
def _scenario():
    """
    main:   [0]  hall built            (occurs)
            [10] council meets         (occurs)
            [20] elder tells old story (telling of frame R)
              R (recollection): [30] rival born, [40] rival cursed
            [60] rival killed          (occurs)
            [70] two campaigns narrated one after another, but a relation says
                 they overlap           (occurs, occurs)
    P (prediction) at [90]: [95] "the rival's son will fall" -> never narrated as happening
    """
    t = FrameTracker("u", 0, 200)
    t.apply([FrameOp("open", 25, "recollection", "elder", "king"), FrameOp("close", 50)])
    t.apply([FrameOp("open", 90, "prediction", "sage", "king"), FrameOp("close", 99)])
    frames = t.finish()
    fid = lambda off: t.frame_at(off).id
    M = lambda i, ev, off, mode="occurs", **k: Mention(i, ev, fid(off), off, off + 5, mode, **k)
    ms = [
        M("m1", "hall", 0), M("m2", "council", 10),
        Mention("m3", "tell_R", fid(20), 20, 24, "occurs", is_telling=True,
                tells_frame=frames[1].id),
        M("m4", "rival_born", 30, event_type="birth", subject="rival",
          participants=[("rival", "patient")]),
        M("m5", "rival_cursed", 40, participants=[("rival", "patient")]),
        M("m6", "rival_killed", 60, event_type="death", subject="rival",
          participants=[("rival", "patient")]),
        M("m7", "north_campaign", 70), M("m8", "east_campaign", 75),
        Mention("m9", "tell_P", fid(89), 89, 90, "occurs", is_telling=True,
                tells_frame=frames[2].id),
        M("m10", "son_falls", 95, mode="predicted"),
    ]
    rels = [LocalRelation("m7", "m8", "overlap", "explicit_connective")]
    return frames, ms, rels


def test_scenario_end_to_end():
    frames, ms, rels = _scenario()
    cons = all_constraints(frames, ms, rels, [])
    g = G(sorted({m.event_id for m in ms}, key=lambda e: [m.start for m in ms if m.event_id == e][0]),
          cons)
    # flashback: told late, but placed before its telling and everything after it
    assert g.relation("rival_born", "tell_R")["label"] == "before"
    assert g.relation("rival_born", "north_campaign")["label"] == "before"
    # ...and NOT ordered against earlier main events the text never relates it to
    assert g.relation("rival_born", "council")["detail"] == "unordered"
    assert g.relation("rival_cursed", "hall")["detail"] == "unordered"
    # the told story is internally ordered
    assert g.relation("rival_born", "rival_cursed")["label"] == "before"
    # lifecycle: curse happens before death
    assert g.relation("rival_cursed", "rival_killed")["label"] == "before"
    # parallel campaigns: succession says before, explicit overlap wins
    r = g.relation("north_campaign", "east_campaign")
    assert r["label"] == "cannot_determine" and r["detail"] == "overlap"
    assert g.stats()["removed_by_source"].get("succession") == 1
    # prediction never narrated as happening: no ordering claim at all beyond the frame
    assert g.relation("son_falls", "hall")["detail"] == "unordered"


def test_prediction_that_comes_true_is_after_utterance():
    t = FrameTracker("u", 0, 100)
    t.apply([FrameOp("open", 10, "prediction", "sage", ""), FrameOp("close", 20)])
    frames = t.finish()
    ms = [Mention("t", "tell", frames[0].id, 5, 9, "occurs", is_telling=True,
                  tells_frame=frames[1].id),
          Mention("p", "vow_fulfilled", frames[1].id, 12, 15, "predicted"),
          Mention("x", "other", frames[0].id, 30, 35, "occurs"),
          Mention("o", "vow_fulfilled", frames[0].id, 60, 65, "occurs")]
    g = G(["tell", "other", "vow_fulfilled"], all_constraints(frames, ms, [], []))
    assert g.relation("tell", "vow_fulfilled")["label"] == "before"
    # the event is positioned by its OCCURRENCE, not by the page where it was foretold
    assert g.relation("other", "vow_fulfilled")["label"] == "before"


def test_in_frame_flashback_not_in_succession():
    ms = [Mention("a", "A", "F", 0, 5, "occurs"), Mention("b", "B", "F", 10, 15, "recounted"),
          Mention("c", "C", "F", 20, 25, "occurs")]
    s = succession(ms)
    assert [(c.a, c.b) for c in s] == [("A", "C")]


def test_posthumous_exempt_from_death_bound():
    ms = [Mention("d", "death", "F", 0, 5, "occurs", event_type="death", subject="x",
                  participants=[("x", "patient")]),
          Mention("g", "ghost", "F", 10, 15, "occurs", participants=[("x", "agent")],
                  posthumous=True)]
    assert not any(c.b == "death" and c.a == "ghost" for c in lifecycle(ms))


def test_genealogy_and_calibration():
    ms = [Mention("a", "pb", "F", 50, 55, "recounted", event_type="birth", subject="parent"),
          Mention("b", "cb", "F", 10, 15, "occurs", event_type="birth", subject="child")]
    cons = all_constraints([], ms, [], [Kinship("parent", "child")])
    assert any(c.source == "genealogy" and (c.a, c.b) == ("pb", "cb") for c in cons)
    cal = calibrate_sources(cons + [C("x", "y", src="succession")],
                            [("pb", "cb", "before"), ("x", "y", "after")])
    assert cal["genealogy"]["agree"] == 1 and cal["succession"]["agree"] == 0
