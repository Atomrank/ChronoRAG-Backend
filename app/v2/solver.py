"""
Temporal constraint solver for Kaalkram v2.

Each event is an interval with a start point `<id>:s` and an end point `<id>:e`.
Constraints become edges between points:

    A before B        A:e --<-->  B:s          (strict)
    A during B        B:s --<=--> A:s,  A:e --<=--> B:e
    A simultaneous B  A:s <=> B:s,  A:e <=> B:e (weak, both ways)
    A overlaps B      A:s --<=--> B:e,  B:s --<=--> A:e
    every event       A:s --<=--> A:e          (internal, never removed)

Why this is exact for before/after: the only edges that go from an END point to
a START point are `before` edges, so "A:e reaches B:s" holds iff a chain of
evidence proves A ended before B started. No total order is ever imposed:
events with no connecting chain come out as `cannot_determine`.

Contradictions are cycles that contain a strict edge (a cycle of weak edges
just means "equal"). They are broken by repeatedly removing the lowest-weight
edge on a shortest contradicting cycle; every removal is logged with the cycle
it broke, so contradictions in the text are reported, not hidden.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Literal

Rel = Literal["before", "after", "during", "contains", "simultaneous", "overlap"]
INTERNAL = "internal"


@dataclass
class Constraint:
    a: str
    b: str
    rel: Rel                      # a <rel> b
    p: float                      # probability-like weight in (0, 1)
    source: str                   # explicit | time_expression | causal | succession | frame | lifecycle | genealogy
    evidence: list[dict] = field(default_factory=list)   # [{"span": [s, e], "note": str}]
    id: str = ""


@dataclass
class Edge:
    u: str
    v: str
    strict: bool
    p: float
    sources: list[str]
    constraint_ids: list[str]
    evidence: list[dict]

    @property
    def key(self) -> tuple[str, str, bool]:
        return (self.u, self.v, self.strict)

    @property
    def removable(self) -> bool:
        return INTERNAL not in self.sources


def noisy_or(ps: list[float]) -> float:
    q = 1.0
    for p in ps:
        q *= (1.0 - min(max(p, 0.0), 0.999999))
    return 1.0 - q


def _edges_for(c: Constraint) -> list[tuple[str, str, bool]]:
    a, b, r = c.a, c.b, c.rel
    if r == "after":
        a, b, r = b, a, "before"
    if r == "contains":
        a, b, r = b, a, "during"
    if r == "before":
        return [(f"{a}:e", f"{b}:s", True)]
    if r == "during":
        return [(f"{b}:s", f"{a}:s", False), (f"{a}:e", f"{b}:e", False)]
    if r == "simultaneous":
        return [(f"{a}:s", f"{b}:s", False), (f"{b}:s", f"{a}:s", False),
                (f"{a}:e", f"{b}:e", False), (f"{b}:e", f"{a}:e", False)]
    if r == "overlap":
        return [(f"{a}:s", f"{b}:e", False), (f"{b}:s", f"{a}:e", False)]
    raise ValueError(f"unknown relation {r}")


class TemporalGraph:
    def __init__(self, event_ids: list[str], discourse_pos: dict[str, int] | None = None):
        self.events = list(dict.fromkeys(event_ids))
        self.pos = discourse_pos or {}
        self.edges: dict[tuple[str, str, bool], Edge] = {}
        self.removed: list[dict] = []
        self._reach: dict[str, int] | None = None
        for ev in self.events:
            self._add(f"{ev}:s", f"{ev}:e", False, 1.0, INTERNAL, f"int_{ev}", [])

    # ---------------- building ----------------
    def _add(self, u, v, strict, p, source, cid, evidence):
        key = (u, v, strict)
        e = self.edges.get(key)
        if e is None:
            self.edges[key] = Edge(u, v, strict, p, [source], [cid], list(evidence))
        else:
            e.p = 1.0 if INTERNAL in e.sources else noisy_or([e.p, p])
            e.sources.append(source)
            e.constraint_ids.append(cid)
            e.evidence.extend(evidence)
        self._reach = None

    def add(self, c: Constraint) -> None:
        if c.a == c.b:
            return
        for ev in (c.a, c.b):
            if ev not in self.events:
                self.events.append(ev)
                self._add(f"{ev}:s", f"{ev}:e", False, 1.0, INTERNAL, f"int_{ev}", [])
        for u, v, strict in _edges_for(c):
            self._add(u, v, strict, c.p, c.source, c.id or f"{c.source}:{c.a}>{c.b}", c.evidence)

    # ---------------- graph utils ----------------
    def _adj(self) -> dict[str, list[Edge]]:
        adj: dict[str, list[Edge]] = defaultdict(list)
        for e in self.edges.values():
            adj[e.u].append(e)
        for lst in adj.values():
            lst.sort(key=lambda e: (e.v, e.strict))
        return adj

    def _nodes(self) -> list[str]:
        return [f"{ev}:{x}" for ev in self.events for x in ("s", "e")]

    def _scc(self, adj) -> dict[str, int]:
        """Iterative Tarjan. Returns node -> component id."""
        index, low, comp = {}, {}, {}
        stack, on = [], set()
        counter = [0]
        cid = 0
        for root in self._nodes():
            if root in index:
                continue
            work = [(root, 0)]
            while work:
                node, i = work.pop()
                if i == 0:
                    index[node] = low[node] = counter[0]
                    counter[0] += 1
                    stack.append(node)
                    on.add(node)
                nbrs = adj.get(node, [])
                recurse = False
                for j in range(i, len(nbrs)):
                    w = nbrs[j].v
                    if w not in index:
                        work.append((node, j + 1))
                        work.append((w, 0))
                        recurse = True
                        break
                    if w in on:
                        low[node] = min(low[node], index[w])
                if recurse:
                    continue
                if low[node] == index[node]:
                    while True:
                        w = stack.pop()
                        on.discard(w)
                        comp[w] = cid
                        if w == node:
                            break
                    cid += 1
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
        return comp

    # ---------------- repair ----------------
    def repair(self, max_iter: int = 100000) -> dict:
        """Break every cycle that contains a strict edge."""
        it = 0
        while it < max_iter:
            it += 1
            adj = self._adj()
            comp = self._scc(adj)
            bad = sorted((e for e in self.edges.values()
                          if e.strict and comp[e.u] == comp[e.v]),
                         key=lambda e: (e.p, e.u, e.v))
            if not bad:
                break
            start = bad[0]
            cycle = [start] + self._shortest_path(adj, start.v, start.u, comp[start.u])
            victim = min((e for e in cycle if e.removable),
                         key=lambda e: (e.p, _SOURCE_RANK.get(e.sources[0], 0), e.u, e.v))
            self.removed.append({
                "removed": {"u": victim.u, "v": victim.v, "strict": victim.strict,
                            "p": victim.p, "sources": victim.sources,
                            "constraint_ids": victim.constraint_ids,
                            "evidence": victim.evidence},
                "cycle": [{"u": e.u, "v": e.v, "strict": e.strict, "p": round(e.p, 4),
                           "sources": e.sources} for e in cycle],
            })
            del self.edges[victim.key]
            self._reach = None
        return self.stats()

    @staticmethod
    def _shortest_path(adj, src, dst, component) -> list[Edge]:
        prev: dict[str, Edge] = {}
        seen = {src}
        q = deque([src])
        while q:
            n = q.popleft()
            if n == dst:
                break
            for e in adj.get(n, []):
                if e.v not in seen:
                    seen.add(e.v)
                    prev[e.v] = e
                    q.append(e.v)
        path, n = [], dst
        while n != src:
            e = prev[n]
            path.append(e)
            n = e.u
        return list(reversed(path))

    # ---------------- reachability ----------------
    def _closure(self) -> dict[str, int]:
        """Bitset reachability over the condensation DAG."""
        if self._reach is not None:
            return self._reach
        adj = self._adj()
        comp = self._scc(adj)
        n_comp = max(comp.values()) + 1 if comp else 0
        succ: dict[int, set[int]] = defaultdict(set)
        for e in self.edges.values():
            if comp[e.u] != comp[e.v]:
                succ[comp[e.u]].add(comp[e.v])
        # Tarjan numbers components in reverse topological order: successors first
        reach = [0] * n_comp
        for c in range(n_comp):
            bits = 1 << c
            for d in succ[c]:
                bits |= reach[d]
            reach[c] = bits
        self._comp = comp
        self._reach = {n: reach[comp[n]] for n in comp}
        return self._reach

    def reaches(self, u: str, v: str) -> bool:
        r = self._closure()
        if u not in r or v not in r:
            return False
        return bool(r[u] >> self._comp[v] & 1)

    def relation(self, a: str, b: str) -> dict:
        """Deterministic relation of event a to event b."""
        if a not in self.events or b not in self.events:
            return {"label": "cannot_determine", "detail": "unknown_event", "chain": []}
        if self.reaches(f"{a}:e", f"{b}:s"):
            return {"label": "before", "detail": "before", "chain": self.chain(f"{a}:e", f"{b}:s")}
        if self.reaches(f"{b}:e", f"{a}:s"):
            return {"label": "after", "detail": "after", "chain": self.chain(f"{b}:e", f"{a}:s")}
        if self.reaches(f"{b}:s", f"{a}:s") and self.reaches(f"{a}:e", f"{b}:e"):
            detail = ("simultaneous" if self.reaches(f"{a}:s", f"{b}:s")
                      and self.reaches(f"{b}:e", f"{a}:e") else "during")
            return {"label": "cannot_determine", "detail": detail,
                    "chain": self.chain(f"{b}:s", f"{a}:s") + self.chain(f"{a}:e", f"{b}:e")}
        if self.reaches(f"{a}:s", f"{b}:s") and self.reaches(f"{b}:e", f"{a}:e"):
            return {"label": "cannot_determine", "detail": "contains",
                    "chain": self.chain(f"{a}:s", f"{b}:s") + self.chain(f"{b}:e", f"{a}:e")}
        if self.reaches(f"{a}:s", f"{b}:e") and self.reaches(f"{b}:s", f"{a}:e"):
            return {"label": "cannot_determine", "detail": "overlap",
                    "chain": self.chain(f"{a}:s", f"{b}:e") + self.chain(f"{b}:s", f"{a}:e")}
        return {"label": "cannot_determine", "detail": "unordered", "chain": []}

    def chain(self, u: str, v: str) -> list[dict]:
        """Most-probable evidence path u -> v (max product of edge weights)."""
        adj = self._adj()
        dist = {u: 0.0}
        prev: dict[str, Edge] = {}
        heap = [(0.0, u)]
        while heap:
            d, n = heapq.heappop(heap)
            if n == v:
                break
            if d > dist.get(n, math.inf):
                continue
            for e in adj.get(n, []):
                w = 0.0 if not e.removable else -math.log(max(e.p, 1e-9))
                nd = d + w
                if nd < dist.get(e.v, math.inf):
                    dist[e.v] = nd
                    prev[e.v] = e
                    heapq.heappush(heap, (nd, e.v))
        if v not in prev and u != v:
            return []
        path, n = [], v
        while n != u:
            e = prev[n]
            if e.removable:
                path.append({"from": e.u, "to": e.v, "strict": e.strict, "p": round(e.p, 4),
                             "sources": sorted(set(e.sources)), "evidence": e.evidence[:3]})
            n = e.u
        return list(reversed(path))

    # ---------------- outputs ----------------
    def linear_extension(self) -> list[str]:
        """ONE order consistent with the graph, for display only. Ties broken by
        position in the book. Not evidence for any ordering claim."""
        self._closure()
        comp = self._comp
        succ: dict[int, set[int]] = defaultdict(set)
        indeg: dict[int, int] = defaultdict(int)
        members: dict[int, list[str]] = defaultdict(list)
        for ev in self.events:
            members[comp[f"{ev}:s"]].append(ev)
        for e in self.edges.values():
            cu, cv = comp[e.u], comp[e.v]
            if cu != cv and cv not in succ[cu]:
                succ[cu].add(cv)
                indeg[cv] += 1
        all_c = set(comp.values())
        key = lambda c: min((self.pos.get(ev, 10 ** 12) for ev in members[c]), default=10 ** 12)
        heap = [(key(c), c) for c in all_c if indeg[c] == 0]
        heapq.heapify(heap)
        order = []
        while heap:
            _, c = heapq.heappop(heap)
            order.extend(sorted(members[c], key=lambda ev: (self.pos.get(ev, 10 ** 12), ev)))
            for d in succ[c]:
                indeg[d] -= 1
                if indeg[d] == 0:
                    heapq.heappush(heap, (key(d), d))
        return order

    def stats(self) -> dict:
        by_src: dict[str, int] = defaultdict(int)
        for e in self.edges.values():
            for s in set(e.sources):
                if s != INTERNAL:
                    by_src[s] += 1
        removed_by: dict[str, int] = defaultdict(int)
        for r in self.removed:
            for s in set(r["removed"]["sources"]):
                removed_by[s] += 1
        return {
            "events": len(self.events),
            "edges": sum(1 for e in self.edges.values() if e.removable),
            "edges_by_source": dict(by_src),
            "removed_edges": len(self.removed),
            "removed_by_source": dict(removed_by),
            "removed_weight": round(sum(r["removed"]["p"] for r in self.removed), 4),
        }

    def to_dict(self) -> dict:
        return {"events": self.events, "pos": self.pos,
                "edges": [e.__dict__ for e in self.edges.values() if e.removable],
                "removed": self.removed}

    @staticmethod
    def from_dict(d: dict) -> "TemporalGraph":
        g = TemporalGraph(d["events"], d.get("pos"))
        for e in d["edges"]:
            g.edges[(e["u"], e["v"], e["strict"])] = Edge(**e)
        g.removed = d.get("removed", [])
        return g


# When weights tie, remove the edge from the source judged less reliable first.
_SOURCE_RANK = {"succession": 0, "lifecycle": 1, "genealogy": 1, "causal": 2,
                "frame": 3, "time_expression": 4, "explicit": 5}
