"""Entity resolution for Kaalkram v2: surface forms -> canonical entity ids."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import buildlog, llm
from ..config import settings
from .extract import ExtractedAlias, ExtractedKinship, ExtractedMention
from .prompts import ENTITY_SYSTEM
from .schemas import EntityResolution


@dataclass
class ResolvedEntity:
    id: str
    canonical: str
    surfaces: list[str] = field(default_factory=list)


@dataclass
class ResolvedKinship:
    parent: str          # entity id
    child: str           # entity id
    quote: str = ""
    start: int | None = None
    end: int | None = None


@dataclass
class EntityResult:
    entities: list[ResolvedEntity]
    mentions: list[ExtractedMention]   # participants/subject mapped to entity ids
    kinship: list[ResolvedKinship]
    # (surface, mention_id|None) -> entity_id; mention_id None for alias/kinship-only
    form_to_entity: dict[tuple[str, str | None], str] = field(default_factory=dict)
    # surface -> entity_id when unambiguous; omitted for collision surfaces
    surface_to_entity: dict[str, str] = field(default_factory=dict)


@dataclass
class _Occ:
    idx: int
    surface: str
    mention_id: str | None
    start: int | None
    end: int | None
    contexts: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Label sent to the LLM; disambiguated when mention-scoped."""
        if self.mention_id:
            return f"{self.surface}#{self.mention_id}"
        return self.surface


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def _norm(s: str) -> str:
    s = s.casefold().strip()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _context_snip(text: str, start: int | None, end: int | None, radius: int) -> str:
    if start is None or end is None or not text:
        return ""
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return text[lo:hi].strip()


def _collect_occurrences(
    doc_text: str,
    mentions: list[ExtractedMention],
    aliases: list[ExtractedAlias],
    kinship: list[ExtractedKinship],
) -> list[_Occ]:
    radius = settings.v2_entity_context_chars
    max_ctx = settings.v2_entity_max_contexts
    # Group contexts by (surface, mention_id) so each mention-local form is one occ.
    buckets: dict[tuple[str, str | None], list[str]] = defaultdict(list)
    spans: dict[tuple[str, str | None], tuple[int | None, int | None]] = {}

    def add(surface: str, mention_id: str | None, start: int | None, end: int | None) -> None:
        surface = (surface or "").strip()
        if not surface:
            return
        key = (surface, mention_id)
        snip = _context_snip(doc_text, start, end, radius)
        if snip and snip not in buckets[key] and len(buckets[key]) < max_ctx:
            buckets[key].append(snip)
        if key not in spans:
            spans[key] = (start, end)

    for m in mentions:
        for p in m.participants:
            add(p.get("surface") or "", m.id, m.start, m.end)
        if m.subject:
            add(m.subject, m.id, m.start, m.end)

    for a in aliases:
        add(a.name_a, None, a.start, a.end)
        add(a.name_b, None, a.start, a.end)

    for k in kinship:
        add(k.parent, None, k.start, k.end)
        add(k.child, None, k.start, k.end)

    occs: list[_Occ] = []
    for i, ((surface, mid), ctxs) in enumerate(buckets.items()):
        s, e = spans.get((surface, mid), (None, None))
        occs.append(_Occ(idx=i, surface=surface, mention_id=mid, start=s, end=e, contexts=ctxs))
    return occs


def _embed_texts(occs: list[_Occ], embed_fn: Callable) -> np.ndarray:
    texts = []
    for o in occs:
        parts = [o.surface] + o.contexts
        texts.append(" | ".join(parts))
    vecs = embed_fn(texts)
    return np.asarray(vecs, dtype=np.float64)


def _candidate_groups(
    occs: list[_Occ],
    aliases: list[ExtractedAlias],
    emb: np.ndarray,
    *,
    doc_id: str,
) -> list[list[int]]:
    n = len(occs)
    if n == 0:
        return []
    uf = _UF(n)
    by_norm: dict[str, list[int]] = defaultdict(list)
    for o in occs:
        by_norm[_norm(o.surface)].append(o.idx)
    for idxs in by_norm.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    # Alias statements link any occurrences of name_a with name_b.
    surf_to_idxs: dict[str, list[int]] = defaultdict(list)
    for o in occs:
        surf_to_idxs[o.surface].append(o.idx)
        surf_to_idxs[_norm(o.surface)].append(o.idx)
    for a in aliases:
        left = surf_to_idxs.get(a.name_a) or surf_to_idxs.get(_norm(a.name_a), [])
        right = surf_to_idxs.get(a.name_b) or surf_to_idxs.get(_norm(a.name_b), [])
        for i in left:
            for j in right:
                uf.union(i, j)

    # Embedding nearest neighbours of form+contexts (top-k, cosine >= threshold).
    thr = settings.v2_entity_cosine
    top_k = settings.v2_entity_nn_top
    buildlog.record(doc_id, "entity_cosine_threshold", detail={"v2_entity_cosine": thr})
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            sims.append((_cosine(emb[i], emb[j]), j))
        sims.sort(reverse=True)
        for score, j in sims[:top_k]:
            if score >= thr:
                uf.union(i, j)

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)
    return list(comps.values())


def _split_large_group(idxs: list[int], emb: np.ndarray, max_size: int) -> list[list[int]]:
    """Recursively bipartition by farthest-pair assignment until size <= max_size."""
    if len(idxs) <= max_size:
        return [idxs]
    # Find farthest pair within the group.
    best, a, b = -1.0, idxs[0], idxs[0]
    for i, ii in enumerate(idxs):
        for jj in idxs[i + 1:]:
            d = 1.0 - _cosine(emb[ii], emb[jj])
            if d > best:
                best, a, b = d, ii, jj
    left, right = [], []
    for ii in idxs:
        if _cosine(emb[ii], emb[a]) >= _cosine(emb[ii], emb[b]):
            left.append(ii)
        else:
            right.append(ii)
    if not left or not right:
        # Degenerate: hard-split in half.
        mid = len(idxs) // 2
        left, right = idxs[:mid], idxs[mid:]
    out: list[list[int]] = []
    for part in (left, right):
        out.extend(_split_large_group(part, emb, max_size))
    return out


def _group_ambiguous(occs: list[_Occ], idxs: list[int]) -> bool:
    """True if the same surface appears under more than one mention_id in the group."""
    by_surf: dict[str, set[str | None]] = defaultdict(set)
    for i in idxs:
        o = occs[i]
        by_surf[o.surface].add(o.mention_id)
    return any(len(mids) > 1 for mids in by_surf.values())


def _user_message(occs: list[_Occ], idxs: list[int], aliases: list[ExtractedAlias],
                  ambiguous: bool) -> str:
    lines = ["FORMS:"]
    for i in idxs:
        o = occs[i]
        label = o.label if ambiguous else o.surface
        ctx = " || ".join(o.contexts) if o.contexts else "(no local context)"
        lines.append(f"- {label}\n  contexts: {ctx}")
    lines.append("\nALIAS STATEMENTS FROM THE TEXT:")
    surfaces = {occs[i].surface for i in idxs}
    norms = {_norm(s) for s in surfaces}
    any_alias = False
    for a in aliases:
        if (a.name_a in surfaces or _norm(a.name_a) in norms) and (
                a.name_b in surfaces or _norm(a.name_b) in norms):
            lines.append(f"- {a.name_a!r} = {a.name_b!r}  ({a.quote!r})")
            any_alias = True
    if not any_alias:
        lines.append("(none in this group)")
    if ambiguous:
        lines.append(
            "\nNOTE: Some surface strings appear more than once with different "
            "mention tags (name#mention_id). Treat each tagged form as a separate "
            "candidate; put two tagged forms in the same cluster only if they are "
            "the same individual. Return members using the exact labels above."
        )
    return "\n".join(lines)


def _parse_member(label: str, ambiguous: bool) -> tuple[str, str | None]:
    if ambiguous and "#" in label:
        surface, mid = label.rsplit("#", 1)
        return surface, mid
    return label, None


def _llm_resolve_group(
    occs: list[_Occ],
    idxs: list[int],
    aliases: list[ExtractedAlias],
    *,
    doc_id: str,
    chat_fn: Callable,
) -> list[tuple[list[int], str]]:
    """Return (occurrence-index cluster, canonical name) pairs."""
    ambiguous = _group_ambiguous(occs, idxs)
    user = _user_message(occs, idxs, aliases, ambiguous)
    if len(user) > settings.llm_max_input_chars:
        mid = len(idxs) // 2
        buildlog.record(doc_id, "entity_group_split_input",
                        detail={"size": len(idxs), "reason": "input_too_long"})
        left = _llm_resolve_group(occs, idxs[:mid], aliases, doc_id=doc_id, chat_fn=chat_fn)
        right = _llm_resolve_group(occs, idxs[mid:], aliases, doc_id=doc_id, chat_fn=chat_fn)
        return left + right

    try:
        result: EntityResolution = chat_fn(ENTITY_SYSTEM, user, EntityResolution)
    except Exception as exc:
        buildlog.record(doc_id, "failed_entity_group",
                        detail={"size": len(idxs), "error": f"{type(exc).__name__}: {exc}"})
        return [([i], occs[i].surface) for i in idxs]

    label_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i in idxs:
        o = occs[i]
        label_to_idxs[o.label if ambiguous else o.surface].append(i)
        if ambiguous:
            label_to_idxs[o.surface].append(i)

    clusters: list[tuple[list[int], str]] = []
    assigned: set[int] = set()
    for cl in result.clusters:
        members: list[int] = []
        for mem in cl.members:
            surface, mid = _parse_member(mem, ambiguous)
            if ambiguous and mid is not None:
                for i in idxs:
                    if occs[i].surface == surface and occs[i].mention_id == mid:
                        members.append(i)
            else:
                for i in label_to_idxs.get(mem, []):
                    members.append(i)
                if surface != mem:
                    for i in label_to_idxs.get(surface, []):
                        members.append(i)
        members = sorted(set(members))
        if members:
            canon = cl.canonical or occs[members[0]].surface
            if ambiguous and "#" in canon:
                canon = canon.rsplit("#", 1)[0]
            clusters.append((members, canon))
            assigned.update(members)

    for i in idxs:
        if i not in assigned:
            buildlog.record(doc_id, "entity_unassigned_form",
                            detail={"surface": occs[i].surface, "mention_id": occs[i].mention_id})
            clusters.append(([i], occs[i].surface))
    return clusters


def _lookup_entity(
    form_to_entity: dict[tuple[str, str | None], str],
    surface: str,
    mention_id: str | None,
) -> str | None:
    if not surface:
        return None
    if (surface, mention_id) in form_to_entity:
        return form_to_entity[(surface, mention_id)]
    # Fall back to surface-only (alias/kinship keys) then any mention-scoped entry.
    if (surface, None) in form_to_entity:
        return form_to_entity[(surface, None)]
    matches = [eid for (s, mid), eid in form_to_entity.items() if s == surface]
    if len(set(matches)) == 1:
        return matches[0]
    return None


def resolve(
    doc_text: str,
    mentions: list[ExtractedMention],
    aliases: list[ExtractedAlias] | None = None,
    kinship: list[ExtractedKinship] | None = None,
    *,
    doc_id: str = "",
    embed_fn: Callable | None = None,
    chat_fn: Callable | None = None,
) -> EntityResult:
    """
    Resolve surface forms to canonical entity ids (`ent_<n>`).

    Parameters
    ----------
    doc_text : full document text (for ±context windows)
    mentions : extracted mentions (participants/subject surfaces)
    aliases, kinship : alias equations and parent/child name pairs
    embed_fn : optional `list[str] -> list[list[float]]` (defaults to llm.embed)
    chat_fn : optional `(system, user, model_cls) -> EntityResolution`
              (defaults to llm.chat_structured at temperature 0)

    Returns
    -------
    EntityResult with entities, mentions (participants gain `entity_id`, subject
    becomes entity id when resolvable), kinship with entity ids, and lookup maps.
    """
    aliases = aliases or []
    kinship = kinship or []
    embed_fn = embed_fn or llm.embed

    def _chat(system: str, user: str, model_cls):
        if chat_fn:
            return chat_fn(system, user, model_cls)
        return llm.chat_structured(system, user, model_cls, temperature=0.0)

    occs = _collect_occurrences(doc_text, mentions, aliases, kinship)
    if not occs:
        return EntityResult(entities=[], mentions=list(mentions), kinship=[],
                            form_to_entity={}, surface_to_entity={})

    try:
        emb = _embed_texts(occs, embed_fn)
    except Exception as exc:
        buildlog.record(doc_id, "failed_entity_embed",
                        detail={"error": f"{type(exc).__name__}: {exc}"})
        raise

    raw_groups = _candidate_groups(occs, aliases, emb, doc_id=doc_id)
    max_g = settings.v2_entity_max_group
    groups: list[list[int]] = []
    for g in raw_groups:
        if len(g) > max_g:
            buildlog.record(doc_id, "entity_group_split_clustering",
                            detail={"size": len(g), "max": max_g})
            groups.extend(_split_large_group(g, emb, max_g))
        else:
            groups.append(g)

    # LLM per group -> union-find across all occurrence indices.
    uf = _UF(len(occs))
    occ_canon: dict[int, str] = {}
    for g in groups:
        if len(g) == 1:
            occ_canon[g[0]] = occs[g[0]].surface
            continue
        for cl_idxs, canon in _llm_resolve_group(
                occs, g, aliases, doc_id=doc_id, chat_fn=_chat):
            for j in cl_idxs[1:]:
                uf.union(cl_idxs[0], j)
            for i in cl_idxs:
                occ_canon[i] = canon

    # Assign ent_<n> ids in stable order of first occurrence offset / index.
    roots_order: list[int] = []
    seen_roots: set[int] = set()
    for o in sorted(occs, key=lambda x: (x.start is None, x.start or 0, x.idx)):
        r = uf.find(o.idx)
        if r not in seen_roots:
            seen_roots.add(r)
            roots_order.append(r)

    root_to_eid: dict[int, str] = {}
    entities: list[ResolvedEntity] = []
    members_by_root: dict[int, list[_Occ]] = defaultdict(list)
    for o in occs:
        members_by_root[uf.find(o.idx)].append(o)

    for n, root in enumerate(roots_order, start=1):
        eid = f"ent_{n}"
        root_to_eid[root] = eid
        mems = members_by_root[root]
        surfaces = list(dict.fromkeys(m.surface for m in mems))
        canons = [occ_canon[m.idx] for m in mems if m.idx in occ_canon]
        canonical = max(set(canons), key=canons.count) if canons else surfaces[0]
        entities.append(ResolvedEntity(id=eid, canonical=canonical, surfaces=surfaces))

    form_to_entity: dict[tuple[str, str | None], str] = {}
    for o in occs:
        form_to_entity[(o.surface, o.mention_id)] = root_to_eid[uf.find(o.idx)]

    # surface_to_entity only when all mention-scoped mappings agree.
    by_surface: dict[str, set[str]] = defaultdict(set)
    for (surface, _mid), eid in form_to_entity.items():
        by_surface[surface].add(eid)
    surface_to_entity = {s: next(iter(eids)) for s, eids in by_surface.items() if len(eids) == 1}

    # Map mentions.
    out_mentions: list[ExtractedMention] = []
    for m in mentions:
        parts = []
        for p in m.participants:
            surface = p.get("surface") or ""
            eid = _lookup_entity(form_to_entity, surface, m.id)
            rec = dict(p)
            if eid:
                rec["entity_id"] = eid
            parts.append(rec)
        subject_eid = _lookup_entity(form_to_entity, m.subject, m.id) if m.subject else ""
        out_mentions.append(ExtractedMention(
            id=m.id, local_id=m.local_id, window_id=m.window_id, frame_id=m.frame_id,
            start=m.start, end=m.end, para_id=m.para_id, quote=m.quote,
            description=m.description, mode=m.mode, event_type=m.event_type,
            subject=subject_eid or m.subject,
            participants=parts, location=m.location,
            time_expressions=list(m.time_expressions), posthumous=m.posthumous,
            is_telling=m.is_telling, tells_frame=m.tells_frame, sample_idx=m.sample_idx,
            event_id=m.event_id,
        ))

    out_kin: list[ResolvedKinship] = []
    for k in kinship:
        pid = _lookup_entity(form_to_entity, k.parent, None) or surface_to_entity.get(k.parent)
        cid = _lookup_entity(form_to_entity, k.child, None) or surface_to_entity.get(k.child)
        if not pid or not cid:
            buildlog.record(doc_id, "unresolved_kinship",
                            detail={"parent": k.parent, "child": k.child})
            continue
        out_kin.append(ResolvedKinship(
            parent=pid, child=cid, quote=k.quote, start=k.start, end=k.end))

    return EntityResult(
        entities=entities,
        mentions=out_mentions,
        kinship=out_kin,
        form_to_entity=form_to_entity,
        surface_to_entity=surface_to_entity,
    )
