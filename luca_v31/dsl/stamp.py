"""LUCA DSL - complete partial copies of an exemplar (M1 'exemplar as key'; OPT178, extended OPT179).

The grid shows one or more whole motifs (multicolour objects, the exemplars) and fragments of the
same motif elsewhere: a single anchor cell, a few cells, scattered anchor cells, or solid blocks
when the copy is magnified. Each fragment is completed into the whole motif.

    spec = (exemplars, transforms, grouping, erase, scales, clip, recolour)
      exemplars  'largest' (the unique largest multicolour object) | 'all' multicolour objects |
                 'big' (multicolour objects with >= 3 cells and >= half the largest) |
                 'largest_norm' (unique largest after undoing each object's own magnification)
      transforms 'id' | 'flips' (id, mirror h/v, 180) | 'd8' (all rotations and reflections)
      grouping   4 | 8   : a fragment is a 4/8-connected group of non-exemplar cells
                 'scatter': fragments are found by placement; one copy may join several separate
                            anchor cells (needs >= 2 matched cells per copy)
      erase      grouping 4/8 : fragments no copy explains are removed
                 'scatter'    : the exemplars are removed (the motif MOVES onto its anchors)
      scales     (1,) | (1, 2, 3, 4): the copy may be magnified; the smallest scale wins
      clip       True: a copy may run off the grid edge (cells outside are dropped)
      recolour   True: motif colours that the fragment shows differently are substituted
                 (one-to-one; at least one colour must match exactly, the anchor)

A fragment takes the one placement that covers all its cells. If several placements fit, the one
that explains the most existing cells wins, and a remaining tie leaves the grid unanswered.
Everything about the motif is read from the grid. Only the spec is fitted.
"""
import time
from itertools import product
from . import perception as P

T8 = {
    "id": lambda r, c: (r, c), "fh": lambda r, c: (r, -c), "fv": lambda r, c: (-r, c), "r180": lambda r, c: (-r, -c),
    "tr": lambda r, c: (c, r), "r90": lambda r, c: (c, -r), "r270": lambda r, c: (-c, r), "atr": lambda r, c: (-c, -r)}
TSETS = {"id": ("id",), "flips": ("id", "fh", "fv", "r180"), "d8": tuple(T8)}
# OPT178 specs first (unchanged behaviour), then the OPT179 extensions
_BASE = [s + (clip, False) for s in product(("largest", "all"), ("id", "flips", "d8"), (8, 4), (False, True),
                                               ((1,), (1, 2, 3, 4))) for clip in (False, True)]
_EXT = [s + (clip, True) for s in product(("largest", "all"), ("id", "flips", "d8"), (8, 4), (False, True),
                                             ((1,), (1, 2, 3, 4))) for clip in (False, True)]
_SCATTER = [(ex, t, "scatter", erase, (1,), clip, rc) for ex in ("big", "largest") for t in ("d8", "id", "flips")
            for erase in (True, False) for clip in (False, True) for rc in (False, True)]
_NORM = [("largest_norm", t, g, e, (1, 2, 3, 4), clip, rc) for t in ("d8", "flips", "id") for g in (8, 4)
         for e in (False, True) for clip in (False, True) for rc in (True, False)]
SPECS = tuple(_BASE + _SCATTER + _EXT + _NORM)


def _native_scale(g, cells):
    """Largest k such that the object is made of aligned k x k uniform blocks."""
    cs = set(cells)
    r0 = min(r for r, _ in cs); c0 = min(c for _, c in cs)
    r1 = max(r for r, _ in cs); c1 = max(c for _, c in cs)
    for k in (4, 3, 2):
        if (r1 - r0 + 1) % k or (c1 - c0 + 1) % k:
            continue
        good = True
        for br in range(r0, r1 + 1, k):
            for bc in range(c0, c1 + 1, k):
                vals = {(g[r][c] if (r, c) in cs else None) for r in range(br, br + k) for c in range(bc, bc + k)}
                if len(vals) != 1:
                    good = False; break
            if not good:
                break
        if good:
            return k
    return 1


def _motif(g, cells, tname, k):
    f = T8[tname]
    pts = []
    for r, c in cells:
        a, b = f(r, c)
        pts += [((a * k + i, b * k + j), g[r][c]) for i in range(k) for j in range(k)]
    r0 = min(p[0] for p, _ in pts); c0 = min(p[1] for p, _ in pts)
    return frozenset(((r - r0, c - c0), v) for (r, c), v in pts)


def _place(g, bg, h, w, md, dr, dc, cmap, clip):
    cells = []
    for (r, c), v in md.items():
        r2, c2 = r + dr, c + dc
        if not (0 <= r2 < h and 0 <= c2 < w):
            if clip:
                continue
            return None
        v2 = cmap.get(v, v)
        if g[r2][c2] != bg and g[r2][c2] != v2:
            return None
        cells.append(((r2, c2), v2))
    return frozenset(cells)


def _colour_map(md, dr, dc, tcols, recolour):
    """Motif colour -> grid colour implied by the fragment cells, or None if inconsistent."""
    m, inv = {}, {}
    for (r, c), col in tcols.items():
        u = md.get((r - dr, c - dc))
        if u is None:
            return None
        if m.setdefault(u, col) != col or inv.setdefault(col, u) != u:
            return None
    if any(u != v for u, v in m.items()):
        if not recolour or not any(u == v for u, v in m.items()):
            return None
    return {u: v for u, v in m.items() if u != v}


def _exemplars(g, bg, exsel):
    multi = [o for o in P.objects(g, bg=bg, conn=8, by_colour=False) if len(P.object_colours(g, o)) >= 2]
    if not multi:
        return None
    if exsel in ("largest", "largest_norm"):
        size = (lambda o: len(o) // _native_scale(g, o) ** 2) if exsel == "largest_norm" else len
        big = max(size(o) for o in multi)
        ex = [o for o in multi if size(o) == big]
        return ex if len(ex) == 1 else None
    if exsel == "big":
        big = max(len(o) for o in multi)
        return [o for o in multi if len(o) >= 3 and 2 * len(o) >= big] or None
    return multi


def _scatter(g, bg, ex, tset, erase, clip, recolour):
    h, w = len(g), len(g[0])
    excells = set().union(*map(set, ex))
    rest = {(r, c): g[r][c] for r in range(h) for c in range(w) if g[r][c] != bg and (r, c) not in excells}
    if not rest or len(rest) > 60:
        return None
    base = [[bg if (erase and (r, c) in excells) else g[r][c] for c in range(w)] for r in range(h)]
    motifs = [dict(m) for m in {_motif(g, o, t, 1) for o in ex for t in TSETS[tset]}]
    cands = {}
    for md in motifs:
        for (r1, c1), v1 in rest.items():
            for (mr, mc), u in md.items():
                if u != v1 and not recolour:
                    continue
                dr, dc = r1 - mr, c1 - mc
                cov = {q: col for q, col in rest.items() if (q[0] - dr, q[1] - dc) in md}
                if len(cov) < 2:
                    continue
                cmap = _colour_map(md, dr, dc, cov, recolour)
                if cmap is None:
                    continue
                pl = _place(base, bg, h, w, md, dr, dc, cmap, clip)
                if pl is None or any(rest.get(q, v) != v for q, v in pl if q in rest):
                    continue
                cands[pl] = frozenset(cov)
    if not cands:
        return None
    left = set(rest); out = [row[:] for row in base]; used = 0
    while left:
        best = max((len(cv & left) for cv in cands.values()), default=0)
        if best < 2:
            return None
        tops = {pl for pl, cv in cands.items() if len(cv & left) == best}
        if len({frozenset(cands[pl] & left) for pl in tops}) != len(tops) or len(tops) > 1:
            covers = {frozenset(cands[pl] & left) for pl in tops}
            if len(covers) == len(tops) and all(not (a & b) for a in covers for b in covers if a != b):
                pass  # disjoint best copies: take them all
            else:
                return None
        for pl in tops:
            for (r, c), v in pl:
                out[r][c] = v
            left -= cands[pl]
            used += 1
    return tuple(map(tuple, out)) if used else None


def apply_spec(g, spec):
    exsel, tset, grouping, erase, scales = spec[:5]
    clip = len(spec) > 5 and spec[5]
    recolour = len(spec) > 6 and spec[6]
    h, w = len(g), len(g[0])
    bg = P.background(g)
    ex = _exemplars(g, bg, exsel)
    if not ex:
        return None
    if grouping == "scatter":
        return _scatter(g, bg, ex, tset, erase, clip, recolour)
    excells = set().union(*map(set, ex))
    rest = [(r, c) for r in range(h) for c in range(w) if g[r][c] != bg and (r, c) not in excells]
    targets = P.cell_components(rest, grouping)
    if not targets or len(targets) > 40:
        return None
    # each motif keeps the rank of the simplest transform producing it (tie-break: least transformed)
    by_k = []
    for k in scales:
        ranked = {}
        for i, t in enumerate(TSETS[tset]):
            for o in ex:
                ranked.setdefault(_motif(g, o, t, k), i)
        by_k.append((k, [(dict(m), i) for m, i in ranked.items()]))
    out = [list(r) for r in g]
    done = 0
    for tg in targets:
        tcols = {(r, c): g[r][c] for r, c in tg}
        r1, c1 = tg[0]
        placements = {}
        for k, motifs in by_k:
            for md, rank in motifs:
                if len(md) <= len(tg):
                    continue
                for (mr, mc), v in md.items():
                    if v != tcols[(r1, c1)] and not recolour:
                        continue
                    dr, dc = r1 - mr, c1 - mc
                    cmap = _colour_map(md, dr, dc, tcols, recolour)
                    if cmap is None:
                        continue
                    pl = _place(g, bg, h, w, md, dr, dc, cmap, clip)
                    if pl is not None:
                        placements[pl] = min(rank, placements.get(pl, rank))
            if placements:
                break
        if len(placements) > 1:
            score = {pl: sum(1 for (r, c), v in pl if g[r][c] == v) for pl in placements}
            top = max(score.values())
            placements = {pl: rk for pl, rk in placements.items() if score[pl] == top}
        if len(placements) > 1:
            low = min(placements.values())
            placements = {pl: rk for pl, rk in placements.items() if rk == low}
        if len(placements) == 1:
            for (r, c), v in next(iter(placements)):
                out[r][c] = v
            done += 1
        elif len(placements) > 1:
            return None
        elif erase:
            for r, c in tg:
                out[r][c] = bg
    return tuple(map(tuple, out)) if done else None


def _plausible(xs, ys):
    """Same shapes; no cell changes from one non-background colour to another... except where an
    exemplar may move away (scatter + erase): then cleared cells are allowed too."""
    added = False
    for x, y in zip(xs, ys):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return False
        bg = P.background(x)
        for rx, ry in zip(x, y):
            for a, b in zip(rx, ry):
                if a != b:
                    if a != bg and b != bg:
                        return False
                    added |= a == bg
    return added


def fit(xs, ys, seconds=2.0):
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if not _plausible(xs, ys):
        return []
    end = time.time() + seconds
    out = []
    for spec in SPECS:
        if time.time() > end:
            break
        try:
            if all(apply_spec(x, spec) == y for x, y in zip(xs, ys)):
                out.append(spec)
        except (IndexError, ValueError, KeyError):
            continue
        if len(out) >= 4:
            break
    return out
