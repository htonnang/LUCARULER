"""crop_transform.py  (LUCA V31 prototype: crop-and-transform module)
===================================================================
Output = (some region of the input) after (a rotation or flip), optionally
with a fixed colour substitution.

The module searches over
  * selectors: how the region is chosen (largest / smallest / most-coloured
    object, largest object of colour c, bounding box of all cells of colour c,
    optionally shrunk by 0-2 cells to drop a frame),
  * the 8 rotations / flips of the crop,
  * an optional colour map learned from the demonstrations,
and accepts a combination only if it reproduces EVERY demonstration output
exactly. Standard library only.
"""
from __future__ import annotations
import os
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from .grid_world import Grid
from .primitives import Pair

BBox = Tuple[int, int, int, int]  # r0, r1, c0, c1 inclusive


def _bg(g: Grid) -> int:
    return _P.background(g, 'most_common_first')  # Phase 2: canonical perception (was 1 lines)


def _components(g: Grid, bg: int, conn: int, mono: bool, radius: int = 1):
    h, w = len(g), len(g[0])
    seen = [[False] * w for _ in range(h)]
    if radius > 1:
        nbrs = [(a, b) for a in range(-radius, radius + 1) for b in range(-radius, radius + 1) if (a, b) != (0, 0)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn == 8 else [])
    comps = []
    for i in range(h):
        for j in range(w):
            if seen[i][j] or g[i][j] == bg:
                continue
            col = g[i][j]
            stack = [(i, j)]; seen[i][j] = True; cells = []
            while stack:
                a, b = stack.pop(); cells.append((a, b))
                for da, db in nbrs:
                    x, y = a + da, b + db
                    if 0 <= x < h and 0 <= y < w and not seen[x][y] and g[x][y] != bg and (not mono or g[x][y] == col):
                        seen[x][y] = True; stack.append((x, y))
            comps.append(cells)
    return comps


def _touches_edge(cells, g) -> bool:
    h, w = len(g), len(g[0])
    return any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)


def _bbox(cells) -> BBox:
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _area(b: BBox) -> int:
    return (b[1] - b[0] + 1) * (b[3] - b[2] + 1)


def _pick(comps, key, g) -> Optional[BBox]:
    if not comps:
        return None
    best = max(comps, key=key)
    return _bbox(best)


def _selectors() -> List[Tuple[str, Callable[[Grid], Optional[BBox]]]]:
    sel: List[Tuple[str, Callable[[Grid], Optional[BBox]]]] = []
    for conn in (8, 4):
        for mono in (False, True):
            tag = f"{'mono' if mono else 'multi'}{conn}"
            sel.append((f"largest_{tag}", lambda g, c=conn, m=mono: _pick(_components(g, _bg(g), c, m), lambda k: len(k), g)))
            sel.append((f"largest_bbox_{tag}", lambda g, c=conn, m=mono: _pick(_components(g, _bg(g), c, m), lambda k: _area(_bbox(k)), g)))
            sel.append((f"smallest_{tag}", lambda g, c=conn, m=mono: _pick(_components(g, _bg(g), c, m), lambda k: -len(k), g)))
    for rad in (2, 3):
        sel.append((f"largest_multi_gap{rad}", lambda g, r=rad: _pick(_components(g, _bg(g), 8, False, r), lambda k: len(k), g)))
        sel.append((f"largest_bbox_multi_gap{rad}", lambda g, r=rad: _pick(_components(g, _bg(g), 8, False, r), lambda k: _area(_bbox(k)), g)))
    for conn in (4, 8):
        sel.append((f"largest_bbox_mono{conn}_interior", lambda g, c=conn: _pick(
            [k for k in _components(g, _bg(g), c, True) if not _touches_edge(k, g)], lambda k: _area(_bbox(k)), g)))
    sel.append(("most_colours_multi8", lambda g: _pick(_components(g, _bg(g), 8, False),
                                                       lambda k: (len({g[r][c] for r, c in k}), len(k)), g)))
    for col in range(10):
        sel.append((f"largest_of_colour_{col}", lambda g, col=col: _largest_of_colour(g, col)))
        sel.append((f"all_of_colour_{col}", lambda g, col=col: _all_of_colour(g, col)))
    return sel


def _largest_of_colour(g: Grid, col: int) -> Optional[BBox]:
    h, w = len(g), len(g[0])
    seen = [[False] * w for _ in range(h)]
    best = None
    for i in range(h):
        for j in range(w):
            if seen[i][j] or g[i][j] != col:
                continue
            stack = [(i, j)]; seen[i][j] = True; cells = []
            while stack:
                a, b = stack.pop(); cells.append((a, b))
                for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    x, y = a + da, b + db
                    if 0 <= x < h and 0 <= y < w and not seen[x][y] and g[x][y] == col:
                        seen[x][y] = True; stack.append((x, y))
            if best is None or len(cells) > len(best):
                best = cells
    return _bbox(best) if best else None


def _all_of_colour(g: Grid, col: int) -> Optional[BBox]:
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == col]
    return _bbox(cells) if cells else None


def _crop(g: Grid, b: BBox, shrink: int) -> Optional[Grid]:
    r0, r1, c0, c1 = b[0] + shrink, b[1] - shrink, b[2] + shrink, b[3] - shrink
    if r1 < r0 or c1 < c0:
        return None
    return tuple(tuple(g[r][c0:c1 + 1]) for r in range(r0, r1 + 1))


def _dihedral(g: Grid, k: int) -> Grid:
    """k in 0..7: identity, rot90, rot180, rot270, flipLR, flipUD, transpose, anti-transpose."""
    h, w = len(g), len(g[0])
    if k == 0:
        return g
    if k == 1:
        return tuple(tuple(g[h - 1 - r][c] for r in range(h)) for c in range(w))
    if k == 2:
        return tuple(tuple(g[h - 1 - r][w - 1 - c] for c in range(w)) for r in range(h))
    if k == 3:
        return tuple(tuple(g[r][w - 1 - c] for r in range(h)) for c in range(w))
    if k == 4:
        return tuple(tuple(row[::-1]) for row in g)
    if k == 5:
        return tuple(tuple(row) for row in g[::-1])
    if k == 6:
        return tuple(tuple(g[r][c] for r in range(h)) for c in range(w))
    return tuple(tuple(g[h - 1 - r][w - 1 - c] for r in range(h)) for c in range(w))


DIHEDRAL_NAMES = ["identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose",
                  "rotate_each_colour_cw_by_outside_count", "rotate_each_colour_ccw_by_outside_count", "orient_to_match_border_colours"]


def _stray_rotate(g: Grid, b: BBox, crop: Grid, clockwise: bool) -> Optional[Grid]:
    """Each colour class of a square crop turns by (number of cells of that colour OUTSIDE the crop)
    quarter turns about the crop centre. Cells of colours with no outside cells stay put."""
    n = len(crop)
    if len(crop[0]) != n:
        return None
    bg = _bg(g)
    stray = Counter(g[r][c] for r in range(len(g)) for c in range(len(g[0]))
                    if g[r][c] != bg and not (b[0] <= r <= b[1] and b[2] <= c <= b[3]))
    out = [[bg] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            v = crop[r][c]
            if v == bg:
                continue
            r2, c2 = r, c
            for _ in range(stray.get(v, 0) % 4):
                r2, c2 = (c2, n - 1 - r2) if clockwise else (n - 1 - c2, r2)
            if out[r2][c2] not in (bg, v):
                return None
            out[r2][c2] = v
    return tuple(tuple(row) for row in out)


def _dom(cells, bg):
    cnt = Counter(v for v in cells if v != bg)
    return cnt.most_common(1)[0][0] if cnt else None


_PROBE = ((0, 1, 0), (2, 9, 3), (0, 4, 0))   # labels: 1=top 2=left 3=right 4=bottom (mid-side probes)


def _side_map(d: int):
    """Where each original side lands after dihedral d: {'top':..,'left':..,'right':..,'bottom':..}."""
    t = _dihedral(_PROBE, d)
    pos = {1: 'top', 2: 'left', 3: 'right', 4: 'bottom'}
    at = {(0, 1): 'top', (1, 0): 'left', (1, 2): 'right', (2, 1): 'bottom'}
    out = {}
    for r in range(3):
        for c in range(3):
            if t[r][c] in pos:
                out[pos[t[r][c]]] = at[(r, c)]
    return out


def _orient(g: Grid, b: BBox, crop: Grid) -> Optional[Grid]:
    """Turn the crop so the coloured lines beside it match the same-coloured lines on the canvas border."""
    h, w = len(g), len(g[0]); bg = _bg(g)
    r0, r1, c0, c1 = b
    adj = {'top': _dom([g[r0 - 1][c] for c in range(c0, c1 + 1)], bg) if r0 > 0 else None,
           'bottom': _dom([g[r1 + 1][c] for c in range(c0, c1 + 1)], bg) if r1 < h - 1 else None,
           'left': _dom([g[r][c0 - 1] for r in range(r0, r1 + 1)], bg) if c0 > 0 else None,
           'right': _dom([g[r][c1 + 1] for r in range(r0, r1 + 1)], bg) if c1 < w - 1 else None}
    canvas = {'top': _dom(g[0], bg), 'bottom': _dom(g[h - 1], bg),
              'left': _dom([g[r][0] for r in range(h)], bg), 'right': _dom([g[r][w - 1] for r in range(h)], bg)}
    scores = []
    for d in range(8):
        m = _side_map(d)
        scores.append(sum(1 for side, col in adj.items() if col is not None and canvas[m[side]] == col))
    best = max(scores)
    if best < 2 or scores.count(best) != 1:
        return None
    return _dihedral(crop, scores.index(best))


def _apply_one(g: Grid, sel: Callable, shrink: int, k: int) -> Optional[Grid]:
    b = sel(g)
    if b is None:
        return None
    c = _crop(g, b, shrink)
    if c is None:
        return None
    if k == 10:
        return _orient(g, (b[0] + shrink, b[1] - shrink, b[2] + shrink, b[3] - shrink), c)
    if k >= 8:
        return _stray_rotate(g, (b[0] + shrink, b[1] - shrink, b[2] + shrink, b[3] - shrink), c, clockwise=(k == 8))
    return _dihedral(c, k)


def induce_all(train_pairs: List[Pair], limit: int = 40):
    """Every (selector, shrink, transform, colour-map) that reproduces all demonstrations exactly,
    simplest first."""
    sels = _selectors()
    found = []
    for shrink in (0, 1, 2):
        for name, sel in sels:
            crops = []
            for x, y in train_pairs:
                b = sel(x)
                c = _crop(x, b, shrink) if b is not None else None
                if c is None:
                    crops = None
                    break
                crops.append((b, c))
            if crops is None:
                continue
            for k in range(11):
                outs = []
                for (x, y), (b, c) in zip(train_pairs, crops):
                    if k < 8:
                        o = _dihedral(c, k)
                    else:
                        if (len(c), len(c[0])) != (len(y), len(y[0])) and k in (8, 9) or (k == 10 and sorted((len(c), len(c[0]))) != sorted((len(y), len(y[0])))):
                            o = None
                        else:
                            bb = (b[0] + shrink, b[1] - shrink, b[2] + shrink, b[3] - shrink)
                            o = _stray_rotate(x, bb, c, clockwise=(k == 8)) if k in (8, 9) else _orient(x, bb, c)
                    if o is None or len(o) != len(y) or len(o[0]) != len(y[0]):
                        outs = None
                        break
                    outs.append(o)
                if outs is None:
                    continue
                if all(o == y for o, (_, y) in zip(outs, train_pairs)):
                    found.append((name, shrink, k, None))
                else:
                    cmap: Dict[int, int] = {}
                    ok = True
                    for o, (_, y) in zip(outs, train_pairs):
                        for ro, ry in zip(o, y):
                            for a_, b_ in zip(ro, ry):
                                if cmap.setdefault(a_, b_) != b_:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok and any(a_ != b_ for a_, b_ in cmap.items()):
                        found.append((name, shrink, k, tuple(sorted(cmap.items()))))
                if len(found) >= limit:
                    return found
    return found


def induce(train_pairs: List[Pair]):
    allspecs = induce_all(train_pairs, limit=1)
    return allspecs[0] if allspecs else None


def predictions(train_pairs: List[Pair], g: Grid):
    """Distinct plausible predictions for g from all consistent specs, best first.
    Plausible = each output dimension lies within the range seen in the demonstrations."""
    specs = induce_all(train_pairs)
    if not specs:
        return []
    hs = [len(y) for _, y in train_pairs]; ws = [len(y[0]) for _, y in train_pairs]
    out = []
    for sp in specs:
        try:
            p = apply_spec(g, sp)
        except Exception:
            p = None
        if p is None or p == g:
            continue
        if all(p != q for q, _ in out):
            out.append((p, sp))
    plaus = lambda p: min(hs) <= len(p) <= max(hs) and min(ws) <= len(p[0]) <= max(ws)
    return [t for t in out if plaus(t[0])] + [t for t in out if not plaus(t[0])]


def apply_spec(g: Grid, spec) -> Optional[Grid]:
    name, shrink, k, cmap = spec
    sel = dict(_selectors())[name]
    out = _apply_one(g, sel, shrink, k)
    if out is None:
        return None
    if cmap:
        m = dict(cmap)
        out = tuple(tuple(m.get(v, v) for v in row) for row in out)
    return out


# ---------------------------------------------------------------- integration stage
from dataclasses import replace
from .primitives import Law
from .dsl import perception as _P  # Phase 2 canonical perception layer

ENABLED = os.environ.get("LUCA_CROPTRANSFORM", "1") != "0"


def upgrade_many(train_pairs: List[Pair], test_inputs: List[Grid], results: list) -> list:
    """Only touches pure fallbacks. The two best plausible crop-and-transform predictions become
    attempts 1 and 2 (any earlier guess is kept only if a slot is left)."""
    if not ENABLED or not any(r.failure_category is not None for r in results):
        return results
    upgraded = []
    for g, r in zip(test_inputs, results):
        if r.failure_category is None:
            upgraded.append(r)
            continue
        try:
            cands = predictions(train_pairs, g)
        except Exception:
            cands = []
        if not cands:
            upgraded.append(r)
            continue
        def mk(sp):
            return Law(name="crop_transform", predict=lambda x, sp=sp: apply_spec(x, sp) or x,
                       description=f"CROP-TRANSFORM({sp[0]}, shrink={sp[1]}, {DIHEDRAL_NAMES[sp[2]]}"
                                   f"{', colour-map' if sp[3] else ''}) verified on all demonstrations",
                       constant_bits=6, op_count=2)
        preds = [p for p, _ in cands[:2]]; laws = [mk(sp) for _, sp in cands[:2]]
        for k, p in enumerate(r.predictions):
            if p not in preds and len(preds) < 2:
                preds.append(p)
                if k < len(r.laws):
                    laws.append(r.laws[k])
        upgraded.append(replace(r, predictions=preds[:2], laws=laws[:2], failure_category=None,
                                explanation=r.explanation + " | crop-transform stage supplied attempts."))
    return upgraded
