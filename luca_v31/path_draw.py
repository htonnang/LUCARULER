"""path_draw.py  (LUCA V31 prototype: path-and-ray drawing module)
================================================================
Same-shape tasks whose output only ADDS coloured cells to background.

Two drawing families with learned parameters:

  cast    : from every source cell (cells of colour c, or all non-background
            cells; optionally only isolated ones) shoot rays in a direction set
            (one of 8 directions, orthogonal, diagonal, all 8, horizontal or
            vertical pair) until the border or the first obstacle (stop) or
            through obstacles (through), painting only background cells.
  connect : join aligned cells of the same colour (row / column / diagonal)
            with a line when only background lies between them.

The paint colour is the source colour, or a source->colour map learned from
the demonstrations. A parameter combination is accepted only if it reproduces
EVERY demonstration output exactly. Standard library only.
"""
from __future__ import annotations
import os
from collections import Counter
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from .grid_world import Grid
from .primitives import Pair, Law
from .dsl import perception as _P  # Phase 2 canonical perception layer

DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1),
        "UL": (-1, -1), "UR": (-1, 1), "DL": (1, -1), "DR": (1, 1)}
DIRSETS = {**{k: [v] for k, v in DIRS.items()},
           "LR": [DIRS["L"], DIRS["R"]], "UD": [DIRS["U"], DIRS["D"]],
           "orth4": [DIRS[k] for k in ("U", "D", "L", "R")], "diag4": [DIRS[k] for k in ("UL", "UR", "DL", "DR")],
           "all8": list(DIRS.values()), "UL_DR": [DIRS["UL"], DIRS["DR"]], "UR_DL": [DIRS["UR"], DIRS["DL"]]}


def _bg(g: Grid) -> int:
    return _P.background(g, 'most_common_first')  # Phase 2: canonical perception (was 1 lines)


def _sources(g: Grid, bg: int, color, srctype: str):
    h, w = len(g), len(g[0])
    cells = [(r, c) for r in range(h) for c in range(w) if g[r][c] != bg and (color == "any" or g[r][c] == color)]
    if srctype == "isolated":
        cs = set(cells)
        cells = [(r, c) for r, c in cells
                 if not any((r + a, c + b) in cs and g[r + a][c + b] == g[r][c] for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0))]
    return cells


def cast(g: Grid, color, srctype: str, dirs: str, stop: str) -> Dict[Tuple[int, int], int]:
    """{painted cell: colour of the source that painted it}; first source in reading order wins."""
    h, w = len(g), len(g[0]); bg = _bg(g); out: Dict[Tuple[int, int], int] = {}
    for r, c in _sources(g, bg, color, srctype):
        for dr, dc in DIRSETS[dirs]:
            x, y = r + dr, c + dc
            while 0 <= x < h and 0 <= y < w:
                if g[x][y] == bg:
                    out.setdefault((x, y), g[r][c])
                elif stop == "obstacle":
                    break
                x += dr; y += dc
    return out


def connect(g: Grid, color, mode: str) -> Dict[Tuple[int, int], int]:
    h, w = len(g), len(g[0]); bg = _bg(g); out: Dict[Tuple[int, int], int] = {}
    cols = [color] if color != "any" else sorted({v for row in g for v in row if v != bg})
    for col in cols:
        cells = [(r, c) for r in range(h) for c in range(w) if g[r][c] == col]
        for i, (r1, c1) in enumerate(cells):
            for (r2, c2) in cells[i + 1:]:
                dr, dc = r2 - r1, c2 - c1
                orth = (dr == 0 or dc == 0); diag = abs(dr) == abs(dc)
                if not ((mode in ("orth", "both") and orth) or (mode in ("diag", "both") and diag)):
                    continue
                n = max(abs(dr), abs(dc)); sr = (dr > 0) - (dr < 0); sc = (dc > 0) - (dc < 0)
                between = [(r1 + sr * k, c1 + sc * k) for k in range(1, n)]
                if between and all(g[a][b] == bg for a, b in between):
                    for a, b in between:
                        out.setdefault((a, b), col)
    return out


def _draw(g: Grid, spec) -> Optional[Grid]:
    family, color, a, b, paint = spec
    painted = cast(g, color, a, b[0], b[1]) if family == "cast" else connect(g, color, a)
    if not painted:
        return None
    pm = dict(paint[1]) if paint[0] == "map" else None
    cur = [list(r) for r in g]
    for (r, c), src in painted.items():
        cur[r][c] = paint[1] if paint[0] == "fixed" else (src if paint[0] == "source" else pm.get(src, src))
    return tuple(tuple(r) for r in cur)


def induce(train_pairs: List[Pair]):
    if not all(len(x) == len(y) and len(x[0]) == len(y[0]) for x, y in train_pairs):
        return None
    added = []
    for x, y in train_pairs:
        bg = _bg(x); d = {}
        for r in range(len(x)):
            for c in range(len(x[0])):
                if x[r][c] != y[r][c]:
                    if x[r][c] != bg:
                        return None            # something other than background was changed
                    d[(r, c)] = y[r][c]
        if not d:
            return None
        added.append(d)
    common = None
    for x, _ in train_pairs:
        bg = _bg(x); s = {v for row in x for v in row if v != bg}
        common = s if common is None else common & s
    colors = sorted(common or []) + ["any"]
    cands = []
    for color in colors:
        for srctype in ("all", "isolated"):
            for dirs in DIRSETS:
                for stop in ("obstacle", "through"):
                    cands.append(("cast", color, srctype, (dirs, stop)))
        for mode in ("orth", "diag", "both"):
            cands.append(("connect", color, mode, None))
    for fam, color, a, b in cands:
        painted = []
        ok = True
        for (x, _), d in zip(train_pairs, added):
            p = cast(x, color, a, b[0], b[1]) if fam == "cast" else connect(x, color, a)
            if set(p) != set(d):
                ok = False
                break
            painted.append(p)
        if not ok:
            continue
        # paint colour: source colour, a fixed colour, or a consistent source->colour map
        if all(p[k] == d[k] for p, d in zip(painted, added) for k in d):
            paint = ("source", None)
        else:
            outs = {v for d in added for v in d.values()}
            if len(outs) == 1:
                paint = ("fixed", next(iter(outs)))
            else:
                m: Dict[int, int] = {}
                for p, d in zip(painted, added):
                    for k in d:
                        if m.setdefault(p[k], d[k]) != d[k]:
                            m = None
                            break
                    if m is None:
                        break
                if m is None:
                    continue
                paint = ("map", tuple(sorted(m.items())))
        return (fam, color, a, b, paint)
    return None


def _step_cells(g: Grid, step) -> Dict[Tuple[int, int], int]:
    fam, color, a, b, paint = step
    return cast(g, color, a, b[0], b[1]) if fam == "cast" else connect(g, color, a)


def induce_compose(train_pairs: List[Pair], max_steps: int = 4):
    """Union of drawing steps, each of which never paints a wrong cell on any demonstration.
    Different colours may follow different rules."""
    if not all(len(x) == len(y) and len(x[0]) == len(y[0]) for x, y in train_pairs):
        return None
    added = []
    for x, y in train_pairs:
        bg = _bg(x); d = {}
        for r in range(len(x)):
            for c in range(len(x[0])):
                if x[r][c] != y[r][c]:
                    if x[r][c] != bg:
                        return None
                    d[(r, c)] = y[r][c]
        if not d:
            return None
        added.append(d)
    common = None
    for x, _ in train_pairs:
        bg = _bg(x); sset = {v for row in x for v in row if v != bg}
        common = sset if common is None else common & sset
    colors = sorted(common or [])
    steps = []
    for color in colors:
        cands = [("cast", color, st, (dn, sp)) for st in ("all", "isolated") for dn in DIRSETS for sp in ("obstacle", "through")]
        cands += [("connect", color, m, None) for m in ("orth", "diag", "both")]
        for fam, col, a_, b_ in cands:
            per = []
            ok = True; nonempty = False
            for (x, _), d in zip(train_pairs, added):
                pnt = cast(x, col, a_, b_[0], b_[1]) if fam == "cast" else connect(x, col, a_)
                if any(k not in d for k in pnt):
                    ok = False
                    break
                nonempty = nonempty or bool(pnt)
                per.append(pnt)
            if not ok or not nonempty:
                continue
            if all(pp[k] == d[k] for pp, d in zip(per, added) for k in pp):
                paint = ("source", None)
            else:
                outs = {d[k] for pp, d in zip(per, added) for k in pp}
                if len(outs) != 1:
                    continue
                paint = ("fixed", next(iter(outs)))
            steps.append(((fam, col, a_, b_, paint), [set(pp) for pp in per]))
    remaining = [set(d) for d in added]
    chosen = []
    for _ in range(max_steps):
        if all(not r for r in remaining):
            break
        best = max(steps, key=lambda st: sum(len(c & r) for c, r in zip(st[1], remaining)), default=None)
        gain = sum(len(c & r) for c, r in zip(best[1], remaining)) if best else 0
        if gain == 0:
            return None
        chosen.append(best[0])
        remaining = [r - c for r, c in zip(remaining, best[1])]
    if any(remaining) or not chosen:
        return None
    spec = ("compose", tuple(chosen))
    return spec if all(apply_spec(x, spec) == y for x, y in train_pairs) else None


def apply_spec(g: Grid, spec) -> Optional[Grid]:
    if spec[0] == "compose":
        cur = [list(r) for r in g]; any_ = False
        for st in spec[1]:
            for (r, c), src in _step_cells(g, st).items():
                cur[r][c] = st[4][1] if st[4][0] == "fixed" else src
                any_ = True
        return tuple(tuple(r) for r in cur) if any_ else None
    return _draw(g, spec)


# ---------------------------------------------------------------- integration stage
ENABLED = os.environ.get("LUCA_PATHDRAW", "1") != "0"


def upgrade_many(train_pairs: List[Pair], test_inputs: List[Grid], results: list) -> list:
    """Only touches pure fallbacks. A verified drawing rule becomes attempt 1."""
    if not ENABLED or not any(r.failure_category is not None for r in results):
        return results
    try:
        spec = induce(train_pairs) or induce_compose(train_pairs)
    except Exception:
        return results
    if spec is None:
        return results
    upgraded = []
    for g, r in zip(test_inputs, results):
        if r.failure_category is None:
            upgraded.append(r)
            continue
        try:
            pred = apply_spec(g, spec)
        except Exception:
            pred = None
        if pred is None or pred == g:
            upgraded.append(r)
            continue
        law = Law(name=f"path_draw:{spec[0]}", predict=lambda x, sp=spec: apply_spec(x, sp) or x,
                  description=f"PATH-DRAW({spec}) verified on all demonstrations", constant_bits=6, op_count=2)
        preds, laws = [pred], [law]
        for k, p in enumerate(r.predictions):
            if p not in preds and len(preds) < 2:
                preds.append(p)
                if k < len(r.laws):
                    laws.append(r.laws[k])
        upgraded.append(replace(r, predictions=preds[:2], laws=laws[:2], failure_category=None,
                                explanation=r.explanation + " | path-draw stage supplied attempt 1."))
    return upgraded
