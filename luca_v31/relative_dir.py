"""relative_dir.py  (LUCA V31 prototype: object-relative direction primitive)
===========================================================================
A cross-task vocabulary for "move / shoot / draw in a direction that depends
on other objects", instead of one-off modules.

Objects   : connected components of non-background cells (single-colour or
            multi-colour, 8-connected).
Direction : each mover gets a direction from a RULE:
              const D            one of the 8 fixed directions
              toward T / away T  toward / away from the nearest object T
                                 (any, of colour c, or the largest); quantised to the
                                 dominant axis (axis4) or to the sign of both offsets (sign8)
              border             toward the nearest grid border
              facing             from the body of a multi-colour object toward its odd cell
              bycolor            a per-colour direction map (learned)
Action    : how the direction is used
              slide              move until blocked (border / another object)
              slide_trail        move until blocked, painting the swept path
              cast_stop          shoot a ray until blocked
              cast_through       shoot a ray through obstacles, painting only background
Movers    : all objects, objects of one colour, single cells, or all but the largest.

A combination is accepted only if it reproduces EVERY demonstration output
exactly. Standard library only.
"""
from __future__ import annotations
import os
import time
from collections import Counter
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from .grid_world import Grid
from .primitives import Pair, Law
from .dsl import perception as _P  # Phase 2 canonical perception layer

Cell = Tuple[int, int]
Dir = Tuple[int, int]
DIRS8: Dict[str, Dir] = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1),
                         "UL": (-1, -1), "UR": (-1, 1), "DL": (1, -1), "DR": (1, 1)}
AXIS4 = ["U", "D", "L", "R"]


def _bg(g: Grid) -> int:
    return _P.background(g, 'most_common_first')  # Phase 2: canonical perception (was 1 lines)


def _objects(g: Grid, bg: int, mono: bool) -> List[List[Cell]]:
    h, w = len(g), len(g[0]); seen = [[False] * w for _ in range(h)]; out = []
    for i in range(h):
        for j in range(w):
            if seen[i][j] or g[i][j] == bg:
                continue
            col = g[i][j]; stack = [(i, j)]; seen[i][j] = True; cells = []
            while stack:
                a, b = stack.pop(); cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        x, y = a + da, b + db
                        if (da or db) and 0 <= x < h and 0 <= y < w and not seen[x][y] and g[x][y] != bg and (not mono or g[x][y] == col):
                            seen[x][y] = True; stack.append((x, y))
            out.append(cells)
    return out


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def _quant(dy: int, dx: int, q: str) -> Optional[Dir]:
    if dy == 0 and dx == 0:
        return None
    if q == "axis4":
        if abs(dy) > abs(dx):
            return (_sign(dy), 0)
        if abs(dx) > abs(dy):
            return (0, _sign(dx))
        return None
    return (_sign(dy), _sign(dx))


def _nearest(a: List[Cell], b: List[Cell]) -> Tuple[int, Dir]:
    best = None
    for (r1, c1) in a:
        for (r2, c2) in b:
            d = max(abs(r2 - r1), abs(c2 - c1)) * 1000 + abs(r2 - r1) + abs(c2 - c1)
            if best is None or d < best[0]:
                best = (d, (r2 - r1, c2 - c1))
    return best[0], best[1]


class _Scene:
    def __init__(self, g: Grid, mono: bool):
        self.g = g; self.h = len(g); self.w = len(g[0]); self.bg = _bg(g)
        self.objs = _objects(g, self.bg, mono)
        self.col = [Counter(g[r][c] for r, c in o) for o in self.objs]


# ---------------------------------------------------------------- movers
def _movers(sc: _Scene, sel) -> List[int]:
    kind = sel[0]
    idx = range(len(sc.objs))
    if kind == "all":
        return list(idx)
    if kind == "color":
        return [i for i in idx if sc.col[i].most_common(1)[0][0] == sel[1]]
    if kind == "single":
        return [i for i in idx if len(sc.objs[i]) == 1]
    if kind == "not_largest":
        m = max((len(o) for o in sc.objs), default=0)
        return [i for i in idx if len(sc.objs[i]) < m]
    return []


def _targets(sc: _Scene, tsel, exclude: List[int]) -> List[int]:
    idx = [i for i in range(len(sc.objs)) if i not in exclude]
    if tsel[0] == "color":
        return [i for i in idx if sc.col[i].most_common(1)[0][0] == tsel[1]]
    if tsel[0] == "largest":
        if not idx:
            return []
        m = max(len(sc.objs[i]) for i in idx)
        return [i for i in idx if len(sc.objs[i]) == m]
    return idx


# ---------------------------------------------------------------- direction rules
def _direction(sc: _Scene, i: int, rule, movers: List[int]) -> Optional[Dir]:
    kind = rule[0]
    cells = sc.objs[i]
    if kind == "const":
        return DIRS8[rule[1]]
    if kind in ("toward", "away"):
        _, tsel, q = rule
        tg = _targets(sc, tsel, [j for j in movers if j == i] if tsel[0] != "any" else [i])
        tg = [j for j in tg if j != i]
        if not tg:
            return None
        best = None
        for j in tg:
            d, v = _nearest(cells, sc.objs[j])
            if best is None or d < best[0]:
                best = (d, v)
        v = best[1]
        return _quant(-v[0], -v[1], q) if kind == "away" else _quant(v[0], v[1], q)
    if kind == "border":
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        d = {"U": min(rs), "D": sc.h - 1 - max(rs), "L": min(cs), "R": sc.w - 1 - max(cs)}
        m = min(d.values()); ks = [k for k, v in d.items() if v == m]
        return DIRS8[ks[0]] if len(ks) == 1 else None
    if kind == "facing":
        cnt = sc.col[i]
        if len(cnt) < 2:
            return None
        mc = min(cnt.values()); odd = [c for c, v in cnt.items() if v == mc]
        if len(odd) != 1:
            return None
        oc = [(r, c) for r, c in cells if sc.g[r][c] == odd[0]]
        body = [(r, c) for r, c in cells if sc.g[r][c] != odd[0]]
        if len(oc) != 1 or not body:
            return None
        by = sum(r for r, _ in body) / len(body); bx = sum(c for _, c in body) / len(body)
        return _quant(int(round((oc[0][0] - by) * 2)), int(round((oc[0][1] - bx) * 2)), rule[1])
    if kind == "bycolor":
        d = dict(rule[1]).get(sc.col[i].most_common(1)[0][0])
        return DIRS8[d] if d else None
    return None


# ---------------------------------------------------------------- actions
def _run(g: Grid, mono: bool, msel, rule, action, paint) -> Optional[Grid]:
    sc = _Scene(g, mono)
    movers = _movers(sc, msel)
    if not movers:
        return None
    dirs = {i: _direction(sc, i, rule, movers) for i in movers}
    if not any(dirs.values()):
        return None
    h, w, bg = sc.h, sc.w, sc.bg
    cur = [list(r) for r in g]
    order = sorted([i for i in movers if dirs[i]], key=lambda i: -max(dirs[i][0] * r + dirs[i][1] * c for r, c in sc.objs[i]))
    for i in order:
        dr, dc = dirs[i]; cells = sc.objs[i]; cs = set(cells)
        colors = {p: g[p[0]][p[1]] for p in cells}
        if action in ("slide", "slide_trail"):
            for p in cells:
                cur[p[0]][p[1]] = bg
            k = 0
            while True:
                nxt = [(r + dr * (k + 1), c + dc * (k + 1)) for r, c in cells]
                if all(0 <= r < h and 0 <= c < w and cur[r][c] == bg for r, c in nxt):
                    k += 1
                else:
                    break
            if action == "slide_trail":
                for s in range(k):
                    for (r, c) in cells:
                        cur[r + dr * s][c + dc * s] = colors[(r, c)] if paint is None else paint
            for (r, c) in cells:
                cur[r + dr * k][c + dc * k] = colors[(r, c)]
        else:
            front = [p for p in cells if (p[0] + dr, p[1] + dc) not in cs]
            for (r, c) in front:
                x, y = r + dr, c + dc
                while 0 <= x < h and 0 <= y < w:
                    if g[x][y] == bg and cur[x][y] == bg:
                        cur[x][y] = colors[(r, c)] if paint is None else paint
                    elif action == "cast_stop":
                        break
                    x += dr; y += dc
    out = tuple(tuple(r) for r in cur)
    return out if out != g else None


# ---------------------------------------------------------------- induction
def _rules(colors: List[int], mover_colors: List[int]):
    rules = [("const", k) for k in DIRS8]
    for q in ("axis4", "sign8"):
        for tsel in [("any",), ("largest",)] + [("color", c) for c in colors]:
            rules.append(("toward", tsel, q))
        rules.append(("away", ("any",), q))
        rules.append(("facing", q))
    rules.append(("border",))
    return rules


def _consistent(train_pairs: List[Pair], deadline: float, limit: int):
    if not all(len(x) == len(y) and len(x[0]) == len(y[0]) for x, y in train_pairs):
        return []
    info = []
    for x, y in train_pairs:
        b = _bg(x)
        nx = {(r, c): v for r, row in enumerate(x) for c, v in enumerate(row) if v != b}
        ny = {(r, c): v for r, row in enumerate(y) for c, v in enumerate(row) if v != b}
        info.append((len(nx) == len(ny), all(ny.get(k) == v for k, v in nx.items()), {y[r][c] for r in range(len(y)) for c in range(len(y[0])) if x[r][c] != y[r][c]}))
    if all(x == y for x, y in train_pairs):
        return []
    same_count = all(i[0] for i in info); keep_all = all(i[1] for i in info)
    actions = []
    if same_count:
        actions.append(("slide", None))
    added_colors = sorted(set().union(*[i[2] for i in info]))
    if keep_all:
        for a in ("cast_stop", "cast_through"):
            actions.append((a, None))
            for k in added_colors[:3]:
                actions.append((a, k))
    actions.append(("slide_trail", None))
    for k in added_colors[:3]:
        actions.append(("slide_trail", k))
    common = None
    for x, _ in train_pairs:
        s = {v for row in x for v in row if v != _bg(x)}
        common = s if common is None else common & s
    colors = sorted(common or [])
    found = []
    for mono in (True, False):
        msels = [("all",), ("single",), ("not_largest",)] + [("color", c) for c in colors]
        for msel in msels:
            for action, paint in actions:
                rules = _rules(colors, colors)
                # per-colour direction maps (only when a few movers' colours exist)
                if msel == ("all",) and 0 < len(colors) <= 3:
                    import itertools
                    for combo in itertools.product([None] + AXIS4, repeat=len(colors)):
                        if any(combo):
                            rules.append(("bycolor", tuple((c, d) for c, d in zip(colors, combo) if d)))
                for rule in rules:
                    if time.monotonic() > deadline:
                        return found
                    ok = True
                    for x, y in train_pairs:
                        if _run(x, mono, msel, rule, action, paint) != y:
                            ok = False
                            break
                    if ok:
                        found.append((mono, msel, rule, action, paint))
                        if len(found) >= limit:
                            return found
    return found


def induce_all(train_pairs: List[Pair], seconds: float = 6.0, limit: int = 12):
    return _consistent(train_pairs, time.monotonic() + seconds, limit)


def induce(train_pairs: List[Pair]):
    s = induce_all(train_pairs, limit=1)
    return s[0] if s else None


def apply_spec(g: Grid, spec) -> Optional[Grid]:
    return _run(g, *spec)


def predictions(train_pairs: List[Pair], g: Grid):
    out = []
    for sp in induce_all(train_pairs):
        try:
            p = apply_spec(g, sp)
        except Exception:
            p = None
        if p is not None and p != g and all(p != q for q, _ in out):
            out.append((p, sp))
    return out


# ---------------------------------------------------------------- integration stage
ENABLED = os.environ.get("LUCA_RELDIR", "1") != "0"


def upgrade_many(train_pairs: List[Pair], test_inputs: List[Grid], results: list) -> list:
    """OPT174 (DSL port 4): retired as a pipeline stage. The motion grammar now runs as the DSL
    operation move_objects (same fill-only contract); original stage in docs/retired/."""
    return results


def _retired_upgrade_many(train_pairs: List[Pair], test_inputs: List[Grid], results: list) -> list:
    """Only touches pure fallbacks. The two best distinct verified predictions become the attempts."""
    if not ENABLED or not any(r.failure_category is not None for r in results):
        return results
    try:
        specs = induce_all(train_pairs, seconds=4.0, limit=8)
    except Exception:
        return results
    if not specs:
        return results
    upgraded = []
    for g, r in zip(test_inputs, results):
        if r.failure_category is None:
            upgraded.append(r)
            continue
        cands = []
        for sp in specs:
            try:
                p = apply_spec(g, sp)
            except Exception:
                p = None
            if p is not None and p != g and all(p != q for q, _ in cands):
                cands.append((p, sp))
        if not cands:
            upgraded.append(r)
            continue
        mk = lambda sp: Law(name="relative_dir", predict=lambda x, sp=sp: apply_spec(x, sp) or x,
                            description=f"RELATIVE-DIR({sp[1]}, {sp[2]}, {sp[3]}) verified on all demonstrations",
                            constant_bits=8, op_count=2)
        # attempt 1 = best verified candidate; attempt 2 = the earlier first guess (so a correct answer supplied
        # by an earlier stage is never displaced), else the next verified candidate
        preds = [cands[0][0]]; laws = [mk(cands[0][1])]
        if r.predictions and r.predictions[0] not in preds:
            preds.append(r.predictions[0])
            if r.laws:
                laws.append(r.laws[0])
        elif len(cands) > 1:
            preds.append(cands[1][0]); laws.append(mk(cands[1][1]))
        upgraded.append(replace(r, predictions=preds[:2], laws=laws[:2], failure_category=None,
                                explanation=r.explanation + " | relative-direction stage supplied attempts."))
    return upgraded
