"""hole_decorate.py  (LUCA V31 prototype: outline objects, fill their enclosed holes)
====================================================================================
Found by tracing a task cell-by-cell: every object gets an outline ring
painted in one learned colour (the background cells 8-adjacent to the
object), and if the object encloses a hole (a background region that
cannot reach the grid border without crossing the object), that hole is
additionally filled with a second learned colour. Verified across all 9
objects in 3 demonstration pairs with zero exceptions: ring colour is
always the same value, hole-fill colour is always the same value, and
objects without a hole are left with just the outline.

This is the primitive identified as missing while individually inspecting
a task (MCTS plateaued at 0.757 average cell match specifically because
no existing family could express "mark shapes that contain a hole").

Standard library only.
"""
from __future__ import annotations
import os
from collections import Counter
from dataclasses import replace
from typing import List, Optional, Set, Tuple

from .grid_world import Grid
from .primitives import Pair, Law
from .dsl import perception as _P  # Phase 2 canonical perception layer

Cell = Tuple[int, int]


def _bg(g: Grid) -> int:
    return _P.background(g, 'most_common_first')  # Phase 2: canonical perception (was 1 lines)


def _components(g: Grid, bg: int):
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
                        if (da or db) and 0 <= x < h and 0 <= y < w and not seen[x][y] and g[x][y] == col:
                            seen[x][y] = True; stack.append((x, y))
            out.append((col, cells))
    return out


def _bbox(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _hole_and_ring(g: Grid, bg: int, cells) -> Tuple[Set[Cell], Set[Cell]]:
    """(hole cells, ring cells) for one object: the hole is background enclosed strictly
    inside the object's bounding box that cannot reach the box's border without crossing
    the object; the ring is every background cell 8-adjacent to the object, excluding the hole."""
    h, w = len(g), len(g[0])
    cs = set(cells)
    r0, r1, c0, c1 = _bbox(cells)
    border = [(r, c) for r in range(r0 - 1, r1 + 2) for c in range(c0 - 1, c1 + 2)
              if r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1)]
    reach = set(border)
    stack = list(border)
    while stack:
        a, b = stack.pop()
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = a + da, b + db
            if r0 - 1 <= x <= r1 + 1 and c0 - 1 <= y <= c1 + 1 and (x, y) not in reach and (x, y) not in cs:
                reach.add((x, y)); stack.append((x, y))
    hole = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if (r, c) not in cs and (r, c) not in reach and g[r][c] == bg}
    ring = set()
    for (r, c) in cells:
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                if not (da or db):
                    continue
                rr, cc = r + da, c + db
                if 0 <= rr < h and 0 <= cc < w and g[rr][cc] == bg:
                    ring.add((rr, cc))
    ring -= hole
    return hole, ring


def induce(train_pairs: List[Pair]):
    """Learns, by reading what every object's ring/hole/BODY became in the demonstrations:
      ring_colour        -- applied to every object's outline, regardless of hole status
      body_recolour       -- objects WITH a hole: what their own cells become (None = unchanged)
      hole_colour         -- objects WITH a hole: what the enclosed hole becomes
    Objects without a hole keep their original body colour and only gain the ring.
    Accepted only if applying this single learned rule reproduces every demonstration
    output exactly."""
    if not all(len(x) == len(y) and len(x[0]) == len(y[0]) for x, y in train_pairs):
        return None
    ring_votes: Counter = Counter()
    hole_votes: Counter = Counter()
    body_votes: Counter = Counter()
    any_hole = False
    for x, y in train_pairs:
        bg = _bg(x)
        for col, cells in _components(x, bg):
            hole, ring = _hole_and_ring(x, bg, cells)
            for (r, c) in ring:
                if y[r][c] != bg:
                    ring_votes[y[r][c]] += 1
            if hole:
                any_hole = True
                for (r, c) in hole:
                    if y[r][c] != bg:
                        hole_votes[y[r][c]] += 1
                for (r, c) in cells:
                    body_votes[y[r][c]] += 1
    ring_colour = ring_votes.most_common(1)[0][0] if ring_votes else None
    hole_colour = hole_votes.most_common(1)[0][0] if (any_hole and hole_votes) else None
    body_recolour = None
    if any_hole and body_votes:
        top = body_votes.most_common(1)[0][0]
        body_recolour = top  # None handled below if it never actually differs from original
    if ring_colour is None and hole_colour is None and body_recolour is None:
        return None
    for spec in [(ring_colour, hole_colour, body_recolour), (ring_colour, hole_colour, None)]:
        if all(apply_spec(x, spec) == y for x, y in train_pairs):
            return spec
    return None


def apply_spec(g: Grid, spec) -> Optional[Grid]:
    ring_colour, hole_colour, body_recolour = spec
    bg = _bg(g)
    out = [list(r) for r in g]
    changed = False
    for col, cells in _components(g, bg):
        hole, ring = _hole_and_ring(g, bg, cells)
        if ring_colour is not None:
            for (r, c) in ring:
                if out[r][c] == bg:
                    out[r][c] = ring_colour; changed = True
        if hole:
            if hole_colour is not None:
                for (r, c) in hole:
                    if out[r][c] == bg:
                        out[r][c] = hole_colour; changed = True
            if body_recolour is not None:
                for (r, c) in cells:
                    out[r][c] = body_recolour; changed = True
    return tuple(tuple(r) for r in out) if changed else None


# ---------------------------------------------------------------- integration stage
ENABLED = os.environ.get("LUCA_HOLEDECORATE", "1") != "0"


def upgrade_many(train_pairs: List[Pair], test_inputs: List[Grid], results: list) -> list:
    if not ENABLED or not any(r.failure_category is not None for r in results):
        return results
    try:
        spec = induce(train_pairs)
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
        law = Law(name="hole_decorate", predict=lambda x, sp=spec: apply_spec(x, sp) or x,
                  description=f"HOLE-DECORATE(ring={spec[0]}, hole_fill={spec[1]}, body={spec[2]}) verified on all demonstrations",
                  constant_bits=6, op_count=3)
        preds, laws = [pred], [law]
        if r.predictions and r.predictions[0] != pred:
            preds.append(r.predictions[0])
            if r.laws:
                laws.append(r.laws[0])
        upgraded.append(replace(r, predictions=preds[:2], laws=laws[:2], failure_category=None,
                                explanation=r.explanation + " | hole-decorate stage supplied attempt 1."))
    return upgraded
