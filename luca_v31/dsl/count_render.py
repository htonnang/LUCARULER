"""LUCA DSL - count something, then draw the number (M8 counting, OPT180 draft).

    spec = (counter, offset, layout, colour)
      counter : what is counted in the input
                'objects4' / 'objects8' (non-background components), 'cells' (non-background cells),
                'colours' (distinct non-background colours)
      offset  : n = count + offset (offset in -1..1)
      layout  : 'row'  1 x n           'col'  n x 1           'diag'  n x n with the diagonal inked
                'square' n x n inked   ('order', (H, W), cells): an H x W canvas whose first n cells
                (in the fitted order `cells`) are inked
      colour  : ink colour; 'object' = the input's (single) non-background colour; or a fixed colour
                learned from the demonstrations. The canvas is the background (0 if present).
"""
from collections import Counter
from . import perception as P
from .evidence import REFIT

COUNTERS = ("objects4", "objects8", "cells", "colours")


def count(g, counter):
    bg = P.background(g, "zero_if_present")
    if isinstance(counter, tuple):                 # ('objects_of', colour): objects of one colour
        return len([o for o in P.objects(g, bg=bg, conn=8, by_colour=True) if g[o[0][0]][o[0][1]] == counter[1]])
    if counter == "shapes":                        # distinct object shapes
        sh = set()
        for o in P.objects(g, bg=bg, conn=8, by_colour=True):
            r0, c0, _, _ = P.bbox(o)
            sh.add(tuple((r - r0, c - c0) for r, c in o))
        return len(sh)
    if counter == "cells":
        return sum(1 for r in g for v in r if v != bg)
    if counter == "colours":
        return len({v for r in g for v in r} - {bg})
    conn = 4 if counter == "objects4" else 8
    return len(P.objects(g, bg=bg, conn=conn, by_colour=False))


def _ink(g, colour):
    bg = P.background(g, "zero_if_present")
    if colour == "object":
        cs = {v for r in g for v in r} - {bg}
        return (cs.pop() if len(cs) == 1 else None), bg
    return colour, bg


def render(g, spec):
    counter, offset, layout, colour = spec
    n = count(g, counter) + offset
    ink, bg = _ink(g, colour)
    if ink is None or n < 0 or n > 30:
        return None
    if layout == "row":
        return ((ink,) * n,) if n else None
    if layout == "col":
        return tuple((ink,) for _ in range(n)) if n else None
    if layout == "diag":
        return tuple(tuple(ink if i == j else bg for j in range(n)) for i in range(n)) if n else None
    if layout == "square":
        return tuple((ink,) * n for _ in range(n)) if n else None
    if isinstance(layout, tuple) and layout[0] == "order":
        (H, W), cells = layout[1], layout[2]
        if n > len(cells):
            return None
        out = [[bg] * W for _ in range(H)]
        for r, c in cells[:n]:
            out[r][c] = ink
        return tuple(map(tuple, out))
    return None


def _order_layout(xs, ys, counter, offset, colour):
    """Fixed-size canvas whose inked cells for n are a prefix of one global cell order."""
    shapes = {(len(y), len(y[0])) for y in ys}
    if len(shapes) != 1:
        return None
    H, W = shapes.pop()
    sets = []
    for x, y in zip(xs, ys):
        ink, bg = _ink(x, colour)
        if ink is None:
            return None
        cells = {(r, c) for r in range(H) for c in range(W) if y[r][c] != bg}
        if any(y[r][c] != ink for r, c in cells):
            return None
        n = count(x, counter) + offset
        if len(cells) != n:
            return None
        sets.append(cells)
    sets.sort(key=len)
    order = []
    for s in sets:
        if len(s) == len(order):
            if s != set(order):
                return None
            continue
        if not set(order) <= s:
            return None
        new = sorted(s - set(order))
        if len(new) > 1:
            return None  # every inked cell's rank must be observed, not guessed
        order += new
    return ("order", (H, W), tuple(order))


def fit(xs, ys):
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if len(xs) < 2:
        return []
    colours = ["object"] + sorted({v for y in ys for r in y for v in r})[:3]
    out = []
    common = sorted(set.intersection(*({v for r in x for v in r} for x in xs)) - {P.background(xs[0], "zero_if_present")})
    for counter in COUNTERS + ("shapes",) + tuple(("objects_of", c) for c in common):
        ns = [count(x, counter) for x in xs]
        if len(set(ns)) < 2:
            continue  # the count must actually vary, or it explains nothing
        for offset in (0, 1, -1):
            for colour in colours:
                for layout in ("row", "col", "diag", "square"):
                    spec = (counter, offset, layout, colour)
                    if all(render(x, spec) == y for x, y in zip(xs, ys)):
                        out.append(spec)
                lay = _order_layout(xs, ys, counter, offset, colour)
                if lay:
                    spec = (counter, offset, lay, colour)
                    if all(render(x, spec) == y for x, y in zip(xs, ys)):
                        out.append(spec)
        if out and not REFIT[0]:
            break   # OPT188: inside a leave-one-out refit all variants are kept so the refit re-finds its spec
    return out[:8] if REFIT[0] else out[:4]

apply_spec = render
