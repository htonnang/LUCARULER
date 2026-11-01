"""LUCA DSL - copy a template onto marker cells (ConceptARC Copy; OPT184).

The template is the largest object (multicolour, 8-connected). Markers are isolated single cells.
Every marker receives a copy of the template, positioned by an anchor that is fitted on the
demonstrations:
    anchor 'centre'    : template centre on the marker (odd height and width)
           'cell:r,c'  : template cell (r, c) of the bounding box on the marker
    paint  'own'       : the copy keeps the template's colours
           'marker'    : the copy is painted in the marker's colour
    keep   True/False  : whether the original template stays
Copies only paint background cells. A copy that would leave the grid is clipped.
"""
from . import perception as P


def _parts(g):
    bg = P.background(g, "zero_if_present")
    objs = P.objects(g, bg=bg, conn=8, by_colour=False)
    if len(objs) < 2:
        return None
    big = max(len(o) for o in objs)
    tmpl = [o for o in objs if len(o) == big]
    if len(tmpl) != 1 or big < 2:
        return None
    markers = [o[0] for o in objs if len(o) == 1]
    if not markers or len(markers) + 1 != len(objs):
        return None
    return bg, tmpl[0], markers


def apply_spec(g, spec):
    anchor, paint, keep = spec
    p = _parts(g)
    if p is None:
        return None
    bg, tmpl, markers = p
    r0, c0, r1, c1 = P.bbox(tmpl)
    if anchor == "centre":
        if (r1 - r0) % 2 or (c1 - c0) % 2:
            return None
        ar, ac = (r1 - r0) // 2, (c1 - c0) // 2
    else:
        ar, ac = map(int, anchor[5:].split(","))
    h, w = len(g), len(g[0])
    out = [list(r) for r in g]
    if not keep:
        for r, c in tmpl:
            out[r][c] = bg
    for mr, mc in markers:
        mcol = g[mr][mc]
        for r, c in tmpl:
            rr, cc = mr + (r - r0) - ar, mc + (c - c0) - ac
            if 0 <= rr < h and 0 <= cc < w and (out[rr][cc] == bg or (rr, cc) == (mr, mc)):
                out[rr][cc] = mcol if paint == "marker" else g[r][c]
    return tuple(map(tuple, out))


def fit(xs, ys):
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if len(xs) < 2 or any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)):
        return []
    p = _parts(xs[0])
    if p is None:
        return []
    r0, c0, r1, c1 = P.bbox(p[1])
    anchors = ["centre"] + ["cell:%d,%d" % (r, c) for r in range(r1 - r0 + 1) for c in range(c1 - c0 + 1)]
    out = []
    for a in anchors:
        for paint in ("own", "marker"):
            for keep in (True, False):
                s = (a, paint, keep)
                if all(apply_spec(x, s) == y for x, y in zip(xs, ys)):
                    out.append(s)
                    if len(out) >= 2:
                        return out
    return out
