"""LUCA DSL - layers: which shape is on top (ConceptARC TopBottom3D; OPT184).

Each non-background colour is one shape. Its full form is assumed to be a filled rectangle or a
rectangle outline (form chosen by the fitter), and the stacking order is inferred from occlusion:
A is under B when cells of A's full form are covered by B.

    spec = (form, action, arg)
      form    'filled' | 'outline'
      action  'top_colour'    : output = an arg[0] x arg[1] grid of the top shape's colour
              'bottom_colour' : same with the bottom shape
              'reverse'       : redraw all shapes, completed, in reversed stacking order
              'raise_bottom'  : draw the bottom shape, completed, on top of everything
The occlusion relation must be acyclic. Shapes that do not overlap may be drawn in any order.
'top_colour' / 'bottom_colour' need a unique top / bottom shape.
"""
from . import perception as P


def _shapes(g, form):
    bg = P.background(g, "zero_if_present")
    cols = sorted({v for r in g for v in r} - {bg})
    if len(cols) < 2 or len(cols) > 6:
        return None
    full = {}
    for col in cols:
        cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == col]
        r0, c0, r1, c1 = P.bbox(cells)
        if form == "filled":
            full[col] = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        else:
            full[col] = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                         if r in (r0, r1) or c in (c0, c1)}
    # every visible cell must belong to its colour's full form
    for col in cols:
        if any((r, c) not in full[col] for r, row in enumerate(g) for c, v in enumerate(row) if v == col):
            return None
    return bg, cols, full


def _order(g, bg, cols, full):
    """Bottom-to-top order from occlusion, or None if the occlusion relation is not a chain."""
    above = {c: set() for c in cols}           # above[a] = colours lying on a
    for a in cols:
        for (r, c) in full[a]:
            v = g[r][c]
            if v != a and v != bg and v in full and (r, c) in full[v]:
                above[a].add(v)
            elif v != a and v == bg:
                return None                    # a hole in a full form: the form is wrong
    order = []
    left = set(cols)
    while left:
        bottoms = sorted(c for c in left if not any(c in above[d] for d in left))
        if not bottoms:
            return None                        # cyclic occlusion
        # shapes that do not touch may come in any order; ties broken by colour
        order.append(bottoms[0]); left.remove(bottoms[0])
    for i, a in enumerate(order):              # consistency: nothing below a lies on a
        if any(order.index(b) < i for b in above[a]):
            return None
    return order


def apply_spec(g, spec):
    form, action, arg = spec
    s = _shapes(g, form)
    if s is None:
        return None
    bg, cols, full = s
    order = _order(g, bg, cols, full)
    if order is None:
        return None
    if action in ("top_colour", "bottom_colour"):
        covered = {b for a in cols for b in cols if a != b and any(
            g[r][c] == b for r, c in full[a])}           # shapes lying on something
        under = {a for a in cols if any(g[r][c] != a and g[r][c] != bg for r, c in full[a])}
        tops = [c for c in cols if c not in under and c in covered] or [c for c in cols if c not in under]
        bots = [c for c in cols if c in under and c not in covered]
        pick = tops if action == "top_colour" else bots
        if len(pick) != 1:
            return None
        col = pick[0]
        return tuple((col,) * arg[1] for _ in range(arg[0]))
    out = [list(r) for r in g]
    seq = order[::-1] if action == "reverse" else order + [order[0]]
    for col in seq:
        for r, c in full[col]:
            out[r][c] = col
    return tuple(map(tuple, out))


def fit(xs, ys):
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if len(xs) < 2:
        return []
    shapes = {(len(y), len(y[0])) for y in ys}
    cands = []
    for form in ("filled", "outline"):
        if len(shapes) == 1:
            sz = shapes.pop(); shapes.add(sz)
            cands += [(form, "top_colour", sz), (form, "bottom_colour", sz)]
        cands += [(form, "reverse", ()), (form, "raise_bottom", ())]
    return [s for s in cands if all(apply_spec(x, s) == y for x, y in zip(xs, ys))][:2]
