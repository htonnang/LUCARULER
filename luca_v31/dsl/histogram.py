"""LUCA DSL - colour histograms (M8 counting, OPT182 draft).

    spec = (layout, skip, axis, counted)
      layout 'bars' : one bar per colour, colours sorted by cell count (largest first); bar length =
                      count; bars start at the top/left; the rest is background 0
             'list' : the colours sorted by count (largest first), one cell each
      skip   number of most frequent non-background colours left out (0 or 1)
      axis   0: bars are columns / the list is a column;  1: rows
      counted 'ink' (colours other than the background, 0 if present) | 'all' (every colour)
Ties in count make the order undefined, and then the op is inapplicable.
"""
from collections import Counter
from . import perception as P


def apply_spec(g, spec):
    layout, skip, axis, counted = spec
    bg = P.background(g, "zero_if_present") if counted == "ink" else None
    cnt = Counter(v for r in g for v in r if v != bg)
    items = sorted(cnt.items(), key=lambda kv: -kv[1])[skip:]
    if not items or len({n for _, n in items}) != len(items):
        return None
    if layout == "list":
        out = tuple((c,) for c, _ in items)
    else:
        H = items[0][1]
        if H > 30 or len(items) > 30:
            return None
        out = tuple(tuple(c if i < n else 0 for c, n in items) for i in range(H))
    if axis == 1:
        out = tuple(zip(*out))
    return tuple(map(tuple, out))


SPECS = tuple((lay, skip, axis, cnt) for cnt in ("ink", "all") for lay in ("bars", "list") for skip in (0, 1)
              for axis in (0, 1))


def fit(xs, ys):
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if len(xs) < 2:
        return []
    return [s for s in SPECS if all(apply_spec(x, s) == y for x, y in zip(xs, ys))][:2]
