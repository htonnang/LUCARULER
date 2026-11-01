"""LUCA DSL - recolour every object by one of its properties (OPT182 draft).

    spec = (conn, prop, table)
      conn   4 | 8 (one-colour objects)
      prop   size | size_parity | size_rank | shape | height | width | holes | touches_border |
             is_largest | is_smallest | is_square
      table  ((property value, new colour), ...)

Every object is repainted with table[its property value]; cells outside objects stay. The table is
learned from the demonstrations: consistent everywhere, and every entry seen in >= 2
demonstrations (evidence rule). An object whose value is not in the table makes the op inapplicable.
"""
from collections import defaultdict
from . import perception as P
from .evidence import REFIT

PROPS = ("size", "size_parity", "size_rank", "shape", "height", "width", "holes", "touches_border",
         "is_largest", "is_smallest", "is_square", "size_clamp", "height_clamp", "width_clamp", "size_extreme")
# OPT195: a table value 'BG' erases the object (paints the background); 'size_extreme' is
# 'max' / 'min' / 'mid' (largest, smallest, any other size)
# OPT188 '<prop>_clamp': a numeric property whose table extends past its ends - a value above the
# largest learned value takes that value's colour (and below the smallest, the smallest's). Values
# past an end with the end's colour need no evidence of their own.


def _lookup(t, prop, v):
    if v in t:
        return t[v]
    if prop.endswith("_clamp") and t:
        ks = sorted(t)
        if v > ks[-1]:
            return t[ks[-1]]
        if v < ks[0]:
            return t[ks[0]]
    return None


def _props(g, objs):
    h, w = len(g), len(g[0])
    sizes = sorted({len(o) for o in objs}, reverse=True)
    out = []
    for o in objs:
        r0, c0, r1, c1 = P.bbox(o)
        oh, ow = r1 - r0 + 1, c1 - c0 + 1
        out.append(dict(size=len(o), size_parity=len(o) % 2, size_rank=sizes.index(len(o)),
                        shape=tuple(sorted((r - r0, c - c0) for r, c in o)), height=oh, width=ow,
                        holes=oh * ow - len(o),
                        touches_border=int(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1),
                        is_largest=int(len(o) == sizes[0]), is_smallest=int(len(o) == sizes[-1]),
                        is_square=int(oh == ow and len(o) == oh * ow), size_clamp=len(o), height_clamp=oh,
                        width_clamp=ow, size_extreme="max" if len(o) == sizes[0] else
                        "min" if len(o) == sizes[-1] else "mid"))
    return out


def apply_spec(g, spec):
    conn, prop, table = spec[:3]
    pol = spec[3] if len(spec) > 3 else None
    t = dict(table)
    bg = P.bg_of(g, pol)
    objs = P.objects(g, bg=bg, conn=conn, by_colour=True)
    if not objs or len(objs) > 60:
        return None
    out = [list(r) for r in g]
    for o, pr in zip(objs, _props(g, objs)):
        n = _lookup(t, prop, pr[prop])
        if n is None:
            return None
        for r, c in o:
            out[r][c] = bg if n == "BG" else n
    return tuple(map(tuple, out))


def _with(spec, pol):
    return spec if pol == "zip" else spec + (pol,)


def fit(xs, ys):
    """OPT199: the background is a fitted policy ('zip' = 0 if present, the default; 'top' = most
    frequent colour); the default is tried first and the others only when it finds nothing."""
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if len(xs) < 2 or any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)):
        return []
    pols = P.bg_policies(xs)
    for pol in pols:
        out = _fit_pol(xs, ys, pol)
        if out:
            return out + _alt_top(out, pols)
    return []


def _alt_top(out, pols):
    """The demonstrations cannot tell 'zip' from 'top' (they agree on every input): the 'top' variant
    of the first spec is offered after the default specs (it differs only where a test input has 0
    as a minority colour)."""
    if len(pols) > 1 or not out or len(out[0]) > 3:
        return []
    return [out[0] + ("top",)]


def _fit_pol(xs, ys, pol):
    # cheap gate: only recolours (no cell becomes or stops being background)
    for x, y in zip(xs, ys):
        bg = P.bg_of(x, pol)
        if any(a == bg and b != bg for rx, ry in zip(x, y) for a, b in zip(rx, ry)):
            return []   # recolour or erase only: no background cell gains ink
        if x == y:
            return []
    out = []
    for conn in (8, 4):
        per = [(x, y, P.objects(x, bg=P.bg_of(x, pol), conn=conn, by_colour=True))
               for x, y in zip(xs, ys)]
        for prop in PROPS:
            table, seen, ok = {}, defaultdict(set), True
            occ = defaultdict(int)
            for i, (x, y, objs) in enumerate(per):
                for o, pr in zip(objs, _props(x, objs)):
                    news = {y[r][c] for r, c in o}
                    if len(news) != 1:
                        ok = False; break
                    v, n = pr[prop], news.pop()
                    if n == P.bg_of(x, pol):
                        n = "BG"
                    if table.setdefault(v, n) != n:
                        ok = False; break
                    seen[v].add(i); occ[v] += 1
                if not ok:
                    break
            from .evidence import enough
            if ok and table and prop.endswith("_clamp"):
                ev = sorted(k for k in table if enough(seen[k], occ[k]))
                if not ev:
                    continue
                table = {k: c for k, c in table.items() if ev[0] <= k <= ev[-1] or
                         (k > ev[-1] and c != table[ev[-1]]) or (k < ev[0] and c != table[ev[0]])}
            if not ok or not table or not all(enough(seen[k], occ[k]) for k in table):
                continue
            if len(set(table.values())) < 2 and all(k == v for k, v in table.items()):
                continue
            spec = _with((conn, prop, tuple(sorted(table.items(), key=repr))), pol)
            if all(apply_spec(x, spec) == y for x, y in zip(xs, ys)):
                out.append(spec)
        # OPT188: inside a leave-one-out refit every connectivity's variants are kept, so the refit can
        # re-find the main fit's spec; the main fit keeps its first-found preference (ranking votes)
        if out and not REFIT[0]:
            break
    return out[:8] if REFIT[0] else out[:3]


# ------------------------------------------------------------------ interiors (OPT194)
def _interior(o):
    s = set(o)
    return [(r, c) for r, c in o if all((r + a, c + b) in s for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0)))]


def apply_interior(g, spec):
    """Every object's interior (cells whose 4 neighbours are all in the object) is repainted with
    table[property]; outlines and everything else stay."""
    conn, prop, table = spec[:3]
    pol = spec[3] if len(spec) > 3 else None
    t = dict(table)
    bg = P.bg_of(g, pol)
    objs = P.objects(g, bg=bg, conn=conn, by_colour=True)
    if not objs or len(objs) > 60:
        return None
    out = [list(r) for r in g]
    did = False
    for o, pr in zip(objs, _props(g, objs)):
        inner = _interior(o)
        if not inner:
            continue
        n = _lookup(t, prop, pr[prop])
        if n is None:
            return None
        for r, c in inner:
            out[r][c] = bg if n == "BG" else n
        did = True
    return tuple(map(tuple, out)) if did else None


def fit_interior(xs, ys):
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if len(xs) < 2 or any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)) or \
            all(x == y for x, y in zip(xs, ys)):
        return []
    pols = P.bg_policies(xs)
    for pol in pols:
        out = _fit_interior_pol(xs, ys, pol)
        if out:
            return out + _alt_top(out, pols)
    return []


def _fit_interior_pol(xs, ys, pol):
    out = []
    from .evidence import enough
    for conn in (4, 8):
        per = [(x, y, P.objects(x, bg=P.bg_of(x, pol), conn=conn, by_colour=True))
               for x, y in zip(xs, ys)]
        if any(not objs or len(objs) > 60 for _, _, objs in per):
            continue
        for prop in PROPS:
            table, seen, occ, ok = {}, defaultdict(set), defaultdict(int), True
            for i, (x, y, objs) in enumerate(per):
                for o, pr in zip(objs, _props(x, objs)):
                    inner = _interior(o)
                    rim = set(o) - set(inner)
                    if any(y[r][c] != x[r][c] for r, c in rim):
                        ok = False; break
                    if not inner:
                        continue
                    news = {y[r][c] for r, c in inner}
                    if len(news) != 1:
                        ok = False; break
                    v, n = pr[prop], news.pop()
                    if n == P.bg_of(x, pol):
                        n = "BG"   # OPT199: the background is a role, not a literal
                    if table.setdefault(v, n) != n:
                        ok = False; break
                    seen[v].add(i); occ[v] += 1
                if not ok:
                    break
            if not ok or not table or not all(enough(seen[k], occ[k]) for k in table):
                continue
            spec = _with((conn, prop, tuple(sorted(table.items(), key=repr))), pol)
            if all(apply_interior(x, spec) == y for x, y in zip(xs, ys)):
                out.append(spec)
        if out and not REFIT[0]:
            break
    return out[:8] if REFIT[0] else out[:3]
