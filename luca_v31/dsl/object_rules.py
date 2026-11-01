"""LUCA DSL - object rules: one property of each object decides what happens to it (OPT183).

Built for the ConceptARC concept groups that are 'a property of each object, then an action':
filled / not filled, horizontal / vertical, inside / outside, above / below, top / bottom,
same / different, order.

    spec = (view, prop, action, table)
      view    (conn, multi): objects are conn-connected (4 / 8), one-colour (multi False) or
              multicolour (multi True)
      prop    a property name from PROPS (values are small ints / strings / shapes)
      action  'recolour' : object repainted with table[value] (a colour); multicolour objects
                           become one colour
              'delete'   : table[value] in {0, 1}; objects with 1 are erased to background
      table   ((value, result), ...)

The table is learned: consistent over all objects in all demonstrations, and every entry seen in
>= 2 demonstrations (evidence rule), unless it is the only value the property takes. An object
whose value is not in the table makes the op inapplicable at test time; nothing is guessed.
"""
from collections import Counter, defaultdict
from . import perception as P
from .evidence import REFIT

PROPS = ("is_hollow", "orientation", "inside_other", "contains_other", "side_of_line", "is_topmost",
         "is_bottommost", "is_leftmost", "is_rightmost", "shape_is_unique", "colour_is_unique",
         "size_rank", "size_rank_rev", "is_largest", "is_smallest", "touches_border", "holes_count",
         "colour", "size", "n_colours", "is_line", "is_rectangle", "is_symmetric", "colour_is_majority")


def _bg(g):
    return P.background(g, "zero_if_present")


def _enclosed_bg(g, cells, bg):
    """Background cells inside the object's box that cannot reach the box edge through non-object cells."""
    cs = set(cells)
    r0, c0, r1, c1 = P.bbox(cells)
    free = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in cs}
    stack = [q for q in free if q[0] in (r0, r1) or q[1] in (c0, c1)]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for dr, dc in P.DIRS4:
            q = (r + dr, c + dc)
            if q in free and q not in seen:
                seen.add(q); stack.append(q)
    return free - seen


def _divider(g, bg):
    """A full-width row or full-height column of one non-background colour: ('row', i) / ('col', j)."""
    h, w = len(g), len(g[0])
    rows = [i for i in range(h) if len(set(g[i])) == 1 and g[i][0] != bg]
    cols = [j for j in range(w) if len({g[i][j] for i in range(h)}) == 1 and g[0][j] != bg]
    if len(rows) == 1 and not cols:
        return ("row", rows[0])
    if len(cols) == 1 and not rows:
        return ("col", cols[0])
    return None


def objects_and_props(g, view):
    conn, multi = view
    bg = _bg(g)
    objs = P.objects(g, bg=bg, conn=conn, by_colour=not multi)
    div = _divider(g, bg)
    if div:
        line = {(div[1], c) for c in range(len(g[0]))} if div[0] == "row" else {(r, div[1]) for r in range(len(g))}
        objs = [o for o in objs if not set(o) <= line]
    if not objs or len(objs) > 60:
        return None, None
    boxes = [P.bbox(o) for o in objs]
    sizes = sorted({len(o) for o in objs})
    shapes = Counter(tuple(sorted((r - b[0], c - b[1]) for r, c in o)) for o, b in zip(objs, boxes))
    cols = Counter(frozenset(g[r][c] for r, c in o) for o in objs)
    ink = Counter(v for row in g for v in row if v != bg).most_common()
    major = ink[0][0] if ink and (len(ink) == 1 or ink[0][1] != ink[1][1]) else None
    h, w = len(g), len(g[0])
    props = []
    for o, b in zip(objs, boxes):
        r0, c0, r1, c1 = b
        oh, ow = r1 - r0 + 1, c1 - c0 + 1
        shape = tuple(sorted((r - r0, c - c0) for r, c in o))
        enc = _enclosed_bg(g, o, bg)
        inside = any(B != b and B[0] <= r0 and B[1] <= c0 and B[2] >= r1 and B[3] >= c1 for B in boxes)
        contains = any(B != b and r0 <= B[0] and c0 <= B[1] and r1 >= B[2] and c1 >= B[3] for B in boxes)
        if div is None:
            side = "none"
        elif div[0] == "row":
            side = "above" if r1 < div[1] else "below" if r0 > div[1] else "across"
        else:
            side = "left" if c1 < div[1] else "right" if c0 > div[1] else "across"
        cset = {g[r][c] for r, c in o}
        sset = set(shape)
        props.append(dict(
            is_hollow=int(bool(enc)), holes_count=len(P.cell_components(enc, 4)) if enc else 0,
            orientation="h" if ow > oh else "v" if oh > ow else "sq", inside_other=int(inside),
            contains_other=int(contains), side_of_line=side,
            is_topmost=int(r0 == min(B[0] for B in boxes)), is_bottommost=int(r1 == max(B[2] for B in boxes)),
            is_leftmost=int(c0 == min(B[1] for B in boxes)), is_rightmost=int(c1 == max(B[3] for B in boxes)),
            shape_is_unique=int(shapes[shape] == 1), colour_is_unique=int(cols[frozenset(cset)] == 1),
            size_rank=sizes[::-1].index(len(o)), size_rank_rev=sizes.index(len(o)),
            is_largest=int(len(o) == sizes[-1]), is_smallest=int(len(o) == sizes[0]),
            touches_border=int(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1),
            colour=min(cset) if len(cset) == 1 else -1, size=len(o), n_colours=len(cset),
            is_line=int((oh == 1 or ow == 1) and len(o) >= 2),
            is_rectangle=int(len(o) == oh * ow), colour_is_majority=int(cset == {major}), is_symmetric=int(sset == {(r, ow - 1 - c) for r, c in sset}
                                                                  or sset == {(oh - 1 - r, c) for r, c in sset})))
    return objs, props


def apply_spec(g, spec):
    if spec[0] == "extract":
        return extract_spec(g, spec)
    view, prop, action, table = spec
    t = dict(table)
    objs, props = objects_and_props(g, view)
    if objs is None:
        return None
    bg = _bg(g)
    out = [list(r) for r in g]
    for o, pr in zip(objs, props):
        v = pr[prop]
        if v not in t:
            return None
        if action == "recolour":
            for r, c in o:
                out[r][c] = t[v]
        elif t[v]:
            for r, c in o:
                out[r][c] = bg
    return tuple(map(tuple, out))


def fit(xs, ys):
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if len(xs) >= 2 and all(len(y) * len(y[0]) < len(x) * len(x[0]) for x, y in zip(xs, ys)):
        return fit_extract(xs, ys)
    if len(xs) < 2 or any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)) or \
            all(x == y for x, y in zip(xs, ys)):
        return []
    # which actions are possible: recolour keeps the ink mask, delete only erases
    recolour_ok = delete_ok = True
    for x, y in zip(xs, ys):
        bg = _bg(x)
        for rx, ry in zip(x, y):
            for a, b in zip(rx, ry):
                if a != b:
                    if a == bg or b == bg:
                        recolour_ok = False
                    if b != bg:
                        delete_ok = False
    if not (recolour_ok or delete_ok):
        return []
    out = []
    for view in ((8, False), (4, False), (8, True), (4, True)):
        per = [(x, y) + objects_and_props(x, view) for x, y in zip(xs, ys)]
        if any(p[2] is None for p in per):
            continue
        for action in (("recolour",) if recolour_ok else ()) + (("delete",) if delete_ok else ()):
            for prop in PROPS:
                table, seen, ok = {}, defaultdict(set), True
                occ = Counter()
                for i, (x, y, objs, props) in enumerate(per):
                    bg = _bg(x)
                    for o, pr in zip(objs, props):
                        news = {y[r][c] for r, c in o}
                        if action == "recolour":
                            if len(news) != 1:
                                ok = False; break
                            res = news.pop()
                        else:
                            if news == {bg}:
                                res = 1
                            elif all(y[r][c] == x[r][c] for r, c in o):
                                res = 0
                            else:
                                ok = False; break
                        if table.setdefault(pr[prop], res) != res:
                            ok = False; break
                        seen[pr[prop]].add(i); occ[pr[prop]] += 1
                    if not ok:
                        break
                if not ok or len(table) < 2 or len(set(table.values())) < 2:
                    continue  # the property must take several values AND matter
                from .evidence import enough
                if not all(enough(s, occ[k]) for k, s in seen.items()):
                    continue
                spec = (view, prop, action, tuple(sorted(table.items(), key=repr)))
                if all(apply_spec(x, spec) == y for x, y in zip(xs, ys)):
                    out.append(spec)
                    if len(out) >= (12 if REFIT[0] else 3):
                        return out
        if out and not REFIT[0]:
            return out   # OPT188: inside a leave-one-out refit all views' variants are kept
    return out


# ------------------------------------------------------------------ extract (ConceptARC ExtractObjects)
def _crop(g, cells, mode):
    r0, c0, r1, c1 = P.bbox(cells)
    if mode == "box":
        return tuple(tuple(g[r][c0:c1 + 1]) for r in range(r0, r1 + 1))
    bg = _bg(g)
    cs = set(cells)
    return tuple(tuple(g[r][c] if (r, c) in cs else bg for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))


def extract_spec(g, spec):
    """spec = ('extract', view, prop, value, crop_mode): crop the ONE object whose property == value."""
    _, view, prop, value, mode = spec
    objs, props = objects_and_props(g, view)
    if objs is None:
        return None
    hits = [o for o, pr in zip(objs, props) if pr[prop] == value]
    return _crop(g, hits[0], mode) if len(hits) == 1 else None


def fit_extract(xs, ys):
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if len(xs) < 2 or all(len(x) == len(y) and len(x[0]) == len(y[0]) for x, y in zip(xs, ys)):
        return []
    out = []
    for view in ((8, False), (8, True), (4, False), (4, True)):
        for mode in ("box", "cells"):
            chosen = []
            for x, y in zip(xs, ys):
                objs, props = objects_and_props(x, view)
                if objs is None:
                    break
                m = [pr for o, pr in zip(objs, props) if _crop(x, o, mode) == y]
                if len(m) != 1:
                    break
                chosen.append((props, m[0]))
            if len(chosen) != len(xs):
                continue
            for prop in PROPS:
                vals = {pr[prop] for _, pr in chosen}
                if len(vals) != 1:
                    continue
                v = vals.pop()
                # the value must single the object out in every demonstration
                if all(sum(1 for q in props if q[prop] == v) == 1 for props, _ in chosen):
                    if prop in ("colour", "size"):
                        continue  # a literal colour / size would just name the answer
                    out.append(("extract", view, prop, v, mode))
            if out and not REFIT[0]:
                return out[:3]   # OPT188: inside a leave-one-out refit all variants are kept
    return out[:12]
