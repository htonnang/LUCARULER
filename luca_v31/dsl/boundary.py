"""LUCA DSL - motion and extension to a boundary; clean-up; centres (OPT183, ConceptARC concepts
MoveToBoundary, ExtendToBoundary, CleanUp, Center).

move   spec = ('move', direction, conn, static colours)   conn 4 / 8 one-colour objects, 48 / 88 multicolour
       Every non-static object slides in `direction` until it touches the grid edge or a cell that is
       already occupied (objects nearer the edge settle first, so they stack). Static objects are
       those whose colour never moves in the demonstrations (the fitter finds them). direction:
       'up' / 'down' / 'left' / 'right' / 'nearest_edge' (each object to its own nearest edge) /
       'toward_static' (each object toward the static object it is aligned with).
extend spec = ('extend', mode, conn)
       mode 'axis'     : every straight line object (>= 2 cells) is extended along its axis, both
                         ends, through background until the edge or an obstacle
            'dir:<d>'  : every object casts its colour from its cells in direction d (8 directions)
                         through background until the edge or an obstacle
            'nearest_edge': every object casts toward its own nearest grid edge
clean  spec = ('clean', kind, k)
       kind 'small'  : objects of <= k cells (8-connected, one colour) are erased; k is fitted
                       between the largest erased and the smallest kept object
            'speck'  : a cell whose 4 neighbours (inside the grid) all share one other colour takes
                       that colour (repair of isolated noise inside shapes)
centre spec = ('centre', scope, colour)
       scope 'objects': the centre cell of every object's box with odd height and width is painted
             'grid'   : the centre cell(s) of the grid are painted
"""
from . import perception as P

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
        "ul": (-1, -1), "ur": (-1, 1), "dl": (1, -1), "dr": (1, 1)}


def _bg(g):
    return P.background(g, "zero_if_present")


def _nearest_edge(g, cells):
    h, w = len(g), len(g[0])
    r0, c0, r1, c1 = P.bbox(cells)
    d = {"up": r0, "down": h - 1 - r1, "left": c0, "right": w - 1 - c1}
    best = min(d.values())
    ks = [k for k, v in d.items() if v == best]
    return ks[0] if len(ks) == 1 else None


# ------------------------------------------------------------------ move
def _move(g, direction, conn, static):
    bg = _bg(g)
    h, w = len(g), len(g[0])
    multi = conn in (48, 88)          # 48 / 88: multicolour objects with 4 / 8 connectivity
    objs = P.objects(g, bg=bg, conn=8 if conn in (8, 88) else 4, by_colour=not multi)
    movers, fixed = [], set()
    for o in objs:
        if {g[r][c] for r, c in o} <= static:
            fixed |= set(o)
        else:
            movers.append(o)
    if not movers:
        return None
    plan = []
    for o in movers:
        d = direction
        if direction == "nearest_edge":
            d = _nearest_edge(g, o)
            if d is None:
                return None
        elif direction == "toward_static":
            r0, c0, r1, c1 = P.bbox(o)
            cands = set()
            for (r, c) in fixed:
                if c0 <= c <= c1:
                    cands.add("down" if r > r1 else "up")
                if r0 <= r <= r1:
                    cands.add("right" if c > c1 else "left")
            if len(cands) != 1:
                return None
            d = cands.pop()
        plan.append((o, DIRS[d]))
    occupied = set(fixed)
    out = [[bg] * w for _ in range(h)]
    for r, c in fixed:
        out[r][c] = g[r][c]

    def lead(item):
        o, (dr, dc) = item
        return -max(r * dr + c * dc for r, c in o)
    for o, (dr, dc) in sorted(plan, key=lead):
        k = 0
        while all(0 <= r + (k + 1) * dr < h and 0 <= c + (k + 1) * dc < w and
                  (r + (k + 1) * dr, c + (k + 1) * dc) not in occupied for r, c in o):
            k += 1
        for r, c in o:
            q = (r + k * dr, c + k * dc)
            out[q[0]][q[1]] = g[r][c]
            occupied.add(q)
    return tuple(map(tuple, out))


# ------------------------------------------------------------------ extend
def _cast(g, out, r, c, dr, dc, colour, bg):
    h, w = len(g), len(g[0])
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and g[rr][cc] == bg:
        out[rr][cc] = colour
        rr += dr; cc += dc


def _extend(g, mode, conn):
    bg = _bg(g)
    out = [list(r) for r in g]
    objs = P.objects(g, bg=bg, conn=conn, by_colour=True)
    if not objs:
        return None
    did = False
    for o in objs:
        col = g[o[0][0]][o[0][1]]
        if mode == "axis":
            r0, c0, r1, c1 = P.bbox(o)
            if len(o) < 2 or not (r0 == r1 or c0 == c1) or len(o) != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            if r0 == r1:
                _cast(g, out, r0, c0, 0, -1, col, bg); _cast(g, out, r0, c1, 0, 1, col, bg)
            else:
                _cast(g, out, r0, c0, -1, 0, col, bg); _cast(g, out, r1, c0, 1, 0, col, bg)
            did = True
        elif mode in ("star4", "star8"):            # single cells radiate in 4 / 8 directions
            if len(o) != 1:
                continue
            for d in (list(DIRS)[:4] if mode == "star4" else list(DIRS)):
                _cast(g, out, o[0][0], o[0][1], DIRS[d][0], DIRS[d][1], col, bg)
            did = True
        else:
            d = mode[4:] if mode.startswith("dir:") else _nearest_edge(g, o)
            if d is None:
                return None
            dr, dc = DIRS[d]
            for r, c in o:
                _cast(g, out, r, c, dr, dc, col, bg)
            did = True
    return tuple(map(tuple, out)) if did else None


# ------------------------------------------------------------------ clean
def _clean(g, kind, k):
    bg = _bg(g)
    h, w = len(g), len(g[0])
    out = [list(r) for r in g]
    if kind == "small":
        for o in P.objects(g, bg=bg, conn=8, by_colour=True):
            if len(o) <= k:
                for r, c in o:
                    out[r][c] = bg
    else:
        for r in range(h):
            for c in range(w):
                nb = [g[r + a][c + b] for a, b in P.DIRS4 if 0 <= r + a < h and 0 <= c + b < w]
                if len(nb) >= 3 and len(set(nb)) == 1 and nb[0] != g[r][c]:
                    out[r][c] = nb[0]
    return tuple(map(tuple, out))


# ------------------------------------------------------------------ centre
def _centre(g, scope, colour):
    bg = _bg(g)
    out = [list(r) for r in g]
    h, w = len(g), len(g[0])
    if scope == "grid":
        for r in {(h - 1) // 2, h // 2}:
            for c in {(w - 1) // 2, w // 2}:
                out[r][c] = colour
        return tuple(map(tuple, out))
    did = False
    for o in P.objects(g, bg=bg, conn=8, by_colour=False):
        r0, c0, r1, c1 = P.bbox(o)
        if (r1 - r0) % 2 or (c1 - c0) % 2:
            continue
        cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
        if scope == "cross":                        # full row and column through the centre, on background
            for c in range(w):
                if g[cr][c] == bg:
                    out[cr][c] = colour
            for r in range(h):
                if g[r][cc] == bg:
                    out[r][cc] = colour
        else:
            out[cr][cc] = colour
        did = True
    return tuple(map(tuple, out)) if did else None


def apply_spec(g, spec):
    kind = spec[0]
    if kind == "move":
        return _move(g, spec[1], spec[2], frozenset(spec[3]))
    if kind == "extend":
        return _extend(g, spec[1], spec[2])
    if kind == "clean":
        return _clean(g, spec[1], spec[2])
    if kind == "centre":
        return _centre(g, spec[1], spec[2])
    return None


def fit(xs, ys):
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if len(xs) < 2 or any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)) or \
            all(x == y for x, y in zip(xs, ys)):
        return []
    cands = []
    # static colours: every cell of that colour is identical in input and output, in every demo
    allc = set().union(*({v for r in x for v in r} for x in xs)) - {_bg(xs[0])}
    static = frozenset(c for c in allc if all(
        all((y[r][cc] == c) == (x[r][cc] == c) for r in range(len(x)) for cc in range(len(x[0]))) for x, y in zip(xs, ys)))
    for d in ("down", "up", "left", "right", "nearest_edge", "toward_static"):
        for conn in (8, 4, 88, 48):
            cands.append(("move", d, conn, tuple(sorted(static))))
    for mode in ["axis", "nearest_edge", "star4", "star8"] + ["dir:" + d for d in DIRS]:
        for conn in (8, 4):
            cands.append(("extend", mode, conn))
    # clean 'small': k between largest erased and smallest kept
    lo, hi = 0, 10 ** 6
    for x, y in zip(xs, ys):
        bg = _bg(x)
        for o in P.objects(x, bg=bg, conn=8, by_colour=True):
            if all(y[r][c] == bg for r, c in o):
                lo = max(lo, len(o))
            elif all(y[r][c] == x[r][c] for r, c in o):
                hi = min(hi, len(o))
    if 0 < lo < hi:
        cands.append(("clean", "small", lo))
    cands.append(("clean", "speck", 0))
    for col in sorted({v for y in ys for r in y for v in r}):
        cands.append(("centre", "objects", col)); cands.append(("centre", "grid", col))
        cands.append(("centre", "cross", col))
    out = []
    for s in cands:
        try:
            if all(apply_spec(x, s) == y for x, y in zip(xs, ys)):
                out.append(s)
        except (IndexError, KeyError, ValueError):
            continue
        if len(out) >= 3:
            break
    return out
