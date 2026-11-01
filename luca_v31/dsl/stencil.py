"""LUCA DSL - stencils: every source paints its kind's learned neighbourhood pattern (OPT181).

    spec = (sources, radius, table)
      sources 'dots'    isolated single cells; kind = colour
              'anydots' isolated single cells; one kind for all (the dot's colour does not matter, OPT187)
              'all'     every non-background cell; kind = colour
              'objects' 8-connected one-colour objects; kind = (colour, shape); the pattern is placed
                        relative to the object's bounding box
              'anyall'  every non-background cell; one kind for all (OPT188)
              'anyobjects' as 'objects', kind = ('*', shape): the object's colour does not matter (OPT188)
      paint   a colour, 'BG' (the grid's background) or, for 'anydots' / 'anyobjects', 'SELF'
              (the source's own colour, OPT188)
      radius  1 or 2: the window is the source's bounding box grown by `radius`
      table   ((kind, ((dr, dc, paint), ...)), ...) offsets from the bounding box's top-left corner

A window cell joins a kind's pattern only if, at every source of that kind in every demonstration,
it was background (or part of the source itself) and the output shows the same colour there,
different from the input. So a source may also be erased or recoloured. Evidence rule: a kind
whose pattern changes anything must occur in >= 2 demonstrations. An unseen kind at test time
makes the program inapplicable rather than guessed.
"""
from collections import defaultdict
from . import perception as P


def _sources(g, bg, which):
    h, w = len(g), len(g[0])
    if which in ("objects", "anyobjects"):
        out = []
        for cells in P.objects(g, bg=bg, conn=8, by_colour=True):
            r0, c0, _, _ = P.bbox(cells)
            col = "*" if which == "anyobjects" else g[cells[0][0]][cells[0][1]]
            kind = (col, tuple((r - r0, c - c0) for r, c in cells))
            out.append((kind, cells, (r0, c0)))
        return out
    out = []
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg:
                continue
            if which in ("dots", "anydots") and any(0 <= r + a < h and 0 <= c + b < w and g[r + a][c + b] != bg
                                       for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b):
                continue
            out.append(("*" if which in ("anydots", "anyall") else g[r][c], [(r, c)], (r, c)))
    return out


def _clear_bg(g, bg):
    """The background is strictly the most frequent colour (a tie makes the 'BG' / 'SELF' roles ambiguous)."""
    ct = P.colour_counts(g)
    return all(n < ct.get(bg, 0) for k, n in ct.items() if k != bg)


def _window(cells, origin, radius):
    r0, c0 = origin
    r1 = max(r for r, _ in cells); c1 = max(c for _, c in cells)
    for r in range(r0 - radius, r1 + radius + 1):
        for c in range(c0 - radius, c1 + radius + 1):
            yield r, c, r - r0, c - c0


def apply_spec(g, spec):
    which, radius, table = spec
    t = dict(table)
    bg = P.background(g, "zero_if_present")
    h, w = len(g), len(g[0])
    out = [list(r) for r in g]
    if which in ("anydots", "anyobjects", "anyall") and not _clear_bg(g, bg):
        return None
    srcs = _sources(g, bg, which)
    if not srcs:
        return None
    for kind, cells, (r0, c0) in srcs:
        pat = t.get(kind)
        if pat is None:
            return None
        own = set(cells)
        selfc = g[cells[0][0]][cells[0][1]]
        for dr, dc, paint in pat:
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and (g[rr][cc] == bg or (rr, cc) in own):
                out[rr][cc] = bg if paint == "BG" else selfc if paint == "SELF" else paint
    return tuple(map(tuple, out))


def fit(xs, ys):
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    if len(xs) < 2 or any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)):
        return []
    found = []
    for which, radius in (("dots", 1), ("dots", 2), ("objects", 1), ("all", 1), ("anydots", 1), ("anydots", 2),
                          ("anyobjects", 1), ("anyall", 1)):
        anyk = which in ("anydots", "anyobjects", "anyall")
        obs = defaultdict(lambda: defaultdict(set))   # kind -> offset -> output values
        changed = defaultdict(set)                    # kind -> offsets where the output differs
        demos = defaultdict(set)
        occ = defaultdict(int)
        blocked = defaultdict(set)                    # kind -> offsets that were foreign ink
        for i, (x, y) in enumerate(zip(xs, ys)):
            bg = P.background(x, "zero_if_present")
            if anyk and not _clear_bg(x, bg):
                obs = {}; break
            h, w = len(x), len(x[0])
            for kind, cells, origin in _sources(x, bg, which):
                demos[kind].add(i); occ[kind] += 1; own = set(cells)
                selfc = x[cells[0][0]][cells[0][1]]
                for r, c, dr, dc in _window(cells, origin, radius):
                    if not (0 <= r < h and 0 <= c < w):
                        continue
                    if x[r][c] == bg or (r, c) in own:
                        # OPT187: painting the grid's own background is recorded as the role 'BG'
                        # OPT188: with colour-free sources, the source's own colour is the role 'SELF'
                        obs[kind][(dr, dc)].add("BG" if y[r][c] == bg else
                                                "SELF" if anyk and y[r][c] == selfc else y[r][c])
                        if y[r][c] != x[r][c]:
                            changed[kind].add((dr, dc))
                    else:
                        blocked[kind].add((dr, dc))
        if not obs:
            continue
        table, ok = [], True
        for kind, offs in obs.items():
            pat = tuple(sorted((dr, dc, next(iter(v))) for (dr, dc), v in offs.items()
                               if len(v) == 1 and (dr, dc) in changed[kind]))
            from .evidence import enough
            if pat and not enough(demos[kind], occ[kind]):
                ok = False; break
            table.append((kind, pat))
        if not ok or not any(p for _, p in table):
            continue
        spec = (which, radius, tuple(sorted(table, key=repr)))
        if all(apply_spec(x, spec) == y for x, y in zip(xs, ys)):
            found.append(spec)   # OPT187: keep every variant so cross-check refits can re-find the same one
    return found
