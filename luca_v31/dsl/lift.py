"""LUCA DSL - lifted programs (OPT200): compose the language's own operations per region.

    per_region(spec)   spec = (view, bg, mode, data)
      view  ('obj', conn, multi)  objects (conn 4/8, one colour or multicolour), each isolated
                                  in its bounding box (other objects' cells read as background)
            ('panels',)           the panels between uniform separator lines
      bg    background value (resolved per grid like every 'bg' parameter)
      mode  'one'    data = program            the same sub-program for every region
            'class'  data = (prop, table)      regions grouped by a property; table maps each
                                               changing value to its own sub-program; regions
                                               of any other value stay unchanged

The sub-programs are ordinary DSL programs. They are found by running the DSL search itself on
the (region input, region output) pairs collected from every demonstration, so new programs are
composed from existing operations instead of being written as new operations.
A region's result is written back only on its own cells and the background cells of its box.
"""
import time
from collections import Counter, defaultdict
from . import perception as P

INNER = [False]          # True while a sub-program search runs (no nested lifting)
SECONDS = [1.5]          # time for one fitting call (all views)
PROPS = ("colour", "colours", "size", "is_largest", "is_smallest", "touches_border", "height", "width", "has_hole",
         "n_colours", "shape_count")


def _shape(g):
    return len(g), len(g[0])


def regions(g, view, bg):
    """List of (cells, mask, box) per region; mask = cells the region may write."""
    H, W = _shape(g)
    if view[0] == "panels":
        sep = P.separator_lines(g)
        if not sep:
            return None
        col, rs, cs = sep
        out = []
        for r0, r1 in P.runs_between(H, rs):
            for c0, c1 in P.runs_between(W, cs):
                cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
                out.append((cells, set(cells), (r0, c0, r1, c1)))
        return out if len(out) >= 2 else None
    _, conn, multi = view
    objs = P.objects(g, bg=bg, conn=conn, by_colour=not multi)
    if not objs or len(objs) > 30:
        return None
    owner = {}
    for i, o in enumerate(objs):
        for rc in o:
            owner[rc] = i
    out = []
    for i, o in enumerate(objs):
        r0, c0, r1, c1 = P.bbox(o)
        mask = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if owner.get((r, c), i) == i}
        out.append((o, mask, (r0, c0, r1, c1)))
    return out


def _crop(g, mask, box, bg):
    r0, c0, r1, c1 = box
    return tuple(tuple(g[r][c] if (r, c) in mask else bg for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))


def _prop(g, cells, box, objs_sizes, all_shapes):
    r0, c0, r1, c1 = box
    H, W = _shape(g)
    cols = Counter(g[r][c] for r, c in cells)
    size = len(cells)
    key = tuple(sorted((r - r0, c - c0) for r, c in cells))
    return dict(colour=cols.most_common(1)[0][0], colours=tuple(sorted(cols)), size=size, is_largest=int(size == max(objs_sizes)),
                is_smallest=int(size == min(objs_sizes)),
                touches_border=int(r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1),
                height=r1 - r0 + 1, width=c1 - c0 + 1, has_hole=int((r1 - r0 + 1) * (c1 - c0 + 1) > size),
                n_colours=len(cols), shape_count=min(all_shapes[key], 2))


def _features(g, regs):
    sizes = [len(c) for c, _, _ in regs]
    shapes = Counter(tuple(sorted((r - b[0], c - b[1]) for r, c in cells)) for cells, _, b in regs)
    return [_prop(g, cells, box, sizes, shapes) for cells, _, box in regs]


HOLE = 10   # placeholder colour bound per region to its context colour


def ctx_colour(g, regs, k, role, bg):
    """Context colour of region k: 'touch' = the one foreign colour 8-adjacent to the region;
    'nearest' = the colour of the nearest other region (unique)."""
    H, W = _shape(g)
    cells = set(regs[k][0])
    own = {g[r][c] for r, c in cells}
    if role == "touch":
        seen = {g[r + a][c + b] for r, c in cells for a in (-1, 0, 1) for b in (-1, 0, 1)
                if 0 <= r + a < H and 0 <= c + b < W and (r + a, c + b) not in cells} - {bg} - own
        return seen.pop() if len(seen) == 1 else None
    best, cols = None, set()
    for j, (oc, _, _) in enumerate(regs):
        if j == k:
            continue
        d = min(max(abs(r - a), abs(c - b)) for r, c in cells for a, b in oc)
        col = {g[a][b] for a, b in oc}
        if len(col) != 1:
            continue
        if best is None or d < best:
            best, cols = d, set(col)
        elif d == best:
            cols |= col
    return cols.pop() if len(cols) == 1 and not (cols & own) else None


def _bind(grid, c):
    return tuple(tuple(HOLE if v == c else v for v in r) for r in grid)


def _unbind(grid, c):
    return tuple(tuple(c if v == HOLE else v for v in r) for r in grid)


def apply_spec(g, spec):
    from .search import run
    from .ops import resolve_bg
    view, bgv, mode, data = spec
    bg = resolve_bg(bgv, g)
    regs = regions(g, view, bg)
    if not regs:
        return None
    feats = _features(g, regs) if mode == "class" else None
    out = [list(r) for r in g]
    did = False
    for k, (cells, mask, box) in enumerate(regs):
        if mode == "bind":
            role, prog = data
            c = ctx_colour(g, regs, k, role, bg)
            if c is None:
                continue
            res = run(prog, _bind(_crop(g, mask, box, bg), c))
            r0, c0, r1, c1 = box
            if res is None or _shape(res) != (r1 - r0 + 1, c1 - c0 + 1):
                return None
            res = _unbind(res, c)
            for r, cc in mask:
                v = res[r - r0][cc - c0]
                if v == HOLE:
                    return None
                if out[r][cc] != v:
                    out[r][cc] = v; did = True
            continue
        if mode == "one":
            prog = data
        else:
            prop, table = data
            t = dict(table)
            prog = t.get(feats[k][prop], ())   # a class never seen changing stays unchanged
        if not prog:
            continue
        res = run(prog, _crop(g, mask, box, bg))
        r0, c0, r1, c1 = box
        if res is None or _shape(res) != (r1 - r0 + 1, c1 - c0 + 1):
            return None
        for r, c in mask:
            v = res[r - r0][c - c0]
            if out[r][c] != v:
                out[r][c] = v; did = True
    return tuple(map(tuple, out)) if did else None


def _inner_fit(pairs, deadline, depth):
    from .search import fit
    if time.time() > deadline:
        return []
    if all(a == b for a, b in pairs):
        return [()]
    INNER[0] = True
    try:
        return fit(pairs, deadline, max_depth=depth)
    except Exception:
        return []
    finally:
        INNER[0] = False


def fit(xs, ys, bgv):
    """Specs reproducing every demonstration (each re-checked on the full grids)."""
    from .ops import resolve_bg
    from .evidence import enough
    if INNER[0] or any(_shape(x) != _shape(y) for x, y in zip(xs, ys)) or all(x == y for x, y in zip(xs, ys)):
        return []
    deadline = time.time() + SECONDS[0]
    out = []
    for view in (("obj", 8, True), ("obj", 4, True), ("obj", 4, False), ("obj", 8, False), ("panels",)):
        if time.time() > deadline:
            break
        per, ok = [], True
        for x, y in zip(xs, ys):
            bg = resolve_bg(bgv, x)
            regs = regions(x, view, bg)
            if not regs:
                ok = False; break
            covered = set().union(*(m for _, m, _ in regs))
            if any(x[r][c] != y[r][c] for r in range(len(x)) for c in range(len(x[0])) if (r, c) not in covered):
                ok = False; break   # every change lies inside some region
            per.append((x, y, bg, regs))
        if not ok or all(len(regs) == 1 for _, _, _, regs in per):
            continue   # one region per grid is the whole-grid search again
        items = []   # (pair, features, demo index)
        ctxs = {"touch": [], "nearest": []}
        for i, (x, y, bg, regs) in enumerate(per):
            for k, ((cells, mask, box), f) in enumerate(zip(regs, _features(x, regs))):
                items.append(((_crop(x, mask, box, bg), _crop(y, mask, box, bg)), f, i))
                for role in ctxs:
                    ctxs[role].append(ctx_colour(x, regs, k, role, bg))
        if not any(a != b for (a, b), _, _ in items):
            continue
        found = []
        # one sub-program for every region
        n_changed = sum(a != b for (a, b), _, _ in items)
        for prog in _inner_fit([p for p, _, _ in items], deadline, 2)[:3]:
            if prog and n_changed >= len(prog) + 1:   # evidence: a k-step program needs k + 1 changed regions
                found.append((view, bgv, "one", prog))
        # one sub-program per class of regions
        if not found:
            for prop in PROPS:
                if time.time() > deadline:
                    break
                groups = defaultdict(list)
                for p, f, i in items:
                    groups[f[prop]].append((p, i))
                if len(groups) < 2:
                    continue
                changing = {v: g for v, g in groups.items() if any(a != b for (a, b), _ in g)}
                if not changing or len(changing) > 3 or len(changing) == len(groups) or \
                        not all(enough({i for _, i in g}, len(g)) for g in changing.values()):
                    continue
                table, good = [], True
                for v, g in sorted(changing.items(), key=lambda kv: repr(kv[0])):
                    nch = sum(a != b for (a, b), _ in g)
                    progs = [q for q in _inner_fit([p for p, _ in g], deadline, 2) if q and nch >= len(q) + 1]
                    if not progs:
                        good = False; break
                    table.append((v, progs[0]))
                if good and any(p for _, p in table):
                    found.append((view, bgv, "class", (prop, tuple(table))))
                    break
        # the regions take a colour from their context (bound per region)
        if not found:
            for role, cs in ctxs.items():
                if time.time() > deadline:
                    break
                bound, good = [], True
                for ((a, b), _, _), c in zip(items, cs):
                    if c is None:
                        if a != b:
                            good = False; break
                        continue
                    bound.append((_bind(a, c), _bind(b, c)))
                changed = sum(a != b for a, b in bound)
                if not good or changed < 2 or not any(HOLE in r for _, b in bound for r in b):
                    continue
                for prog in _inner_fit(bound, deadline, 2)[:2]:
                    if prog and changed >= len(prog) + 1:
                        found.append((view, bgv, "bind", (role, prog)))
                if found:
                    break
        for spec in found:
            if all(apply_spec(x, spec) == y for x, y in zip(xs, ys)):
                out.append(spec)
        if out:
            break
    return out[:3]
