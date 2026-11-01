"""LUCA DSL v1 - canonical perception layer (Phase 2).

One parameterised implementation for each basic perception idea that the solver previously
re-implemented many times. Each docstring names the behaviour class(es) it reproduces exactly
(see docs/PHASE2_perception.md for the full mapping of old helpers to canonical calls).
All functions are pure, accept tuple-or-list grids, and never raise on ordinary grids.
"""
from collections import Counter

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = DIRS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


# ------------------------------------------------------------------ background
def colour_counts(g):
    return Counter(v for r in g for v in r)


def background(g, policy="most_common"):
    """Background colour.
    policy 'most_common'       : most frequent colour; ties -> lowest colour value
    policy 'most_common_first' : most frequent colour; ties -> first encountered in row-major order
    policy 'zero_if_present'   : 0 if the grid contains 0, else most_common
    policy 'border'            : most frequent colour on the outer border; ties -> lowest value
    policy 'unique'            : the most frequent colour if it is unique, else None
    policy 'weighted'          : 0.55 * overall share + 0.45 * border share (+0.05 for colour 0);
                                 ties -> lowest value (the former grid_world.guess_backgrounds)
    """
    if policy == "weighted":
        return backgrounds(g, 1, "weighted")[0]
    if policy == "unique":
        ct = colour_counts(g)
        if not ct:
            return None
        top = max(ct.values())
        tops = [k for k, v in ct.items() if v == top]
        return tops[0] if len(tops) == 1 else None
    if policy == "zero_if_present":
        if any(0 in r for r in g):
            return 0
        policy = "most_common"
    if policy == "border":
        h, w = len(g), len(g[0])
        ring = [g[r][c] for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
        ct = Counter(ring)
        top = max(ct.values())
        return min(k for k, v in ct.items() if v == top)
    ct = colour_counts(g)
    top = max(ct.values())
    if policy == "most_common_first":
        return Counter(v for r in g for v in r).most_common(1)[0][0]
    return min(k for k, v in ct.items() if v == top)


def backgrounds(g, n=2, policy="most_common"):
    """Colours ranked as background candidates. policy 'most_common' (ties -> lowest value) or
    'weighted' (border-weighted score, see background())."""
    ct = colour_counts(g)
    if policy == "weighted":
        h, w = len(g), len(g[0])
        border = Counter(g[r][c] for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1))
        n_all, n_border = max(1, h * w), max(1, 2 * h + 2 * w - 4)
        ranked = sorted(ct, key=lambda k: (-(0.55 * ct[k] / n_all + 0.45 * border[k] / n_border
                                              + (0.05 if k == 0 else 0)), k))
        return ranked[:n] if ranked else [0]
    return [k for k, _ in sorted(ct.items(), key=lambda kv: (-kv[1], kv[0]))][:n]


# ------------------------------------------------------------------ objects
def objects(g, bg=None, conn=4, by_colour=True, include_bg=False):
    """Connected components as a list of sorted (r, c) cell lists, in row-major order of first cell.
    conn       : 4 or 8
    by_colour  : True  -> a component has one colour; False -> any non-bg colours connect (multicolour)
    include_bg : also return background components (by_colour only)
    """
    if bg is None:
        bg = background(g)
    h, w = len(g), len(g[0])
    dirs = DIRS8 if conn == 8 else DIRS4
    seen = set()
    out = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            v = g[r][c]
            if v == bg and not include_bg:
                continue
            comp, stack = [], [(r, c)]
            seen.add((r, c))
            while stack:
                a, b = stack.pop()
                comp.append((a, b))
                for da, db in dirs:
                    x, y = a + da, b + db
                    if 0 <= x < h and 0 <= y < w and (x, y) not in seen:
                        u = g[x][y]
                        if (u == v) if by_colour else (u != bg):
                            seen.add((x, y)); stack.append((x, y))
            out.append(sorted(comp))
    return out


def label_map(g, conn=4):
    """Label every cell with the index of its same-colour component (background included);
    returns (labels, components) where components follow objects(..., include_bg=True) order."""
    comps = objects(g, bg=-1, conn=conn, by_colour=True, include_bg=True)
    labels = [[-1] * len(g[0]) for _ in g]
    for k, cells in enumerate(comps):
        for r, c in cells:
            labels[r][c] = k
    return labels, comps


def cell_components(cells, conn=4):
    """Connected groups within a set of (r, c) cells; groups ordered by their smallest cell,
    cells within a group sorted."""
    cells = set(map(tuple, cells))
    dirs = DIRS8 if conn == 8 else DIRS4
    seen, out = set(), []
    for p0 in sorted(cells):
        if p0 in seen:
            continue
        comp, stack = [], [p0]
        seen.add(p0)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in dirs:
                q = (r + dr, c + dc)
                if q in cells and q not in seen:
                    seen.add(q); stack.append(q)
        out.append(sorted(comp))
    return out


def object_colours(g, cells):
    return Counter(g[r][c] for r, c in cells)


# ------------------------------------------------------------------ boxes
def bbox(cells):
    """(r0, c0, r1, c1) inclusive bounds of a cell collection, or None if empty."""
    cells = list(cells)
    if not cells:
        return None
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def bbox_of(g, pred):
    """Bounds of all cells whose colour satisfies pred."""
    return bbox((r, c) for r, row in enumerate(g) for c, v in enumerate(row) if pred(v))


def content_bbox(g, bg=None):
    """Bounds of all non-background cells."""
    if bg is None:
        bg = background(g)
    return bbox_of(g, lambda v: v != bg)


# ------------------------------------------------------------------ separators / panels
def separator_lines(g):
    """Colour whose full rows and/or columns split the grid (most lines wins): (colour, rows, cols)."""
    h, w = len(g), len(g[0])
    best = None
    for col in sorted({v for r in g for v in r}):
        rs = [r for r in range(h) if all(v == col for v in g[r])]
        cs = [c for c in range(w) if all(g[r][c] == col for r in range(h))]
        if (rs or cs) and len(rs) < h and len(cs) < w:
            if best is None or len(rs) + len(cs) > len(best[1]) + len(best[2]):
                best = (col, rs, cs)
    return best


def runs_between(n, seps):
    out, start = [], None
    seps = set(seps)
    for i in range(n + 1):
        if i == n or i in seps:
            if start is not None:
                out.append((start, i - 1))
            start = None
        elif start is None:
            start = i
    return out


def panels(g):
    """Panels delimited by separator lines: list of rows of (r0, c0, r1, c1) inclusive bounds."""
    s = separator_lines(g)
    if not s:
        return None
    _, rs, cs = s
    rr, cc = runs_between(len(g), rs), runs_between(len(g[0]), cs)
    return [[(r0, c0, r1, c1) for c0, c1 in cc] for r0, r1 in rr]


def crop(g, box):
    r0, c0, r1, c1 = box
    return tuple(tuple(row[c0:c1 + 1]) for row in g[r0:r1 + 1])


# ------------------------------------------------------------------ background policies (OPT199)
def bg_of(g, pol=None):
    """Background under a policy fitted per task: None / 'zip' = 0 if present else most frequent;
    'top' = the most frequent colour; an int = that literal colour."""
    if pol is None or pol == "zip":
        return background(g, "zero_if_present")
    if pol == "top":
        return background(g, "most_common")
    return pol


def bg_policies(xs):
    """Policies worth trying on these demonstration inputs, default first: 'zip', then 'top' when it
    differs on some input."""
    pols = ["zip"]
    if any(bg_of(x, "top") != bg_of(x, "zip") for x in xs):
        pols.append("top")
    return pols
