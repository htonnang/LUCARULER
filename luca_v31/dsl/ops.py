"""LUCA DSL v1 - operations.

Every operation is a pure function  op(g, ctx, **params) -> Grid | None
  g    : current grid (tuple of tuples of ints)
  ctx  : Context with the original input grid (ctx.input), for operations that read it
  None : the operation does not apply (precondition failed); never raises on bad input.
Parameters are never hard-coded by a program author: each is filled by an inference rule
(see infer.py) from the demonstration pairs. The registry records, per operation, which
inference rule supplies each parameter.
"""
from collections import Counter
from . import perception as P

MAX = 30


class Context:
    __slots__ = ("input",)

    def __init__(self, inp):
        self.input = inp


def G(rows):
    return tuple(tuple(r) for r in rows)


def shape(g):
    return len(g), len(g[0]) if g else 0


def colours(g):
    return {v for r in g for v in r}


def most_common(g):
    return P.background(g, "most_common_first")


def border_colour(g):
    """Most frequent colour on the grid's border (a 1-row / 1-column grid: its two end cells)."""
    H, W = shape(g)
    if H == 1 or W == 1:
        cells = [g[0][0], g[H - 1][W - 1]]
    else:
        cells = [g[r][c] for r in range(H) for c in range(W) if r in (0, H - 1) or c in (0, W - 1)]
    ct = Counter(cells)
    top = max(ct.values())
    return min(k for k, n in ct.items() if n == top)


def resolve_bg(v, g):
    """Background parameter values: 'auto' = the grid's most frequent colour; 'task:c' (OPT199) =
    the colour c every demonstration input shares, kept on a grid where c is the most frequent or
    the border colour, else that grid's most frequent colour; an int = that colour."""
    if v == "auto":
        return most_common(g)
    if isinstance(v, str) and v.startswith("task:"):
        c = int(v[5:])
        return c if c in (most_common(g), border_colour(g)) else most_common(g)
    return v


def ok(g):
    # OPT197: sides up to 100 with at most 2,500 cells (ARC grids are <= 30x30; other ARC-style sets,
    # such as 1D-ARC rows, are longer) - still bounds runaway tiling
    return g is not None and 0 < len(g) <= 100 and 0 < len(g[0]) <= 100 and len(g) * len(g[0]) <= 2500


# ------------------------------------------------------------------ geometry
def rot90(g, ctx): return G(zip(*g[::-1]))
def rot180(g, ctx): return G(r[::-1] for r in g[::-1])
def rot270(g, ctx): return G(zip(*g))[::-1]
def flip_h(g, ctx): return G(r[::-1] for r in g)
def flip_v(g, ctx): return G(g[::-1])
def transpose(g, ctx): return G(zip(*g))


def crop_content(g, ctx, bg):
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != bg]
    if not cells:
        return None
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    return G(row[c0:c1 + 1] for row in g[r0:r1 + 1])


def crop_bbox_of(g, ctx, colour):
    """Crop the current grid to the bounding box that `colour` occupies in the ORIGINAL input."""
    src = ctx.input
    cells = [(r, c) for r, row in enumerate(src) for c, v in enumerate(row) if v == colour]
    if not cells or shape(src) != shape(g):
        return None
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    return G(row[c0:c1 + 1] for row in g[r0:r1 + 1])


# ------------------------------------------------------------------ colour
def recolor(g, ctx, mapping):
    m = dict(mapping)
    if "INK" in m:
        # OPT188: role mapping - every non-background colour becomes m['INK']
        bg = P.background(g, "zero_if_present")
        return G([v if v == bg else m["INK"] for v in r] for r in g)
    return G([m.get(v, v) for v in r] for r in g)


def obj_recolour(g, ctx, spec):
    """OPT200: every object takes the colour a learned object rule gives it (dsl/objlearn.py):
    a literal or a role such as the colour it touches, its container's or its nearest object's."""
    from .objlearn import apply_recolour
    return apply_recolour(g, spec)


def obj_move(g, ctx, spec):
    """OPT200: objects slide in a learned direction (a literal, or a role such as towards the
    nearest / largest object or the nearest border) until they touch something (dsl/objlearn.py)."""
    from .objlearn import apply_move
    return apply_move(g, spec)


def obj_rays(g, ctx, spec):
    """OPT201: objects selected by a learned object rule cast rays in learned directions (literal sets
    or roles such as towards the nearest object), in a learned colour, stopping at the first obstacle
    or running to the border over background (dsl/objlearn.py)."""
    from .objlearn import apply_rays
    return apply_rays(g, spec)


def obj_connect(g, ctx, spec):
    """OPT201: aligned objects (same row / column / diagonal through their centres) with only
    background between them are joined by a line (dsl/objlearn.py)."""
    from .objlearn import apply_connect
    return apply_connect(g, spec)


def obj_map(g, ctx, spec):
    """OPT202: object mapping engine - every object class gets a program (shape transform, colour
    rule, placement) read off matched input/output objects (dsl/objmap.py)."""
    from .objmap import apply_spec as _om
    return _om(g, spec)


def where(g, ctx, spec):
    """OPT200: a step whose changes are kept only on the objects having a property value.
    spec = (view, bg, prop, value, step); built by the goal-directed composition (search.greedy_compose)."""
    from .lift import regions, _features
    from .search import _apply_step
    view, bgv, prop, value, step = spec
    bg = resolve_bg(bgv, g)
    regs = regions(g, view, bg)
    if not regs:
        return None
    new = _apply_step(step, (g,), (ctx.input,))
    if new is None:
        return None
    new = new[0]
    if len(new) != len(g) or len(new[0]) != len(g[0]):
        return None
    keep = set()
    for (cells, _, _), f in zip(regs, _features(g, regs)):
        if f[prop] == value:
            keep.update(cells)
    out = G([new[r][c] if (r, c) in keep else g[r][c] for c in range(len(g[0]))] for r in range(len(g)))
    return out if out != g else None


def per_region(g, ctx, spec):
    """OPT200: a DSL sub-program applied to every region (object or panel), or one sub-program per
    class of regions (see dsl/lift.py)."""
    from .lift import apply_spec as _lift_apply
    return _lift_apply(g, spec)


def fill_gaps(g, ctx, axis, colour, bg):
    """OPT199: along every row ('h') / column ('v') / both, a run of background cells with ink on
    both sides becomes `colour`."""
    out = [list(r) for r in g]
    did = False
    for ax in (("h", "v") if axis == "both" else (axis,)):
        src = g if ax == "h" else G(zip(*g))
        for i, row in enumerate(src):
            ink = [j for j, v in enumerate(row) if v != bg]
            for a, b in zip(ink, ink[1:]):
                for j in range(a + 1, b):
                    r, c = (i, j) if ax == "h" else (j, i)
                    out[r][c] = colour; did = True
    return G(out) if did else None


def edge_pair_lines(g, ctx, bg):
    """OPT199: the one colour with cells facing each other on opposite grid edges (left/right in a
    row, top/bottom in a column) draws a full line between every such pair on a blank canvas."""
    H, W = shape(g)
    pairs = {}
    for r in range(H):
        if g[r][0] == g[r][W - 1] != bg:
            pairs.setdefault(g[r][0], []).append(("h", r))
    for c in range(W):
        if g[0][c] == g[H - 1][c] != bg:
            pairs.setdefault(g[0][c], []).append(("v", c))
    if not pairs:
        return None
    top = max(len(v) for v in pairs.values())
    best = [k for k, v in pairs.items() if len(v) == top]
    if len(best) != 1:
        return None
    col = best[0]
    out = [[bg] * W for _ in range(H)]
    for ax, i in pairs[col]:
        if ax == "h":
            out[i] = [col] * W
        else:
            for r in range(H):
                out[r][i] = col
    return G(out)


def line_uniform(g, ctx, spec):
    """OPT199: every row ('h') or column ('v') made of one colour becomes `yes`, every other line
    becomes `no` (spec = (axis, yes, no))."""
    axis, yes, no = spec
    src = g if axis == "h" else G(zip(*g))
    out = G([yes] * len(r) if len(set(r)) == 1 else [no] * len(r) for r in src)
    return out if axis == "h" else G(zip(*out))


def role_recolour(g, ctx, spec):
    """OPT199: a two-colour grid with a fixed colour k (shared by every input) and one other colour X.
    spec = (k, new_k, new_x); a new value is 'X' (the other colour), 'K' (k) or a literal colour."""
    k, nk, nx = spec
    cs = sorted(colours(g))
    if len(cs) != 2 or k not in cs:
        return None
    x = cs[0] if cs[1] == k else cs[1]
    val = lambda v: x if v == "X" else k if v == "K" else v
    a, b = val(nk), val(nx)
    return G([a if v == k else b for v in r] for r in g)


def swap_colours(g, ctx):
    """Exchange the two colours of a two-colour grid (relational: no colour is named)."""
    cs = colours(g)
    if len(cs) != 2:
        return None
    a, b = sorted(cs)
    return G([b if v == a else a for v in r] for r in g)


# ------------------------------------------------------------------ size: shrink
def _reduce(cells, reducer, bg):
    ct = Counter(cells)
    if reducer == "majority":
        top = ct.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None
        return top[0][0] if top[0][1] * 2 > len(cells) else None
    if reducer == "uniform":
        return cells[0] if len(ct) == 1 else None
    if reducer == "only_non_bg":
        rest = set(ct) - {bg}
        return next(iter(rest)) if len(rest) == 1 else None
    if reducer == "any_non_bg":
        rest = set(ct) - {bg}
        return bg if not rest else (next(iter(rest)) if len(rest) == 1 else None)
    return None


def pool(g, ctx, block, reducer, bg):
    """Reduce each block_h x block_w block to one cell."""
    h, w = shape(g)
    kind, a, b = block
    if kind == "cells":  # fixed output shape -> block size from this grid
        if a < 1 or b < 1 or h % a or w % b:
            return None
        a, b = h // a, w // b
    if a < 1 or b < 1 or (a, b) == (1, 1) or h % a or w % b:
        return None
    out = []
    for r in range(h // a):
        row = []
        for c in range(w // b):
            v = _reduce([g[x][y] for x in range(r * a, (r + 1) * a) for y in range(c * b, (c + 1) * b)], reducer, bg)
            if v is None:
                return None
            row.append(pbg if v == "BG" else v)
        out.append(row)
    return G(out)


def separator_lines(g):
    """Colour whose full rows AND/OR columns split the grid (see perception.separator_lines)."""
    return P.separator_lines(g)


def _runs(n, seps):
    out, start = [], None
    for i in range(n + 1):
        if i == n or i in seps:
            if start is not None:
                out.append((start, i))
            start = None
        elif start is None:
            start = i
    return out


def pool_cells(g, ctx, reducer, bg):
    """Split at separator lines and reduce every cell to one value."""
    s = separator_lines(g)
    if not s:
        return None
    _, rs, cs = s
    rr, cc = _runs(len(g), set(rs)), _runs(len(g[0]), set(cs))
    if len(rr) * len(cc) < 2:
        return None
    out = []
    for r0, r1 in rr:
        row = []
        for c0, c1 in cc:
            v = _reduce([g[x][y] for x in range(r0, r1) for y in range(c0, c1)], reducer, bg)
            if v is None:
                return None
            row.append(pbg if v == "BG" else v)
        out.append(row)
    return G(out)


def dedup_rows(g, ctx):
    return G(r for i, r in enumerate(g) if i == 0 or r != g[i - 1])


def dedup_cols(g, ctx):
    return transpose(dedup_rows(transpose(g, ctx), ctx), ctx)


# ------------------------------------------------------------------ size: grow
def tile(g, ctx, reps):
    n, m = reps
    h, w = shape(g)
    if (n, m) == (1, 1) or h * n > MAX or w * m > MAX:
        return None
    return G([g[r % h][c % w] for c in range(w * m)] for r in range(h * n))


_D8 = {
    "id": lambda g: g, "fh": lambda g: G(r[::-1] for r in g), "fv": lambda g: G(g[::-1]),
    "r180": lambda g: G(r[::-1] for r in g[::-1]), "tr": lambda g: G(zip(*g)),
    "r90": lambda g: G(zip(*g[::-1])), "r270": lambda g: G(list(zip(*g))[::-1]),
    "atr": lambda g: G(r[::-1] for r in list(zip(*g))[::-1])}


def arrange(g, ctx, layout):
    """Output = a grid of blocks; block (i, j) is the input under the D8 transform layout[i][j]
    (mirror / rotation tilings). The layout is read off the demonstrations."""
    rows = []
    for lrow in layout:
        blocks = [_D8[t](g) for t in lrow]
        if len({len(b) for b in blocks}) != 1:
            return None
        for k in range(len(blocks[0])):
            rows.append(sum((tuple(b[k]) for b in blocks), ()))
    if len(rows) > MAX or len(rows[0]) > MAX or len({len(r) for r in rows}) != 1:
        return None
    return G(rows)


def _split_by_lines(g):
    """Equal panels separated by full uniform lines of one colour (any number of panels, rows or columns)."""
    h, w = shape(g)
    found = []
    for axis in ("h", "v"):
        n = w if axis == "h" else h
        line = (lambda j: {g[r][j] for r in range(h)}) if axis == "h" else (lambda j: set(g[j]))
        for col in sorted({v for r in g for v in r}):
            seps = [j for j in range(n) if line(j) == {col}]
            if not seps or len(seps) >= n - 1:
                continue
            cuts, start = [], 0
            for j in seps + [n]:
                if j > start:
                    cuts.append((start, j))
                start = j + 1
            widths = {b - a for a, b in cuts}
            if len(cuts) >= 2 and len(widths) == 1:
                found.append((axis, cuts))
    if len(found) != 1:
        return None
    axis, cuts = found[0]
    if axis == "h":
        return [tuple(tuple(r[a:b]) for r in g) for a, b in cuts]
    return [tuple(g[a:b]) for a, b in cuts]


def _split_halves(g):
    by_lines = _split_by_lines(g)
    if by_lines:
        return by_lines
    """Two equal panels found from the grid itself (size may differ per grid): a uniform middle
    separator line (odd width / height) is preferred; otherwise an even split if only one axis allows it."""
    h, w = shape(g)
    cands = []
    if w % 2 == 1 and w >= 3 and len({g[r][w // 2] for r in range(h)}) == 1:
        cands.append(("h", True))
    if h % 2 == 1 and h >= 3 and len(set(g[h // 2])) == 1:
        cands.append(("v", True))
    if not cands:
        if w % 2 == 0 and h % 2 == 1:
            cands.append(("h", False))
        elif h % 2 == 0 and w % 2 == 1:
            cands.append(("v", False))
    if len(cands) != 1:
        return None
    axis, sep = cands[0]
    if axis == "h":
        k = w // 2
        return [tuple(tuple(r[:k]) for r in g), tuple(tuple(r[w - k:]) for r in g)]
    k = h // 2
    return [tuple(g[:k]), tuple(g[h - k:])]


def _split_panels(g, n_cells):
    if n_cells == "halves":
        return _split_halves(g)
    """Equal panels of size n_cells, cut along one axis, with or without 1-cell separator lines
    (a separator line must be one uniform colour)."""
    h, w = shape(g)
    ph, pw = n_cells
    for axis in (1, 0):
        n, p, other, op_ = (w, pw, h, ph) if axis == 1 else (h, ph, w, pw)
        if other != op_:
            continue
        for sep in (0, 1):
            if (n + sep) % (p + sep):
                continue
            k = (n + sep) // (p + sep)
            if k < 2:
                continue
            cuts = [i * (p + sep) for i in range(k)]
            if sep:
                lines = [c + p for c in cuts[:-1]]
                vals = [{g[r][c] for r in range(h)} if axis == 1 else set(g[c]) for c in lines]
                if any(len(v) != 1 for v in vals):
                    continue
            if axis == 1:
                pans = [tuple(tuple(r[c:c + p]) for r in g) for c in cuts]
            else:
                pans = [tuple(g[c:c + p]) for c in cuts]
            return pans
    return _split_grid(g, n_cells)


def _split_grid(g, n_cells):
    """OPT199: panels in a 2D grid (>= 2 rows and >= 2 columns of panels), with or without uniform
    1-cell separator lines; row-major order."""
    h, w = shape(g)
    ph, pw = n_cells
    for sep in (0, 1):
        if (h + sep) % (ph + sep) or (w + sep) % (pw + sep):
            continue
        kr, kc = (h + sep) // (ph + sep), (w + sep) // (pw + sep)
        if kr < 2 or kc < 2:
            continue
        rs = [i * (ph + sep) for i in range(kr)]
        cs = [i * (pw + sep) for i in range(kc)]
        if sep:
            if any(len(set(g[r + ph])) != 1 for r in rs[:-1]):
                continue
            if any(len({g[r][c + pw] for r in range(h)}) != 1 for c in cs[:-1]):
                continue
        return [tuple(tuple(row[c0:c0 + pw]) for row in g[r0:r0 + ph]) for r0 in rs for c0 in cs]
    return None


BOOL = {"and": lambda a: all(a), "or": lambda a: any(a), "xor": lambda a: sum(a) % 2 == 1,
        "nor": lambda a: not any(a), "one": lambda a: sum(a) == 1, "first_only": lambda a: a[0] and not any(a[1:]),
        "last_only": lambda a: a[-1] and not any(a[:-1])}


def _panel_bg(pans, g, bgpol):
    if bgpol == "panel":
        from collections import Counter as _C
        return [_C(v for r in p for v in r).most_common(1)[0][0] for p in pans]
    b = P.background(g, bgpol)
    return [b] * len(pans)


def panel_logic(g, ctx, spec):
    """Cell-wise combination of equal panels (cut with or without separator lines).
    spec = (panel shape, background policy, mode, data)
      mode 'table'    : output cell = data[(is-ink in panel 1, ..., panel k)]
      mode 'priority' : output cell = colour of the first panel (in order data) that has ink there,
                        else background
      mode 'bool'     : data = (op, ink colour, blank colour): ink where BOOL[op](is-ink per panel)
    background policy 'panel' (OPT199): each panel's own most frequent colour"""
    size, bgpol, mode, data = spec
    pans = _split_panels(g, size)
    if not pans:
        return None
    if size == "halves":
        size = (len(pans[0]), len(pans[0][0]))
    bgs = _panel_bg(pans, g, bgpol)
    if mode == "bool":
        f, ink, blank = BOOL[data[0]], data[1], data[2]
        return G([[ink if f([p[r][c] != b for p, b in zip(pans, bgs)]) else blank for c in range(size[1])]
                  for r in range(size[0])])
    bg = bgs[0]
    out = []
    if mode == "table":
        t = dict(data)
        if len(next(iter(t))) != len(pans):
            return None
    elif max(data) >= len(pans):
        return None
    for r in range(size[0]):
        row = []
        for c in range(size[1]):
            if mode == "table":
                key = tuple(int(p[r][c] != b) for p, b in zip(pans, bgs))
                if key not in t:
                    return None
                row.append(t[key])
            else:
                v = bg
                for k in data:
                    if pans[k][r][c] != bgs[k]:
                        v = pans[k][r][c]; break
                row.append(v)
        out.append(row)
    return G(out)


def count_render(g, ctx, spec):
    """Count something in the input and draw the number (see dsl/count_render.py)."""
    from .count_render import render
    return render(g, spec)


def stencil(g, ctx, spec):
    """Every source cell of colour c paints its colour's learned neighbourhood pattern (dsl/stencil.py)."""
    from .stencil import apply_spec as stencil_apply
    return stencil_apply(g, spec)


def box_objects(g, ctx, conn, mode, ink, bg):
    """For every one-colour object (conn 4/8): mode 'fill' paints the background cells inside its
    bounding box with ink; 'ring' paints the one-cell ring just outside the box."""
    h, w = shape(g)
    out = [list(r) for r in g]
    objs = P.objects(g, bg=bg, conn=conn, by_colour=True)
    if not objs or len(objs) > 40:
        return None
    for cells in objs:
        r0, c0, r1, c1 = P.bbox(cells)
        if mode == "fill":
            pts = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        else:
            pts = [(r, c) for r in range(r0 - 1, r1 + 2) for c in range(c0 - 1, c1 + 2)
                   if not (r0 <= r <= r1 and c0 <= c <= c1)]
        for r, c in pts:
            if 0 <= r < h and 0 <= c < w and g[r][c] == bg:
                out[r][c] = ink
    return G(out)


def colour_histogram(g, ctx, spec):
    """Colours sorted by cell count, drawn as bars or a list (see dsl/histogram.py)."""
    from .histogram import apply_spec as hist_apply
    return hist_apply(g, spec)


def recolour_by_property(g, ctx, spec):
    """Repaint every object with the colour its property value maps to (dsl/prop_recolour.py)."""
    from .prop_recolour import apply_spec as pr_apply
    return pr_apply(g, spec)


def object_rule(g, ctx, spec):
    """One property of each object decides its new colour or whether it is erased (dsl/object_rules.py)."""
    from .object_rules import apply_spec as or_apply
    return or_apply(g, spec)


def to_boundary(g, ctx, spec):
    """Move / extend to a boundary, clean up noise, mark centres (dsl/boundary.py)."""
    from .boundary import apply_spec as b_apply
    return b_apply(g, spec)


def layers(g, ctx, spec):
    """Stacking order of overlapping shapes: report / reverse / raise (dsl/layers.py)."""
    from .layers import apply_spec as l_apply
    return l_apply(g, spec)


def copy_to_markers(g, ctx, spec):
    """Copy the template object onto every marker cell (dsl/copy_markers.py)."""
    from .copy_markers import apply_spec as cm_apply
    return cm_apply(g, spec)


def scale_up(g, ctx, factor):
    if factor == "NCOL":
        # role: the factor is the number of distinct non-background colours of this grid (OPT196)
        n = len(colours(g) - {most_common(g)}) if len(colours(g)) > 1 else 1
        factor = (n, n)
    a, b = factor
    h, w = shape(g)
    if (a, b) == (1, 1) or h * a > MAX or w * b > MAX:
        return None
    return G([g[r // a][c // b] for c in range(w * b)] for r in range(h * a))


def substitute_cells(g, ctx, table):
    """Replace every cell by the tile the demonstrations associate with its colour."""
    t = dict(table)
    if not colours(g) <= set(t):
        return None
    a, b = shape(next(iter(t.values())))
    h, w = shape(g)
    if h * a > MAX or w * b > MAX:
        return None
    return G([t[g[r // a][c // b]][r % a][c % b] for c in range(w * b)] for r in range(h * a))


def self_kron(g, ctx, selector, bg):
    """n x n copies of the grid on a bg canvas; block (i, j) is a copy where selector(i, j) holds."""
    h, w = shape(g)
    if h * h > MAX or w * w > MAX:
        return None
    sel = SELECTORS[selector]
    keep = [[sel(g, i, j, bg) for j in range(w)] for i in range(h)]
    if all(all(r) for r in keep) or not any(any(r) for r in keep):
        return None
    return G([g[r % h][c % w] if keep[r // h][c // w] else bg for c in range(w * w)] for r in range(h * h))


SELECTORS = {
    "cell_not_bg": lambda g, i, j, bg: g[i][j] != bg,
    "cell_is_bg": lambda g, i, j, bg: g[i][j] == bg,
    "row_uniform": lambda g, i, j, bg: len(set(g[i])) == 1,
    "col_uniform": lambda g, i, j, bg: len({r[j] for r in g}) == 1,
    "line_uniform": lambda g, i, j, bg: len(set(g[i])) == 1 or len({r[j] for r in g}) == 1,
    # OPT182: relational colour selectors (no colour is named)
    "cell_is_majority": lambda g, i, j, bg: g[i][j] == _rank_colour(g, 0),
    "cell_is_minority": lambda g, i, j, bg: g[i][j] == _rank_colour(g, -1),
}


def _rank_colour(g, k):
    """Most (k=0) / least (k=-1) frequent colour; None on a tie."""
    cnt = Counter(v for r in g for v in r).most_common()
    if len(cnt) < 2:
        return None
    i = 0 if k == 0 else len(cnt) - 1
    j = 1 if k == 0 else len(cnt) - 2
    return cnt[i][0] if cnt[i][1] != cnt[j][1] else None


def pad(g, ctx, width, mode, corner):
    """Add a border of `width` cells: 'edge' copies the nearest edge cell (corners get `corner`),
    'const' fills with `corner`."""
    h, w = shape(g)
    k = width
    if k < 1 or h + 2 * k > MAX or w + 2 * k > MAX:
        return None
    out = [[corner] * (w + 2 * k) for _ in range(h + 2 * k)]
    for r in range(h + 2 * k):
        for c in range(w + 2 * k):
            ir, ic = r - k, c - k
            inr, inc = 0 <= ir < h, 0 <= ic < w
            if inr and inc:
                out[r][c] = g[ir][ic]
            elif mode == "edge" and (inr or inc):
                out[r][c] = g[min(max(ir, 0), h - 1)][min(max(ic, 0), w - 1)]
    return G(out)


def canvas(g, ctx, size, bg):
    """Place the grid at the top-left of a larger bg canvas (size = multiplier per axis)."""
    a, b = size
    h, w = shape(g)
    if (a, b) == (1, 1) or h * a > MAX or w * b > MAX:
        return None
    return G([g[r][c] if r < h and c < w else bg for c in range(w * b)] for r in range(h * a))


def rays(g, ctx, direction, bg):
    """From every non-bg cell, extend a ray of its colour in `direction` until the edge.
    Conflicting colours make the operation inapplicable."""
    dr, dc = direction
    h, w = shape(g)
    out = [list(r) for r in g]
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v == bg:
                continue
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                if out[rr][cc] not in (bg, v):
                    return None
                out[rr][cc] = v
                rr += dr; cc += dc
    res = G(out)
    return res if res != g else None


def top_left_quadrant(g, ctx):
    """Top-left quarter (rounded up) of the grid: the repeating unit of a 4-fold symmetric pattern."""
    h, w = shape(g)
    if h < 2 or w < 2:
        return None
    return G(row[:(w + 1) // 2] for row in g[:(h + 1) // 2])


# ------------------------------------------------------------------ periodicity
def _min_period(seq):
    n = len(seq)
    for p in range(1, n):
        if 2 * n >= 3 * p and all(seq[i] == seq[i % p] for i in range(n)):
            return p
    return None


def extend_period(g, ctx, axis, ratio):
    """Repeat the minimal period along an axis until the axis is ratio (num/den) times longer."""
    num, den = ratio
    seq = list(g) if axis == 0 else list(zip(*g))
    n = len(seq)
    if n * num % den or n * num // den <= n or n * num // den > MAX:
        return None
    p = _min_period(seq)
    if not p:
        return None
    v = [seq[i % p] for i in range(n * num // den)]
    return G(v) if axis == 0 else G(zip(*v))


def repair_period(g, ctx, hole):
    """Fill cells of colour `hole` from the smallest 2D period consistent with the rest."""
    h, w = shape(g)
    holes = [(r, c) for r in range(h) for c in range(w) if g[r][c] == hole]
    if not holes:
        return None
    best = None
    for pr in range(1, h // 2 + 1):
        for pc in range(1, w // 2 + 1):
            if best and pr * pc >= best[0]:
                continue
            vals, good = {}, True
            for r in range(h):
                for c in range(w):
                    if g[r][c] == hole:
                        continue
                    k = (r % pr, c % pc)
                    if vals.setdefault(k, g[r][c]) != g[r][c]:
                        good = False; break
                if not good:
                    break
            if good and all((r % pr, c % pc) in vals for r, c in holes):
                best = (pr * pc, pr, pc, vals)
    if not best:
        return None
    _, pr, pc, vals = best
    return G([vals[(r % pr, c % pc)] if g[r][c] == hole else g[r][c] for c in range(w)] for r in range(h))



# ------------------------------------------------------------------ drawing (port 2: marker_drawing)
def _paint(out, r, c, v, bg):
    """Paint a background cell; a conflicting non-background cell makes the drawing inapplicable."""
    if out[r][c] == bg or out[r][c] == v:
        out[r][c] = v
        return True
    return False


def fill_bbox_by_colour(g, ctx, bg, mode):
    """For every colour with 2+ cells, fill ('fill') or outline ('outline') the box its cells span."""
    out = [list(r) for r in g]
    for col in sorted(colours(g) - {bg}):
        box = P.bbox_of(g, lambda v, col=col: v == col)
        r0, c0, r1, c1 = box
        if sum(v == col for row in g for v in row) < 2 or r0 == r1 or c0 == c1:
            continue
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if mode == "outline" and r0 < r < r1 and c0 < c < c1:
                    continue
                if not _paint(out, r, c, col, bg):
                    return None
    res = G(out)
    return res if res != g else None


def marker_rays(g, ctx, direction, pattern, ink, bg):
    """From every single-cell object, draw a ray to the edge. pattern 'solid' uses the marker colour,
    'alternate' alternates ink and marker colour (ink first)."""
    dr, dc = direction
    h, w = shape(g)
    out = [list(r) for r in g]
    singles = [o for o in P.objects(g, bg, 4, True) if len(o) == 1]
    if not singles:
        return None
    for (r, c), in singles:
        col, k = g[r][c], 1
        while 0 <= r + k * dr < h and 0 <= c + k * dc < w:
            v = col if pattern == "solid" else (ink if k % 2 else col)
            if not _paint(out, r + k * dr, c + k * dc, v, bg):
                return None
            k += 1
    res = G(out)
    return res if res != g else None


def connect_to_block(g, ctx, bg):
    """Draw a straight line from every single-cell marker to the unique solid rectangle it is
    aligned with (row or column overlap), in the marker's colour, stopping at the rectangle."""
    objs = P.objects(g, bg, 4, True)
    blocks = [o for o in objs if len(o) > 1 and (lambda b: len(o) == (b[2] - b[0] + 1) * (b[3] - b[1] + 1))(P.bbox(o))]
    if len(blocks) != 1:
        return None
    a, b, x, y = P.bbox(blocks[0])
    out = [list(r) for r in g]
    for o in objs:
        if len(o) != 1:
            continue
        (r, c), col = o[0], g[o[0][0]][o[0][1]]
        if a <= r <= x:
            near = b if c < b else y
            span = range(min(c, near) + 1, max(c, near))
            cells = [(r, k) for k in span]
        elif b <= c <= y:
            near = a if r < a else x
            cells = [(k, c) for k in range(min(r, near) + 1, max(r, near))]
        else:
            continue
        for rr, cc in cells:
            if not _paint(out, rr, cc, col, bg):
                return None
    res = G(out)
    return res if res != g else None


def extend_lines(g, ctx, bg, cross):
    """Every straight line object (2+ cells in one row or column) is extended across the grid;
    cells where a row line meets a column line get colour `cross`."""
    h, w = shape(g)
    rows, cols = {}, {}
    for o in P.objects(g, bg, 4, True):
        if len(o) < 2:
            return None
        rs, cs = {r for r, _ in o}, {c for _, c in o}
        col = g[o[0][0]][o[0][1]]
        if len(rs) == 1:
            rows[next(iter(rs))] = col
        elif len(cs) == 1:
            cols[next(iter(cs))] = col
        else:
            return None
    if not rows or not cols:
        return None
    out = [list(r) for r in g]
    for r in range(h):
        for c in range(w):
            if r in rows and c in cols:
                out[r][c] = cross
            elif r in rows:
                out[r][c] = rows[r]
            elif c in cols:
                out[r][c] = cols[c]
    return G(out)


def corner_rays(g, ctx, kind, colour, which, bg):
    """Diagonal rays leaving shapes at their corners, drawn outward to the edge on background cells.
    kind   'tip'            : cells attached to their (8-connected) shape only diagonally
           'missing_corner' : bounding-box corners NOT in the shape (ray starts beyond the corner)
           'bbox_corner'    : bounding-box corners that are in the shape
    colour 'own' (shape colour) or 'other' (the single other non-background colour)
    which  'all' shapes, only the 'largest', or only 'hollow' shapes (not filling their bounding box)
    """
    h, w = shape(g)
    out = [list(r) for r in g]
    objs = P.objects(g, bg, 8 if kind == "tip" else 4, True)
    if which == "largest":
        if not objs:
            return None
        big = max(len(o) for o in objs)
        objs = [o for o in objs if len(o) == big]
        if len(objs) != 1:
            return None
    elif which == "hollow":
        objs = [o for o in objs if (lambda b: len(o) < (b[2] - b[0] + 1) * (b[3] - b[1] + 1))(P.bbox(o))]
        if not objs:
            return None
    cols = colours(g) - {bg}
    starts = []
    for o in objs:
        own = g[o[0][0]][o[0][1]]
        if colour == "other":
            rest = cols - {own}
            if len(rest) != 1:
                return None
            col = next(iter(rest))
        else:
            col = own
        cells = set(o)
        r0, c0, r1, c1 = P.bbox(o)
        if kind == "tip":
            for r, c in o:
                orth = [(r + a, c + b) in cells for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                diag = [(a, b) for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1)) if (r + a, c + b) in cells]
                if not any(orth) and len(diag) == 1 and len(o) > 1:
                    starts.append((r, c, -diag[0][0], -diag[0][1], col))
        else:
            if r0 == r1 or c0 == c1:
                continue
            for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
                inside = (r, c) in cells
                if (kind == "bbox_corner") == inside:
                    starts.append((r, c, -1 if r == r0 else 1, -1 if c == c0 else 1, col))
    if not starts:
        return None
    for r, c, dr, dc, col in starts:
        r, c = r + dr, c + dc
        while 0 <= r < h and 0 <= c < w:
            if not _paint(out, r, c, col, bg):
                return None
            r, c = r + dr, c + dc
    res = G(out)
    return res if res != g else None


# ------------------------------------------------------------------ object selection (port 3)
from functools import lru_cache as _lru


@_lru(maxsize=4096)
def _scene(g, view):
    """Objects of a grid under a view = (background rank, connectivity, multicolour)."""
    rank, conn, multi = view
    if isinstance(rank, str):
        # OPT199: 'bg<c>' = the task-level background colour c (shared by every demonstration input)
        bg = int(rank[2:])
    else:
        bgs = P.backgrounds(g, 2, "weighted")
        if rank >= len(bgs):
            return None, ()
        bg = bgs[rank]
    objs = P.objects(g, bg, conn, not multi)
    if not objs or len(objs) > 40:
        return bg, ()
    return bg, tuple(tuple(o) for o in objs)


def _shape_key(o):
    r0 = min(r for r, _ in o); c0 = min(c for _, c in o)
    return tuple(sorted((r - r0, c - c0) for r, c in o))


def _pick(g, objs, selector):
    def colour(o):
        return Counter(g[r][c] for r, c in o).most_common(1)[0][0]

    def size(o, axis):
        b = P.bbox(o)
        return b[2] - b[0] + 1 if axis == 0 else b[3] - b[1] + 1

    if selector in ("largest", "smallest"):
        v = (max if selector == "largest" else min)(len(o) for o in objs)
        return [o for o in objs if len(o) == v]
    if selector in ("most_common_shape", "least_common_shape"):
        # OPT199: the shape (with colours) that occurs most / least often; must be unique
        ct = Counter(_shape_key(o) for o in objs)
        v = (max if selector == "most_common_shape" else min)(ct.values())
        keys = {k for k, n in ct.items() if n == v}
        if len(keys) != 1 or len(ct) < 2:
            return []
        return [o for o in objs if _shape_key(o) in keys]
    if selector in ("unique_shape", "repeated_shape"):
        ct = Counter(_shape_key(o) for o in objs)
        return [o for o in objs if (ct[_shape_key(o)] == 1) == (selector == "unique_shape")]
    if selector in ("rare_colour", "common_colour"):
        ct = Counter(colour(o) for o in objs)
        v = (min if selector == "rare_colour" else max)(ct.values())
        return [o for o in objs if ct[colour(o)] == v]
    if selector in ("tallest", "shortest", "widest", "narrowest"):
        axis = 0 if selector in ("tallest", "shortest") else 1
        v = (max if selector in ("tallest", "widest") else min)(size(o, axis) for o in objs)
        return [o for o in objs if size(o, axis) == v]
    if selector in ("most_colours", "fewest_colours"):
        k = lambda o: len({g[r][c] for r, c in o})
        v = (max if selector == "most_colours" else min)(k(o) for o in objs)
        return [o for o in objs if k(o) == v]
    if selector in ("most_marks", "fewest_marks"):
        # marks = cells that are not the object's own dominant colour
        k = lambda o: len(o) - Counter(g[r][c] for r, c in o).most_common(1)[0][1]
        v = (max if selector == "most_marks" else min)(k(o) for o in objs)
        if v == 0 and selector == "most_marks":
            return []
        return [o for o in objs if k(o) == v]
    if isinstance(selector, tuple) and selector[0] == "most_of_colour":
        # most cells of a given colour (the colour comes from the demonstrations)
        col = selector[1]
        k = lambda o: sum(g[r][c] == col for r, c in o)
        v = max(k(o) for o in objs)
        return [o for o in objs if k(o) == v] if v > 0 else []
    if selector == "most_rare_colour":
        # most cells of the grid's least frequent non-background colour
        ct = Counter(v for row in g for v in row)
        bgc = ct.most_common(1)[0][0]
        rare = min((k for k in ct if k != bgc), key=lambda k: (ct[k], k), default=None)
        if rare is None:
            return []
        k = lambda o: sum(g[r][c] == rare for r, c in o)
        v = max(k(o) for o in objs)
        return [o for o in objs if k(o) == v] if v > 0 else []
    if selector == "most_inside":
        # most cells of other (non-background) colours inside the object's bounding box
        bgc = Counter(v for row in g for v in row).most_common(1)[0][0]
        def k(o):
            cells = set(o); b = P.bbox(o)
            return sum(1 for r in range(b[0], b[2] + 1) for c in range(b[1], b[3] + 1)
                       if (r, c) not in cells and g[r][c] != bgc)
        v = max(k(o) for o in objs)
        return [o for o in objs if k(o) == v] if v > 0 else []
    if selector in ("densest", "sparsest"):
        k = lambda o: len(o) / (size(o, 0) * size(o, 1))
        v = (max if selector == "densest" else min)(k(o) for o in objs)
        return [o for o in objs if k(o) == v]
    return []


def select_objects(g, ctx, view, selector, mode):
    """Choose objects by a property and render them.
    mode 'keep'        : selected objects on a background canvas of the same size
         'delete'      : erase the selected objects
         'crop'        : bounding box of the selected objects, only their cells
         'crop_context': bounding box of the selected objects, everything inside it
    """
    bg, objs = _scene(g, view)
    if not objs:
        return None
    sel = _pick(g, objs, selector)
    if not sel or len(sel) == len(objs) and mode in ("keep", "delete"):
        return None
    if mode in ("crop", "crop_inside") and len(sel) > 1:
        # OPT199: several selected objects that look identical: crop one of them
        crops = {G([g[r][c] if (r, c) in set(o) else bg for c in range(P.bbox(o)[1], P.bbox(o)[3] + 1)]
                   for r in range(P.bbox(o)[0], P.bbox(o)[2] + 1)) for o in sel}
        if len(crops) == 1:
            sel = sel[:1]
    cells = {(r, c) for o in sel for r, c in o}
    h, w = shape(g)
    if mode in ("keep", "delete"):
        keep = (lambda rc: rc in cells) if mode == "keep" else (lambda rc: rc not in cells)
        return G([g[r][c] if g[r][c] == bg or keep((r, c)) else bg for c in range(w)] for r in range(h)) \
            if mode == "delete" else G([g[r][c] if (r, c) in cells else bg for c in range(w)] for r in range(h))
    r0, c0, r1, c1 = P.bbox(cells)
    if mode == "crop":
        return G([g[r][c] if (r, c) in cells else bg for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1))
    if mode == "crop_inside":
        # OPT199: everything strictly inside the selection's bounding box (a frame's contents)
        if r1 - r0 < 2 or c1 - c0 < 2:
            return None
        return G(g[r][c0 + 1:c1] for r in range(r0 + 1, r1))
    return G(g[r][c0:c1 + 1] for r in range(r0, r1 + 1))


# ------------------------------------------------------------------ lines and motion (port 4)
def connect_points(g, ctx, spec):
    """Line grammar (formerly line_relations): connect or extend aligned marker cells along rows,
    columns or diagonals; spec = selector, source, extent, axes, paint, crossing, background."""
    from ..line_relations import execute
    try:
        out = execute(g, dict(spec))
    except (ValueError, KeyError, IndexError):
        return None
    return out  # may legitimately change nothing on some grids


def move_objects(g, ctx, spec):
    """Motion grammar (formerly relative_dir): movers slide / leave trails / cast rays in a direction
    derived from other objects (toward / away / border / facing / per colour / constant)."""
    from ..relative_dir import apply_spec
    try:
        out = apply_spec(g, spec)
    except Exception:
        return None
    return G(out) if out is not None else None


# ------------------------------------------------------------------ keys (M1, OPT177)
def apply_key(g, ctx, spec):
    """Read a key from the grid (where / what) and apply it (how / output); see dsl/keys.py.
    The key's content comes from each grid, never from the demonstrations."""
    from .keys import apply_spec
    return apply_spec(g, spec)


def complete_copies(g, ctx, spec):
    """Complete every partial copy of an exemplar motif (see dsl/stamp.py)."""
    from .stamp import apply_spec as stamp_apply
    return stamp_apply(g, spec)


# ------------------------------------------------------------------ general concepts (OPT193)
def crop_window(g, ctx, spec):
    """A fixed-size window anchored at a corner or the centre of the grid (size learned)."""
    anchor, h, w = spec
    H, W = shape(g)
    if h > H or w > W or (h, w) == (H, W):
        return None
    r0 = 0 if anchor in ("tl", "tr") else H - h if anchor in ("bl", "br") else (H - h) // 2
    c0 = 0 if anchor in ("tl", "bl") else W - w if anchor in ("tr", "br") else (W - w) // 2
    if anchor == "c" and ((H - h) % 2 or (W - w) % 2):
        return None
    return G(row[c0:c0 + w] for row in g[r0:r0 + h])


def _ring_index(H, W, r, c):
    return min(r, c, H - 1 - r, W - 1 - c)


def rings(g, ctx, mode):
    """Concentric rings (distance to the border), each one colour; their colours are reversed
    ('reverse': outermost <-> innermost) or rotated one step inward / outward."""
    H, W = shape(g)
    cols = {}
    for r in range(H):
        for c in range(W):
            k = _ring_index(H, W, r, c)
            if cols.setdefault(k, g[r][c]) != g[r][c]:
                return None
    n = len(cols)
    if n < 2:
        return None
    seq = [cols[k] for k in range(n)]
    if mode == "reverse":
        new = seq[::-1]
    elif mode == "inward":
        new = seq[-1:] + seq[:-1]
    else:
        new = seq[1:] + seq[:1]
    return G([new[_ring_index(H, W, r, c)] for c in range(W)] for r in range(H))


def colour_order(g, ctx, spec):
    """The colours of the grid in their order along its main axis (by mean position), as one row
    (ordered left to right) or one column (top to bottom). spec = (axis, drop): axis 'auto' picks the
    axis along which the colours are most separated; drop 'bg' leaves out the most frequent colour."""
    axis, drop = spec
    H, W = shape(g)
    pos = {}
    for r in range(H):
        for c in range(W):
            pos.setdefault(g[r][c], []).append((r, c))
    if drop == "bg":
        pos.pop(most_common(g), None)
    if len(pos) < 2:
        return None
    mr = {k: sum(r for r, _ in v) / len(v) for k, v in pos.items()}
    mc = {k: sum(c for _, c in v) / len(v) for k, v in pos.items()}
    if axis == "auto":
        spread = lambda m: (max(m.values()) - min(m.values())) / max(1, (H if m is mr else W))
        axis = "rows" if spread(mr) >= spread(mc) else "cols"
    m = mr if axis == "rows" else mc
    order = sorted(pos, key=lambda k: (m[k], k))
    if len({m[k] for k in order}) != len(order):
        return None
    return G([[k] for k in order]) if axis == "rows" else G([order])


def fill_touching(g, ctx, colour, conn, bg):
    """Flood fill: every background region (conn-connected) that touches a cell of `colour`
    (4-adjacent) becomes `colour`."""
    H, W = shape(g)
    out = [list(r) for r in g]
    seen = set()
    nb = ((0, 1), (1, 0), (0, -1), (-1, 0)) + (((1, 1), (1, -1), (-1, 1), (-1, -1)) if conn == 8 else ())
    did = False
    for r in range(H):
        for c in range(W):
            if g[r][c] != bg or (r, c) in seen:
                continue
            comp, stack, touch = [], [(r, c)], False
            seen.add((r, c))
            while stack:
                a, b = stack.pop(); comp.append((a, b))
                for da, db in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    x, y = a + da, b + db
                    if 0 <= x < H and 0 <= y < W and g[x][y] == colour:
                        touch = True
                for da, db in nb:
                    x, y = a + da, b + db
                    if 0 <= x < H and 0 <= y < W and g[x][y] == bg and (x, y) not in seen:
                        seen.add((x, y)); stack.append((x, y))
            if touch:
                did = True
                for a, b in comp:
                    out[a][b] = colour
    return G(out) if did else None


def mirror_about(g, ctx, mode, bg):
    """Reflect every other object through the centre lines of the pivot object - the one object
    whose colour is unique and whose shape is symmetric both ways - across its vertical axis ('h'),
    horizontal axis ('v') or both ('both'); reflections only paint background."""
    objs = P.objects(g, bg=bg, conn=8, by_colour=True)
    if len(objs) < 2:
        return None
    cand = []
    for o in objs:
        cs = {g[r][c] for r, c in o}
        r0, c0, r1, c1 = P.bbox(o)
        pts = set(o)
        sym = all((r, c0 + c1 - c) in pts and (r0 + r1 - r, c) in pts for r, c in o)
        others = {g[r][c] for q in objs if q is not o for r, c in q}
        if sym and not (cs & others):
            cand.append((o, (r0 + r1) / 2, (c0 + c1) / 2))
    if len(cand) > 1:
        # several symmetric unique-colour objects: the pivot is the unique smallest one
        m = min(len(o) for o, _, _ in cand)
        cand = [x for x in cand if len(x[0]) == m]
    if len(cand) != 1:
        return None
    piv, cr, cc = cand[0]
    H, W = shape(g)
    out = [list(r) for r in g]
    pset = set(piv)
    move = mode.endswith("_move")      # OPT197: '<mode>_move' erases the original (a reflection, not a copy)
    mode = mode[:-5] if move else mode
    if move:
        for o in objs:
            if o is not piv:
                for r, c in o:
                    out[r][c] = bg
    for o in objs:
        if o is piv:
            continue
        for r, c in o:
            v = g[r][c]
            targets = []
            if mode in ("h", "both"):
                targets.append((r, 2 * cc - c))
            if mode in ("v", "both"):
                targets.append((2 * cr - r, c))
            if mode == "both":
                targets.append((2 * cr - r, 2 * cc - c))
            for a, b in targets:
                if a != int(a) or b != int(b):
                    return None
                a, b = int(a), int(b)
                if not (0 <= a < H and 0 <= b < W):
                    return None
                if (a, b) not in pset and (g[a][b] == bg or move):
                    out[a][b] = v
    return G(out)


def border_lines(g, ctx, colour, bg):
    """Rows / columns whose two border cells are both `colour` become full lines of `colour` on an
    otherwise cleared (background) grid."""
    H, W = shape(g)
    if colour == "PAIR":
        # role: the one non-background colour that sits at both ends of some row or column
        cs = {g[r][0] for r in range(H) if g[r][0] == g[r][W - 1] != bg} | \
             {g[0][c] for c in range(W) if g[0][c] == g[H - 1][c] != bg}
        if len(cs) != 1:
            return None
        colour = cs.pop()
    rows = [r for r in range(H) if g[r][0] == colour and g[r][W - 1] == colour]
    cols = [c for c in range(W) if g[0][c] == colour and g[H - 1][c] == colour]
    if not rows and not cols:
        return None
    out = [[bg] * W for _ in range(H)]
    for r in rows:
        out[r] = [colour] * W
    for c in cols:
        for r in range(H):
            out[r][c] = colour
    return G(out)


def fill_largest_rect(g, ctx, colour, bg):
    """The largest all-background rectangle (unique) is filled with `colour`."""
    H, W = shape(g)
    best, area, ties = None, 0, 0
    heights = [0] * W
    for r in range(H):
        for c in range(W):
            heights[c] = heights[c] + 1 if g[r][c] == bg else 0
        for c0 in range(W):
            h = heights[c0]
            for c1 in range(c0, W):
                h = min(h, heights[c1])
                if h == 0:
                    break
                a = h * (c1 - c0 + 1)
                if a > area:
                    best, area, ties = (r - h + 1, c0, r, c1), a, 0
                elif a == area and best != (r - h + 1, c0, r, c1):
                    ties += 1
    if best is None or ties or area < 2:
        return None
    r0, c0, r1, c1 = best
    out = [list(x) for x in g]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r][c] = colour
    return G(out)


def project_markers(g, ctx, bg):
    """Every marker cell (any ink outside the largest object) looks along its row and column through
    background; the first cell of the largest object it meets takes the marker's colour."""
    H, W = shape(g)
    objs = P.objects(g, bg=bg, conn=8, by_colour=False)
    if len(objs) < 2:
        return None
    big = max(objs, key=len)
    if sum(1 for o in objs if len(o) == len(big)) > 1:
        return None
    bset = set(big)
    out = [list(x) for x in g]
    did = False
    cells = [(r, c) for o in objs if o is not big for r, c in o]
    for r, c in cells:
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            a, b = r + dr, c + dc
            while 0 <= a < H and 0 <= b < W and (a, b) not in bset:
                if g[a][b] != bg:
                    break
                a += dr; b += dc
            if 0 <= a < H and 0 <= b < W and (a, b) in bset:
                out[a][b] = g[r][c]; did = True
                break
    return G(out) if did else None


def interior_by_property(g, ctx, spec):
    """Repaint every object's interior by one of its properties (see dsl/prop_recolour.py)."""
    from .prop_recolour import apply_interior
    return apply_interior(g, spec)


def gravity(g, ctx, direction, bg):
    """Cell-level gravity: in every column (or row) the non-background cells fall toward one edge,
    keeping their order; static obstacles are not modelled (everything falls)."""
    H, W = shape(g)
    out = [[bg] * W for _ in range(H)]
    if direction in ("down", "up"):
        for c in range(W):
            col = [g[r][c] for r in range(H) if g[r][c] != bg]
            if direction == "down":
                for i, v in enumerate(reversed(col)):
                    out[H - 1 - i][c] = v
            else:
                for i, v in enumerate(col):
                    out[i][c] = v
    else:
        for r in range(H):
            row = [v for v in g[r] if v != bg]
            if direction == "right":
                for i, v in enumerate(reversed(row)):
                    out[r][W - 1 - i] = v
            else:
                for i, v in enumerate(row):
                    out[r][i] = v
    res = G(out)
    return None if res == g else res


def fill_enclosed(g, ctx, colour, conn, bg, outside="keep"):
    """Background regions (conn-connected) that do not reach the grid border become `colour`;
    OPT199: the regions that do reach it become `outside` (or stay, 'keep')."""
    H, W = shape(g)
    out = [list(r) for r in g]
    seen = set()
    nb = ((0, 1), (1, 0), (0, -1), (-1, 0)) + (((1, 1), (1, -1), (-1, 1), (-1, -1)) if conn == 8 else ())
    did = False
    for r in range(H):
        for c in range(W):
            if g[r][c] != bg or (r, c) in seen:
                continue
            comp, stack, edge = [], [(r, c)], False
            seen.add((r, c))
            while stack:
                a, b = stack.pop(); comp.append((a, b))
                if a in (0, H - 1) or b in (0, W - 1):
                    edge = True
                for da, db in nb:
                    x, y = a + da, b + db
                    if 0 <= x < H and 0 <= y < W and g[x][y] == bg and (x, y) not in seen:
                        seen.add((x, y)); stack.append((x, y))
            if not edge:
                did = True
                for a, b in comp:
                    out[a][b] = colour
            elif outside != "keep":
                for a, b in comp:
                    out[a][b] = outside
    return G(out) if did else None


def symmetrize(g, ctx, mode, bg):
    """Overlay the grid with its own mirror image ('h' left-right, 'v' up-down, 'both', 'r180',
    'tr' transpose for square grids): background cells take the mirrored colour."""
    H, W = shape(g)
    maps = {"h": [lambda r, c: (r, W - 1 - c)], "v": [lambda r, c: (H - 1 - r, c)],
            "both": [lambda r, c: (r, W - 1 - c), lambda r, c: (H - 1 - r, c), lambda r, c: (H - 1 - r, W - 1 - c)],
            "r180": [lambda r, c: (H - 1 - r, W - 1 - c)], "tr": [lambda r, c: (c, r)]}[mode]
    if mode == "tr" and H != W:
        return None
    out = [list(x) for x in g]
    for r in range(H):
        for c in range(W):
            if g[r][c] == bg:
                for f in maps:
                    a, b = f(r, c)
                    if g[a][b] != bg:
                        out[r][c] = g[a][b]; break
    res = G(out)
    return None if res == g else res


def frame(g, ctx, colour):
    """The outermost ring of the grid is painted `colour`."""
    H, W = shape(g)
    if H < 3 or W < 3:
        return None
    out = [list(x) for x in g]
    for r in range(H):
        for c in range(W):
            if r in (0, H - 1) or c in (0, W - 1):
                out[r][c] = colour
    res = G(out)
    return None if res == g else res


def recolour_by_contact(g, ctx, mode, bg):
    """Every object (8-connected, one colour, size > 1) that touches exactly one other colour (a
    marker, 8-adjacent) takes that colour; markers stay ('keep') or are erased ('erase').
    'erase_all' (OPT199): bodies (4-connected, size > 1, all one colour) and single-cell markers of
    other colours; every marker is erased, and a body 4-touching one marker takes its colour."""
    if mode == "erase_all":
        return _contact_erase_all(g, bg)
    H, W = shape(g)
    objs = P.objects(g, bg=bg, conn=8, by_colour=True)
    big = [o for o in objs if len(o) > 1]
    if not big:
        return None
    out = [list(x) for x in g]
    did = False
    for o in big:
        col = g[o[0][0]][o[0][1]]
        s_ = set(o)
        touch = {}
        for r, c in o:
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    x, y = r + a, c + b
                    if (a or b) and 0 <= x < H and 0 <= y < W and (x, y) not in s_ and g[x][y] not in (bg, col):
                        touch.setdefault(g[x][y], set()).add((x, y))
        if len(touch) != 1:
            continue
        (m, cells), = touch.items()
        for r, c in o:
            out[r][c] = m
        if mode == "erase":
            for r, c in cells:
                out[r][c] = bg
        did = True
    return G(out) if did else None


def _contact_erase_all(g, bg):
    objs = P.objects(g, bg=bg, conn=4, by_colour=True)
    bodies = [o for o in objs if len(o) > 1]
    if not bodies:
        return None
    bc = {g[o[0][0]][o[0][1]] for o in bodies}
    if len(bc) != 1:
        return None
    bc = bc.pop()
    marks = {o[0]: g[o[0][0]][o[0][1]] for o in objs if len(o) == 1 and g[o[0][0]][o[0][1]] != bc}
    if not marks:
        return None
    out = [list(x) for x in g]
    for r, c in marks:
        out[r][c] = bg
    for o in bodies:
        adj = {(r + a, c + b) for r, c in o for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0))} & set(marks)
        if len(adj) > 1:
            return None
        if adj:
            m = marks[adj.pop()]
            for r, c in o:
                out[r][c] = m
    return G(out)


def fill_pairs(g, ctx, axis, bg):
    """OPT199: along every row ('h') or column ('v'), the cells of each colour are paired in order
    (1st with 2nd, 3rd with 4th, ...) and the gap of each pair is filled with that colour. A line
    with an odd number of cells of a colour makes the op inapplicable."""
    src = g if axis == "h" else G(zip(*g))
    out = [list(r) for r in src]
    did = False
    for i, row in enumerate(src):
        pos = {}
        for j, v in enumerate(row):
            if v != bg:
                pos.setdefault(v, []).append(j)
        for v, ps in pos.items():
            if len(ps) % 2:
                return None
            for a, b in zip(ps[0::2], ps[1::2]):
                for j in range(a + 1, b):
                    if out[i][j] == bg:
                        out[i][j] = v; did = True
    if not did:
        return None
    out = G(out)
    return out if axis == "h" else G(zip(*out))


def marker_stripes(g, ctx, bg):
    """OPT199: exactly two single-cell markers of different colours. Full lines across the grid's
    short side go through each marker and repeat towards the far end with the markers' spacing,
    the colours alternating."""
    H, W = shape(g)
    pts = [(r, c) for r in range(H) for c in range(W) if g[r][c] != bg]
    if len(pts) != 2 or H == W:
        return None
    (a, b), (x, y) = pts
    if g[a][b] == g[x][y]:
        return None
    axis = 0 if H > W else 1
    (p, v), (q, z) = sorted((r if axis == 0 else c, g[r][c]) for r, c in pts)
    step = q - p
    if step <= 0:
        return None
    out = [list(r) for r in g]
    for k, pos in enumerate(range(p, H if axis == 0 else W, step)):
        col = v if k % 2 == 0 else z
        for j in range(W if axis == 0 else H):
            if axis == 0:
                out[pos][j] = col
            else:
                out[j][pos] = col
    return G(out)


def recolour_to_marker(g, ctx, bg):
    """A single-cell marker names the colour: every other ink cell takes the marker's colour and
    the marker itself is erased. The marker is the one colour that occupies exactly one cell."""
    from collections import Counter as _C
    ct = _C(v for r in g for v in r if v != bg)
    singles = [k for k, n in ct.items() if n == 1]
    if len(singles) != 1 or len(ct) < 2:
        return None
    m = singles[0]
    out = [[bg if v == m else (m if v != bg else bg) for v in r] for r in g]
    return G(out)


def panel_reduce(g, ctx, spec):
    """Panels between separator lines -> one output cell each: the panel's count of `ink` cells
    (cells of neither the separator nor the panel background colour) is looked up in a learned table
    (values past the learned range take the nearest end's colour; 'BG' = the panels' background)."""
    t = dict(spec)
    sep = P.separator_lines(g)
    if not sep:
        return None
    col, rs, cs = sep
    H, W = shape(g)
    rows = P.runs_between(H, rs); cols = P.runs_between(W, cs)
    if len(rows) * len(cols) < 2:
        return None
    flat = [v for r in g for v in r if v != col]
    if not flat:
        return None
    pbg = Counter(flat).most_common(1)[0][0]
    out = []
    ks = sorted(t)
    for r0, r1 in rows:
        row = []
        for c0, c1 in cols:
            n = sum(1 for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if g[r][c] not in (col, pbg))
            v = t.get(n, t[ks[-1]] if n > ks[-1] else t[ks[0]] if n < ks[0] else None)
            if v is None:
                return None
            row.append(pbg if v == "BG" else v)
        out.append(row)
    return G(out)


def recolour_by_position(g, ctx, spec, bg):
    """Ink cells whose row (axis 'r') or column (axis 'c') index is phase modulo k become `colour`."""
    axis, k, phase, colour = spec
    H, W = shape(g)
    out = [list(x) for x in g]
    did = False
    for r in range(H):
        for c in range(W):
            i = r if axis == "r" else c
            if g[r][c] != bg and i % k == phase and g[r][c] != colour:
                out[r][c] = colour; did = True
    return G(out) if did else None


def fill_holes_by_property(g, ctx, spec, bg):
    """Enclosed background regions (not reaching the border, 4-connected) are filled with
    table[property of the region]; property in size / height / width / parity / is_square."""
    prop, table = spec
    t = dict(table)
    H, W = shape(g)
    out = [list(x) for x in g]
    seen = set()
    did = False
    for r in range(H):
        for c in range(W):
            if g[r][c] != bg or (r, c) in seen:
                continue
            comp, stack, edge = [], [(r, c)], False
            seen.add((r, c))
            while stack:
                a, b = stack.pop(); comp.append((a, b))
                if a in (0, H - 1) or b in (0, W - 1):
                    edge = True
                for da, db in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    x, y = a + da, b + db
                    if 0 <= x < H and 0 <= y < W and g[x][y] == bg and (x, y) not in seen:
                        seen.add((x, y)); stack.append((x, y))
            if edge:
                continue
            v = _region_prop(comp, prop)
            if v not in t:
                return None
            for a, b in comp:
                out[a][b] = t[v]
            did = True
    return G(out) if did else None


def _region_prop(cells, prop):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
    return {"size": len(cells), "height": h, "width": w, "parity": len(cells) % 2,
            "side_parity": w % 2, "is_square": int(h == w and h * w == len(cells))}[prop]


def _pos_class(i, n):
    return "first" if i == 0 else "last" if i == n - 1 else "mid"


def panel_paint(g, ctx, spec):
    """Panels between separator lines are painted by their position class (first / mid / last
    along each axis): table[(row class, col class)] is a colour or 'KEEP'."""
    t = dict(spec)
    sep = P.separator_lines(g)
    if not sep:
        return None
    col, rs, cs = sep
    H, W = shape(g)
    rows = P.runs_between(H, rs); cols = P.runs_between(W, cs)
    if len(rows) < 2 or len(cols) < 2:
        return None
    out = [list(x) for x in g]
    for a, (r0, r1) in enumerate(rows):
        for b, (c0, c1) in enumerate(cols):
            v = t.get((_pos_class(a, len(rows)), _pos_class(b, len(cols))))
            if v is None:
                return None
            if v == "KEEP":
                continue
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    out[r][c] = v
    return G(out)


def translate(g, ctx, shift, bg):
    """Shift the whole content by (dr, dc); vacated cells become background and content shifted
    past the edge is dropped."""
    dr, dc = shift
    H, W = shape(g)
    out = [[bg] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if g[r][c] != bg:
                a, b = r + dr, c + dc
                if 0 <= a < H and 0 <= b < W:
                    out[a][b] = g[r][c]
    res = G(out)
    return None if res == g else res


def _two_markers(g, bg):
    cells = [(r, c, v) for r, row in enumerate(g) for c, v in enumerate(row) if v != bg]
    return cells if len(cells) == 2 and cells[0][2] != cells[1][2] else None


def elbow_path(g, ctx, spec, bg):
    """Two single-cell markers of different colours are joined by an L-shaped path of `colour`
    (background cells only): from the `start` marker along its row first ('hv') or its column
    first ('vh'), then straight to the other marker."""
    start, mode, colour = spec
    m = _two_markers(g, bg)
    if not m:
        return None
    a = next((x for x in m if x[2] == start), None)
    if a is None:
        return None
    b = m[0] if m[1] is a else m[1]
    (ra, ca, _), (rb, cb, _) = a, b
    out = [list(x) for x in g]
    corner = (ra, cb) if mode == "hv" else (rb, ca)
    def seg(p, q):
        (r0, c0), (r1, c1) = p, q
        dr = (r1 > r0) - (r1 < r0); dc = (c1 > c0) - (c1 < c0)
        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr; c += dc
            if out[r][c] == bg:
                out[r][c] = colour
    seg((ra, ca), corner); seg(corner, (rb, cb))
    res = G(out)
    return None if res == g else res




def extend_to_marker(g, ctx, bg):
    """Every object (one colour, size > 1) that shares a row or column with a single-cell marker
    grows along that line through background until it touches the marker."""
    H, W = shape(g)
    objs = P.objects(g, bg=bg, conn=8, by_colour=True)
    markers = [o[0] for o in objs if len(o) == 1]
    bodies = [o for o in objs if len(o) > 1]
    if not markers or not bodies:
        return None
    out = [list(x) for x in g]
    did = False
    for o in bodies:
        col = g[o[0][0]][o[0][1]]
        cells = set(o)
        for mr, mc in markers:
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                # the object cell nearest the marker on the marker's line, looking from the marker
                r, c = mr - dr, mc - dc
                path = []
                while 0 <= r < H and 0 <= c < W and (r, c) not in cells and g[r][c] == bg:
                    path.append((r, c)); r -= dr; c -= dc
                if path and 0 <= r < H and 0 <= c < W and (r, c) in cells:
                    for a, b in path:
                        out[a][b] = col
                    did = True
    return G(out) if did else None


def keep_largest(g, ctx, conn, bg):
    """Erase every object except the (unique) largest one (objects of any colours, conn-connected)."""
    objs = P.objects(g, bg=bg, conn=conn, by_colour=False)
    if len(objs) < 2:
        return None
    sizes = sorted((len(o) for o in objs), reverse=True)
    if sizes[0] == sizes[1]:
        return None
    big = set(max(objs, key=len))
    return G([v if (r, c) in big or v == bg else bg for c, v in enumerate(row)] for r, row in enumerate(g))


def flip_objects(g, ctx, axis, bg):
    """Every object (8-connected, any colours) is mirrored inside its own bounding box
    ('h' left-right, 'v' up-down)."""
    objs = P.objects(g, bg=bg, conn=8, by_colour=False)
    if not objs:
        return None
    out = [list(x) for x in g]
    for o in objs:
        r0, c0, r1, c1 = P.bbox(o)
        for r, c in o:
            out[r][c] = bg
        for r, c in o:
            a, b = (r, c0 + c1 - c) if axis == "h" else (r0 + r1 - r, c)
            out[a][b] = g[r][c]
    res = G(out)
    return None if res == g else res


def hollow(g, ctx, bg):
    """Interior cells of every one-colour object (all in-grid 4-neighbours belong to the object)
    become background."""
    H, W = shape(g)
    objs = P.objects(g, bg=bg, conn=4, by_colour=True)
    out = [list(x) for x in g]
    did = False
    for o in objs:
        s_ = set(o)
        for r, c in o:
            nb = [(r + a, c + b) for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0)) if 0 <= r + a < H and 0 <= c + b < W]
            if nb and all(q in s_ for q in nb):
                out[r][c] = bg; did = True
    return G(out) if did else None


def shift_objects(g, ctx, spec, bg):
    """Selected objects (4-connected, multicolour: the largest, the smallest, or all) move by
    (dr, dc); the rest stays. Moving out of the grid or onto other ink makes the op inapplicable."""
    selector, dr, dc = spec
    objs = P.objects(g, bg=bg, conn=4, by_colour=False)
    if not objs:
        return None
    if selector in ("largest", "smallest"):
        sizes = sorted(len(o) for o in objs)
        pick = sizes[-1] if selector == "largest" else sizes[0]
        sel = [o for o in objs if len(o) == pick]
        if len(sel) != 1:
            return None
    else:
        sel = objs
    H, W = shape(g)
    out = [list(x) for x in g]
    for o in sel:
        for r, c in o:
            out[r][c] = bg
    for o in sel:
        for r, c in o:
            a, b = r + dr, c + dc
            if not (0 <= a < H and 0 <= b < W) or out[a][b] not in (bg, g[r][c]):
                return None
            out[a][b] = g[r][c]
    res = G(out)
    return None if res == g else res


# ------------------------------------------------------------------ registry
# name -> (function, [(param, inference rule)], cost)
REGISTRY = {
    "rot90": (rot90, [], 1), "rot180": (rot180, [], 1), "rot270": (rot270, [], 1),
    "flip_h": (flip_h, [], 1), "flip_v": (flip_v, [], 1), "transpose": (transpose, [], 1),
    "crop_content": (crop_content, [("bg", "background")], 2),
    "crop_bbox_of": (crop_bbox_of, [("colour", "input_rectangle_colour")], 2),
    "swap_colours": (swap_colours, [], 1),
    "recolor": (recolor, [("mapping", "colour_map")], 3),
    "pool": (pool, [("block", "shrink_ratio"), ("reducer", "reducer"), ("bg", "background")], 3),
    "pool_cells": (pool_cells, [("reducer", "reducer"), ("bg", "background")], 3),
    "dedup_rows": (dedup_rows, [], 1), "dedup_cols": (dedup_cols, [], 1),
    "tile": (tile, [("reps", "grow_ratio")], 2),
    "arrange": (arrange, [("layout", "transform_layout")], 2),
    "panel_logic": (panel_logic, [("spec", "panel_truth_table")], 3),
    "count_render": (count_render, [("spec", "count_spec")], 3),
    "stencil": (stencil, [("spec", "stencil_spec")], 3),
    "colour_histogram": (colour_histogram, [("spec", "histogram_spec")], 3),
    "recolour_by_property": (recolour_by_property, [("spec", "property_recolour_spec")], 3),
    "object_rule": (object_rule, [("spec", "object_rule_spec")], 3),
    "to_boundary": (to_boundary, [("spec", "boundary_spec")], 3),
    "layers": (layers, [("spec", "layers_spec")], 3),
    "copy_to_markers": (copy_to_markers, [("spec", "copy_markers_spec")], 3),
    "box_objects": (box_objects, [("conn", "connectivity"), ("mode", "object_box_mode"), ("ink", "changed_colour"),
                                  ("bg", "background")], 2),
    "scale_up": (scale_up, [("factor", "grow_ratio")], 2),
    "substitute_cells": (substitute_cells, [("table", "cell_tile_table")], 4),
    "self_kron": (self_kron, [("selector", "kron_selector"), ("bg", "background")], 3),
    "pad": (pad, [("width", "pad_width"), ("mode", "pad_mode"), ("corner", "output_colour")], 3),
    "canvas": (canvas, [("size", "grow_ratio"), ("bg", "background")], 2),
    "rays": (rays, [("direction", "diagonal_direction"), ("bg", "background")], 2),
    "extend_period": (extend_period, [("axis", "grown_axis"), ("ratio", "axis_ratio")], 2),
    "repair_period": (repair_period, [("hole", "hole_colour")], 3),
    "fill_bbox_by_colour": (fill_bbox_by_colour, [("bg", "background"), ("mode", "box_mode")], 2),
    "marker_rays": (marker_rays, [("direction", "diagonal_direction"), ("pattern", "ray_pattern"),
                                  ("ink", "changed_colour"), ("bg", "background")], 3),
    "connect_to_block": (connect_to_block, [("bg", "background")], 2),
    "corner_rays": (corner_rays, [("kind", "corner_kind"), ("colour", "ray_colour_source"),
                                  ("which", "shape_scope"), ("bg", "background")], 3),
    "top_left_quadrant": (top_left_quadrant, [], 2),
    "crop_window": (crop_window, [("spec", "window_spec")], 2),
    "rings": (rings, [("mode", "ring_mode")], 2),
    "colour_order": (colour_order, [("spec", "order_spec")], 2),
    "fill_touching": (fill_touching, [("colour", "changed_colour"), ("conn", "connectivity"), ("bg", "background")], 2),
    "mirror_about": (mirror_about, [("mode", "mirror_mode"), ("bg", "background")], 2),
    "border_lines": (border_lines, [("colour", "pair_colour"), ("bg", "background")], 2),
    "fill_largest_rect": (fill_largest_rect, [("colour", "changed_colour"), ("bg", "background")], 2),
    "project_markers": (project_markers, [("bg", "background")], 2),
    "marker_stripes": (marker_stripes, [("bg", "background")], 2),
    "role_recolour": (role_recolour, [("spec", "role_recolour_spec")], 2),
    "line_uniform": (line_uniform, [("spec", "line_uniform_spec")], 2),
    "per_region": (per_region, [("spec", "region_programs")], 3),
    "where": (where, [("spec", "none")], 3),
    "obj_recolour": (obj_recolour, [("spec", "obj_recolour_rule")], 2),
    "obj_move": (obj_move, [("spec", "obj_move_rule")], 2),
    "obj_rays": (obj_rays, [("spec", "obj_rays_rule")], 2),
    "obj_connect": (obj_connect, [("spec", "obj_connect_rule")], 2),
    "obj_map": (obj_map, [("spec", "obj_map_rule")], 2),
    "fill_gaps": (fill_gaps, [("axis", "gap_axis"), ("colour", "changed_colour"), ("bg", "background")], 2),
    "edge_pair_lines": (edge_pair_lines, [("bg", "background")], 2),
    "fill_pairs": (fill_pairs, [("axis", "hv_axis"), ("bg", "background")], 2),
    "fill_enclosed": (fill_enclosed, [("colour", "changed_colour"), ("conn", "connectivity"), ("bg", "background"),
                                      ("outside", "outside_colour")], 2),
    "symmetrize": (symmetrize, [("mode", "symmetry_mode"), ("bg", "background")], 2),
    "frame": (frame, [("colour", "changed_colour")], 2),
    "recolour_by_contact": (recolour_by_contact, [("mode", "contact_mode"), ("bg", "background")], 2),
    "recolour_to_marker": (recolour_to_marker, [("bg", "background")], 2),
    "panel_reduce": (panel_reduce, [("spec", "panel_reduce_spec")], 3),
    "recolour_by_position": (recolour_by_position, [("spec", "position_spec"), ("bg", "background")], 2),
    "fill_holes_by_property": (fill_holes_by_property, [("spec", "hole_property_spec"), ("bg", "background")], 3),
    "panel_paint": (panel_paint, [("spec", "panel_paint_spec")], 3),
    "translate": (translate, [("shift", "content_shift"), ("bg", "background")], 2),
    "elbow_path": (elbow_path, [("spec", "elbow_spec"), ("bg", "background")], 2),
    "extend_to_marker": (extend_to_marker, [("bg", "background")], 2),
    "keep_largest": (keep_largest, [("conn", "connectivity"), ("bg", "background")], 2),
    "flip_objects": (flip_objects, [("axis", "flip_axis"), ("bg", "background")], 2),
    "hollow": (hollow, [("bg", "background")], 2),
    "shift_objects": (shift_objects, [("spec", "object_shift"), ("bg", "background")], 2),
    "gravity": (gravity, [("direction", "gravity_direction"), ("bg", "background")], 2),
    "interior_by_property": (interior_by_property, [("spec", "interior_spec")], 3),
    "connect_points": (connect_points, [("spec", "line_spec")], 3),
    "move_objects": (move_objects, [("spec", "motion_spec")], 3),
    "select_objects": (select_objects, [("view", "object_view"), ("selector", "object_selector"),
                                        ("mode", "selection_mode")], 2),
    "extend_lines": (extend_lines, [("bg", "background"), ("cross", "changed_colour")], 2),
    "apply_key": (apply_key, [("spec", "key_spec")], 3),
    "complete_copies": (complete_copies, [("spec", "stamp_spec")], 3),
}


# ------------------------------------------------------------------ engine operations (OPT198)
def _engine_op(name):
    def op(g, ctx, spec):
        from .engines import apply
        return apply(name, g, spec)
    op.__name__ = name
    op.__doc__ = "Engine operation (see dsl/engines.py)."
    return op


from .engines import ENGINES as _ENGINES  # noqa: E402
for _name in _ENGINES:
    REGISTRY[_name] = (_engine_op(_name), [("spec", _name + "_spec")], 3)
