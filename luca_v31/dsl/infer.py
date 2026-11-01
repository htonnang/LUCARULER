"""LUCA DSL v1 - parameter inference rules.

rule(xs, ys) -> list of candidate values
  xs: the current grids (one per demonstration, after any earlier program steps)
  ys: the demonstration outputs
A value is only a candidate; the search keeps it only if the whole program reproduces every
demonstration. No rule returns a value that is not derived from the demonstrations
(or from a small fixed vocabulary such as directions and reducer names).
"""
from collections import Counter
from math import gcd
from .ops import shape, colours, most_common, SELECTORS


def _common_colours(gs):
    s = set.intersection(*[colours(g) for g in gs])
    freq = Counter(v for g in gs for r in g for v in r)
    return sorted(s, key=lambda c: -freq[c])


def background(xs, ys, inputs=None):
    # OPT196: the background is decided once per task - when every demonstration input has the same
    # most frequent colour, that literal colour comes first and is applied to the test input as is
    # (a per-grid 'auto' can flip on a test input dominated by one large object)
    from .ops import most_common as _mc
    tops = {_mc(x) for x in xs}
    out = ["task:%d" % tops.pop(), "auto"] if len(tops) == 1 else ["auto"]
    for c in _common_colours(xs)[:2] + _common_colours(ys)[:2]:
        if c not in out:
            out.append(c)
    return out


def output_colour(xs, ys, inputs=None):
    return _common_colours(ys)[:4]


def rectangle_colour(xs, ys, inputs=None):
    """Colours that occupy one filled rectangle in every grid (holes, windows, markers)."""
    out = []
    for c in _common_colours(xs):
        good = True
        for g in xs:
            cells = [(r, k) for r, row in enumerate(g) for k, v in enumerate(row) if v == c]
            r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
            c0, c1 = min(k for _, k in cells), max(k for _, k in cells)
            if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1) or len(cells) == len(g) * len(g[0]):
                good = False; break
        if good:
            out.append(c)
    return out


def input_rectangle_colour(xs, ys, inputs=None):
    """rectangle_colour measured on the ORIGINAL inputs (for steps that read ctx.input)."""
    return rectangle_colour(list(inputs or xs), ys)


def _confirmed(keys, xs):
    """Evidence rule: every learned entry must be seen in at least two demonstrations."""
    from .evidence import enough
    seen = Counter(k for g in xs for k in colours(g))
    cells = Counter(v for g in xs for r in g for v in r)
    return len(xs) >= 2 and all(enough(seen[k], cells[k]) for k in keys)


def colour_map(xs, ys, inputs=None):
    if any(shape(x) != shape(y) for x, y in zip(xs, ys)):
        return []
    m = {}
    for x, y in zip(xs, ys):
        for rx, ry in zip(x, y):
            for a, b in zip(rx, ry):
                if m.setdefault(a, b) != b:
                    return []
    m = {a: b for a, b in m.items() if a != b}
    out = [tuple(sorted(m.items()))] if m and _confirmed(m, xs) else []
    # OPT188: role mapping - all ink (every non-background colour) becomes one colour; the ink's own
    # colours may differ per demonstration
    from . import perception as P
    targets = set()
    for x, y in zip(xs, ys):
        bg = P.background(x, "zero_if_present")
        for rx, ry in zip(x, y):
            for a, b in zip(rx, ry):
                if a == bg:
                    if b != a:
                        return out
                else:
                    targets.add(b)
        if bg in targets:
            return out
    if len(targets) == 1:
        c = targets.pop()
        inks = {v for x in xs for r in x for v in r} - {P.background(x, "zero_if_present") for x in xs}
        if inks - {c} and len(xs) >= 2:
            if len(inks - {c}) >= 2:
                return [(("INK", c),)]   # several ink colours, one target: the role subsumes the literal map
            out.append((("INK", c),))
    return out


def _ratios(xs, ys, grow):
    rs = set()
    for x, y in zip(xs, ys):
        (hx, wx), (hy, wy) = shape(x), shape(y)
        if grow:
            if hy % hx or wy % wx:
                return None
            rs.add((hy // hx, wy // wx))
        else:
            if hx % hy or wx % wy:
                return None
            rs.add((hx // hy, wx // wy))
    return rs


def grow_ratio(xs, ys, inputs=None):
    rs = _ratios(xs, ys, True)
    if rs and len(rs) > 1:
        # OPT196: the ratio differs per demonstration - it may be the number of colours (role 'NCOL')
        if all((hy // hx) == (wy // wx) == max(1, len(colours(x) - {most_common(x)}))
               for x, y in zip(xs, ys) for (hx, wx), (hy, wy) in [(shape(x), shape(y))]):
            return ["NCOL"]
        return []
    if not rs or len(rs) != 1 or rs == {(1, 1)}:
        return []
    out = sorted(rs)
    a, b = out[0]
    if a == b and all(a == len(colours(x) - {most_common(x)}) for x in xs):
        out.append("NCOL")   # the same ratio is also the colour count: offer the role as an alternative
    return out


def shrink_ratio(xs, ys, inputs=None):
    out = []
    rs = _ratios(xs, ys, False)
    if rs and len(rs) == 1 and rs != {(1, 1)}:
        a, b = next(iter(rs)); out.append(("factor", a, b))
    shapes = {shape(y) for y in ys}
    if len(shapes) == 1:
        oh, ow = next(iter(shapes)); out.append(("cells", oh, ow))
    return out


def reducer(xs, ys, inputs=None):
    return ["majority", "uniform", "only_non_bg", "any_non_bg"]


def cell_tile_table(xs, ys, inputs=None):
    rs = _ratios(xs, ys, True)
    if not rs or len(rs) != 1 or rs == {(1, 1)}:
        return []
    a, b = next(iter(rs)); table = {}
    for x, y in zip(xs, ys):
        for r, row in enumerate(x):
            for c, v in enumerate(row):
                t = tuple(tuple(z[c * b:(c + 1) * b]) for z in y[r * a:(r + 1) * a])
                if table.setdefault(v, t) != t:
                    return []
    return [tuple(sorted(table.items()))] if _confirmed(table, xs) else []


def kron_selector(xs, ys, inputs=None):
    return sorted(SELECTORS)


def pad_width(xs, ys, inputs=None):
    ks = set()
    for x, y in zip(xs, ys):
        (hx, wx), (hy, wy) = shape(x), shape(y)
        if hy - hx != wy - wx or (hy - hx) % 2 or hy <= hx:
            return []
        ks.add((hy - hx) // 2)
    return sorted(ks) if len(ks) == 1 else []


def pad_mode(xs, ys, inputs=None):
    return ["edge", "const"]


def diagonal_direction(xs, ys, inputs=None):
    return [(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 1), (1, 0), (0, -1), (-1, 0)]


def grown_axis(xs, ys, inputs=None):
    axes = set()
    for x, y in zip(xs, ys):
        (hx, wx), (hy, wy) = shape(x), shape(y)
        if hy > hx and wy == wx:
            axes.add(0)
        elif wy > wx and hy == hx:
            axes.add(1)
        else:
            return []
    return sorted(axes) if len(axes) == 1 else []


def axis_ratio(xs, ys, inputs=None):
    ax = grown_axis(xs, ys)
    if not ax:
        return []
    rs = set()
    for x, y in zip(xs, ys):
        u, v = (len(y), len(x)) if ax[0] == 0 else (len(y[0]), len(x[0]))
        d = gcd(u, v); rs.add((u // d, v // d))
    return sorted(rs) if len(rs) == 1 else []


def changed_colour(xs, ys, inputs=None):
    """Colours written into cells that change, in every demonstration."""
    if any(shape(x) != shape(y) for x, y in zip(xs, ys)):
        return []
    sets = [{b for rx, ry in zip(x, y) for a, b in zip(rx, ry) if a != b} for x, y in zip(xs, ys)]
    common = set.intersection(*sets) if sets else set()
    return sorted(common)[:3]


def box_mode(xs, ys, inputs=None):
    return ["fill", "outline"]


def ray_pattern(xs, ys, inputs=None):
    return ["solid", "alternate"]


def corner_kind(xs, ys, inputs=None):
    return ["tip", "missing_corner", "bbox_corner"]


def ray_colour_source(xs, ys, inputs=None):
    return ["own", "other"]


def shape_scope(xs, ys, inputs=None):
    return ["all", "largest", "hollow"]


def object_view(xs, ys, inputs=None):
    # OPT199: when every demonstration input has the same background, that literal colour comes
    # first (a per-grid background can flip on a test input dominated by its objects)
    from . import perception as P
    tops = {P.backgrounds(x, 1, "weighted")[0] for x in xs}
    ranks = ["bg%d" % tops.pop() if len(tops) == 1 else 0, 1]   # identical to rank 0 on the demos
    return [(rank, conn, multi) for rank in ranks for conn in (4, 8) for multi in (False, True)]


def object_selector(xs, ys, inputs=None):
    by_colour = [("most_of_colour", c) for c in _common_colours(list(inputs or xs))[1:5]]
    return by_colour + ["largest", "smallest", "unique_shape", "repeated_shape", "rare_colour", "common_colour",
            "tallest", "shortest", "widest", "narrowest", "most_colours", "fewest_colours", "densest", "sparsest",
            "most_marks", "fewest_marks", "most_inside", "most_rare_colour", "most_common_shape", "least_common_shape"]


def selection_mode(xs, ys, inputs=None):
    same = all(shape(x) == shape(y) for x, y in zip(xs, ys))
    return ["keep", "delete"] if same else ["crop", "crop_context", "crop_inside"]


def line_spec(xs, ys, inputs=None):
    """Settings of the line grammar that reproduce every demonstration (its own fitter)."""
    from ..line_relations import fit
    try:
        laws = fit(list(zip(xs, ys)), seconds=1.5, limit=1800)
    except Exception:
        return []
    return [tuple(sorted(l.program_spec["params"].items())) for l in laws[:6]]


def motion_spec(xs, ys, inputs=None):
    """Settings of the motion grammar that reproduce every demonstration (its own fitter)."""
    from ..relative_dir import induce_all
    try:
        return list(induce_all(list(zip(xs, ys)), seconds=4.0, limit=8))
    except Exception:
        return []


def key_spec(xs, ys, inputs=None):
    """Key-reading settings (dsl/keys.py) that reproduce every demonstration (the search only
    offers apply_key as a first step, so xs are the raw inputs)."""
    from .keys import fit
    return fit(xs, ys)


def stamp_spec(xs, ys, inputs=None):
    """Exemplar-completion settings (dsl/stamp.py) that reproduce every demonstration."""
    from .stamp import fit
    return fit(xs, ys)


def line_uniform_spec(xs, ys, inputs=None):
    """(axis, yes, no) such that every output line is `yes` exactly where the input line is uniform."""
    if any(shape(x) != shape(y) for x, y in zip(xs, ys)):
        return []
    out = []
    for axis in ("h", "v"):
        yes, no, ok = set(), set(), True
        for x, y in zip(xs, ys):
            X, Y = (x, y) if axis == "h" else (tuple(zip(*x)), tuple(zip(*y)))
            for rx, ry in zip(X, Y):
                if len(set(ry)) != 1:
                    ok = False; break
                (yes if len(set(rx)) == 1 else no).add(ry[0])
            if not ok:
                break
        if ok and len(yes) == 1 and len(no) == 1 and yes != no:
            out.append((axis, yes.pop(), no.pop()))
    return out


def role_recolour_spec(xs, ys, inputs=None):
    """(k, new_k, new_x) for role_recolour: every input has exactly two colours, k shared by all,
    the other colour X varying (so a literal mapping cannot express it)."""
    if any(shape(x) != shape(y) for x, y in zip(xs, ys)):
        return []
    cs = [sorted(colours(x)) for x in xs]
    if any(len(c) != 2 for c in cs):
        return []
    shared = set(cs[0]).intersection(*map(set, cs[1:]))
    out = []
    for k in sorted(shared):
        others = {c[0] if c[1] == k else c[1] for c in cs}
        if len(others) < 2:
            continue
        nks, nxs = None, None
        for x, y, c in zip(xs, ys, cs):
            X = c[0] if c[1] == k else c[1]
            ky = {y[r][q] for r in range(len(x)) for q in range(len(x[0])) if x[r][q] == k}
            xy = {y[r][q] for r in range(len(x)) for q in range(len(x[0])) if x[r][q] == X}
            if len(ky) != 1 or len(xy) != 1:
                nks = set(); break
            a, b = ky.pop(), xy.pop()
            ra = {a} | ({"X"} if a == X else set()) | ({"K"} if a == k else set())
            rb = {b} | ({"X"} if b == X else set()) | ({"K"} if b == k else set())
            nks = ra if nks is None else nks & ra
            nxs = rb if nxs is None else nxs & rb
        if not nks or not nxs:
            continue
        pick = lambda st: sorted(st, key=lambda v: (not isinstance(v, str), str(v)))[0]
        spec = (k, pick(nks), pick(nxs))
        if spec[1:] != ("K", "X"):
            out.append(spec)
    return out


def obj_recolour_rule(xs, ys, inputs=None):
    from .objlearn import fit_recolour
    return fit_recolour(list(xs), list(ys), background(xs, ys)[0])


def obj_move_rule(xs, ys, inputs=None):
    from .objlearn import fit_move
    return fit_move(list(xs), list(ys), background(xs, ys)[0])


def obj_rays_rule(xs, ys, inputs=None):
    from .objlearn import fit_rays
    return fit_rays(list(xs), list(ys), background(xs, ys)[0])


def obj_connect_rule(xs, ys, inputs=None):
    from .objlearn import fit_connect
    return fit_connect(list(xs), list(ys), background(xs, ys)[0])


def obj_map_rule(xs, ys, inputs=None):
    from .objmap import fit
    return fit(list(xs), list(ys), background(xs, ys)[0])


def none(xs, ys, inputs=None):
    return []   # parameters built by a composer, never inferred


def region_programs(xs, ys, inputs=None):
    """Lifted sub-programs (dsl/lift.py), found by the DSL search on region pairs."""
    from .lift import fit
    return fit(list(xs), list(ys), background(xs, ys)[0])


def gap_axis(xs, ys, inputs=None):
    return ["h", "v", "both"]


def hv_axis(xs, ys, inputs=None):
    return ["h", "v"]


def outside_colour(xs, ys, inputs=None):
    """'keep', then the one colour that every changed cell on the grid border takes (fill_enclosed)."""
    out = ["keep"]
    cols = set()
    for x, y in zip(xs, ys):
        if shape(x) != shape(y):
            return out
        H, W = shape(x)
        cols |= {y[r][c] for r in range(H) for c in range(W)
                 if (r in (0, H - 1) or c in (0, W - 1)) and y[r][c] != x[r][c]}
    if len(cols) == 1:
        out.append(cols.pop())
    return out


def transform_layout(xs, ys, inputs=None):
    """The k x m layout of D8 transforms whose blocks reproduce every output (same in all demos).
    OPT199: each block keeps every transform that matches, intersected across the demonstrations
    (a symmetric input matches several), and the first remaining one in _D8 order is used."""
    from .ops import _D8
    found, dims = None, None
    for x, y in zip(xs, ys):
        (h, w), (H, W) = shape(x), shape(y)
        if H % h or W % w or (H, W) == (h, w):
            return []
        if dims is not None and dims != (H // h, W // w):
            return []
        dims = (H // h, W // w)
        lay = []
        for i in range(H // h):
            for j in range(W // w):
                blk = tuple(tuple(r[j * w:(j + 1) * w]) for r in y[i * h:(i + 1) * h])
                lay.append({t for t, f in _D8.items() if tuple(map(tuple, f(x))) == blk})
        found = lay if found is None else [a & b for a, b in zip(found, lay)]
        if any(not a for a in found):
            return []
    if not found:
        return []
    order = list(_D8)
    pick = [min(a, key=order.index) for a in found]
    return [tuple(tuple(pick[i * dims[1]:(i + 1) * dims[1]]) for i in range(dims[0]))]


def panel_truth_table(xs, ys, inputs=None):
    """Panel-logic specs (see ops.panel_logic) that reproduce every demonstration.
    'table' entries must each be seen in >= 2 demonstrations (evidence rule)."""
    from itertools import permutations
    from .ops import _split_panels, panel_logic
    from . import perception as P
    shapes = {shape(y) for y in ys}
    if len(xs) < 2:
        return []
    if len(shapes) == 1:
        size = shapes.pop()
        splits = [_split_panels(x, size) for x in xs]
    else:
        # OPT187: the panel size may differ per demonstration (it is read from each grid)
        size = "halves"
        splits = [_split_panels(x, size) for x in xs]
        if any(s is None or (len(s[0]), len(s[0][0])) != shape(y) for s, y in zip(splits, ys)):
            return []
    if any(s is None for s in splits) or len({len(s) for s in splits}) != 1:
        return []
    k = len(splits[0])
    out = []
    from .ops import _panel_bg, BOOL
    ycols = _common_colours(ys)
    for bgpol in ("zero_if_present", "most_common", "panel"):
        pbgs = [_panel_bg(pans, x, bgpol) for pans, x in zip(splits, xs)]
        if bgpol == "panel" and all(len(set(b)) == 1 for b in pbgs):
            continue   # same as a grid-level policy
        table, seen, ok = {}, Counter(), True
        occ = Counter()
        for pans, y, pb in zip(splits, ys, pbgs):
            keys = set()
            sz = size if size != "halves" else (len(pans[0]), len(pans[0][0]))
            for r in range(sz[0]):
                for c in range(sz[1]):
                    key = tuple(int(p[r][c] != b) for p, b in zip(pans, pb))
                    if table.setdefault(key, y[r][c]) != y[r][c]:
                        ok = False; break
                    keys.add(key); occ[key] += 1
                if not ok:
                    break
            if not ok:
                break
            for kk in keys:
                seen[kk] += 1
        from .evidence import enough
        if ok and all(enough(v, occ[k]) for k, v in seen.items()) and len(set(table.values())) >= 2:
            out.append((size, bgpol, "table", tuple(sorted(table.items()))))
        elif ok and len(set(table.values())) == 2:
            # OPT199: one boolean operator with an ink and a blank colour (no per-entry evidence needed)
            for op in BOOL:
                for ink in ycols[:3]:
                    for blank in ycols[:3]:
                        if blank == ink:
                            continue
                        spec = (size, bgpol, "bool", (op, ink, blank))
                        if all(panel_logic(x, None, spec) == y for x, y in zip(xs, ys)):
                            out.append(spec); break
                    else:
                        continue
                    break
        if k <= 4:
            for order in permutations(range(k)):
                spec = (size, bgpol, "priority", order)
                if all(panel_logic(x, None, spec) == y for x, y in zip(xs, ys)):
                    out.append(spec); break
    return out


def count_spec(xs, ys, inputs=None):
    """Count-and-draw settings (dsl/count_render.py) that reproduce every demonstration."""
    from .count_render import fit
    return fit(xs, ys)


def stencil_spec(xs, ys, inputs=None):
    """Colour-stencil settings (dsl/stencil.py) that reproduce every demonstration."""
    from .stencil import fit
    return fit(xs, ys)


def connectivity(xs, ys, inputs=None):
    return [8, 4]


def object_box_mode(xs, ys, inputs=None):
    return ["fill", "ring"]


def histogram_spec(xs, ys, inputs=None):
    """Histogram layouts (dsl/histogram.py) that reproduce every demonstration."""
    from .histogram import fit
    return fit(xs, ys)


def property_recolour_spec(xs, ys, inputs=None):
    """Object-property recolour tables (dsl/prop_recolour.py) that reproduce every demonstration."""
    from .prop_recolour import fit
    return fit(xs, ys)


def object_rule_spec(xs, ys, inputs=None):
    """Object property -> recolour / erase tables (dsl/object_rules.py) reproducing every demonstration."""
    from .object_rules import fit
    return fit(xs, ys)


def boundary_spec(xs, ys, inputs=None):
    """Move / extend / clean / centre settings (dsl/boundary.py) reproducing every demonstration."""
    from .boundary import fit
    return fit(xs, ys)


def layers_spec(xs, ys, inputs=None):
    """Layer-order settings (dsl/layers.py) reproducing every demonstration."""
    from .layers import fit
    return fit(xs, ys)


def copy_markers_spec(xs, ys, inputs=None):
    """Template-to-marker settings (dsl/copy_markers.py) reproducing every demonstration."""
    from .copy_markers import fit
    return fit(xs, ys)


def window_spec(xs, ys, inputs=None):
    """Corner / centre windows of the (constant) output size that reproduce every demonstration."""
    from .ops import crop_window
    shapes = {shape(y) for y in ys}
    if len(shapes) != 1:
        return []
    h, w = shapes.pop()
    return [("" + a, h, w) for a in ("tl", "tr", "bl", "br", "c")
            if all(crop_window(x, None, (a, h, w)) == y for x, y in zip(xs, ys))]


def ring_mode(xs, ys, inputs=None):
    from .ops import rings
    if any(shape(x) != shape(y) for x, y in zip(xs, ys)):
        return []
    return [m for m in ("reverse", "inward", "outward") if all(rings(x, None, m) == y for x, y in zip(xs, ys))]


def order_spec(xs, ys, inputs=None):
    from .ops import colour_order
    if any(min(shape(y)) != 1 for y in ys):
        return []
    return [s for s in (("auto", "none"), ("auto", "bg"), ("rows", "none"), ("cols", "none"), ("rows", "bg"),
                        ("cols", "bg")) if all(colour_order(x, None, s) == y for x, y in zip(xs, ys))][:2]


def mirror_mode(xs, ys, inputs=None):
    return ["both", "h", "v", "h_move", "v_move", "both_move"]


def pair_colour(xs, ys, inputs=None):
    """A literal line colour shared by all outputs, else the role 'PAIR'."""
    return output_colour(xs, ys)[:2] + ["PAIR"]


def interior_spec(xs, ys, inputs=None):
    from .prop_recolour import fit_interior
    return fit_interior(xs, ys)


def gravity_direction(xs, ys, inputs=None):
    return ["down", "up", "left", "right"]


def symmetry_mode(xs, ys, inputs=None):
    return ["h", "v", "both", "r180", "tr"]


def contact_mode(xs, ys, inputs=None):
    return ["keep", "erase", "erase_all"]


def panel_reduce_spec(xs, ys, inputs=None):
    """count -> colour table for panel_reduce, learned from the demonstrations (evidence rule)."""
    from . import perception as P
    from .evidence import enough
    table, seen, occ = {}, {}, Counter()
    for i, (x, y) in enumerate(zip(xs, ys)):
        sep = P.separator_lines(x)
        if not sep:
            return []
        col, rs, cs = sep
        rows = P.runs_between(len(x), rs); cols = P.runs_between(len(x[0]), cs)
        if (len(rows), len(cols)) != shape(y):
            return []
        flat = [v for r in x for v in r if v != col]
        pbg = Counter(flat).most_common(1)[0][0]
        for a, (r0, r1) in enumerate(rows):
            for b, (c0, c1) in enumerate(cols):
                n = sum(1 for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if x[r][c] not in (col, pbg))
                v = "BG" if y[a][b] == pbg else y[a][b]
                if table.setdefault(n, v) != v:
                    return []
                seen.setdefault(n, set()).add(i); occ[n] += 1
    if len(set(table.values())) < 2:
        return []
    # values seen in one demonstration only are kept when they agree with their neighbours' colour
    ks = sorted(table)
    if not all(enough(seen[k], occ[k]) or any(table.get(j) == table[k] for j in ks if j != k) for k in ks):
        return []
    return [tuple(sorted(table.items(), key=repr))]


def position_spec(xs, ys, inputs=None):
    """(axis, k, phase, colour) such that exactly the ink cells at index = phase mod k change to colour."""
    from . import perception as P
    if any(shape(x) != shape(y) for x, y in zip(xs, ys)):
        return []
    cols = changed_colour(xs, ys)
    if len(cols) != 1:
        return []
    col = cols[0]
    out = []
    for axis in ("c", "r"):
        for k in (2, 3, 4):
            for phase in range(k):
                ok = True
                for x, y in zip(xs, ys):
                    bg = P.background(x, "zero_if_present")
                    for r in range(len(x)):
                        for c in range(len(x[0])):
                            i = r if axis == "r" else c
                            want = col if (x[r][c] != bg and i % k == phase) else x[r][c]
                            if y[r][c] != want:
                                ok = False; break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    out.append((axis, k, phase, col))
    return out[:2]


def hole_property_spec(xs, ys, inputs=None):
    from . import perception as P
    from .ops import _region_prop
    from .evidence import enough
    if any(shape(x) != shape(y) for x, y in zip(xs, ys)):
        return []
    out = []
    for prop in ("side_parity", "parity", "size", "width", "height", "is_square"):
        table, seen, occ, ok = {}, {}, Counter(), True
        for i, (x, y) in enumerate(zip(xs, ys)):
            bg = P.background(x, "zero_if_present")
            H, W = len(x), len(x[0]); vis = set()
            for r in range(H):
                for c in range(W):
                    if x[r][c] != bg or (r, c) in vis:
                        continue
                    comp, st, edge = [], [(r, c)], False
                    vis.add((r, c))
                    while st:
                        a, b = st.pop(); comp.append((a, b))
                        if a in (0, H - 1) or b in (0, W - 1):
                            edge = True
                        for da, db in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                            p_, q_ = a + da, b + db
                            if 0 <= p_ < H and 0 <= q_ < W and x[p_][q_] == bg and (p_, q_) not in vis:
                                vis.add((p_, q_)); st.append((p_, q_))
                    if edge:
                        continue
                    news = {y[a][b] for a, b in comp}
                    if len(news) != 1:
                        ok = False; break
                    v, n = _region_prop(comp, prop), news.pop()
                    if table.setdefault(v, n) != n:
                        ok = False; break
                    seen.setdefault(v, set()).add(i); occ[v] += 1
                if not ok:
                    break
            if not ok:
                break
        if ok and len(set(table.values())) >= 2 and all(enough(seen[k], occ[k]) for k in table):
            out.append((prop, tuple(sorted(table.items()))))
    return out[:2]


def panel_paint_spec(xs, ys, inputs=None):
    from . import perception as P
    from .ops import _pos_class
    from .evidence import enough
    if any(shape(x) != shape(y) for x, y in zip(xs, ys)):
        return []
    table, seen, occ = {}, {}, Counter()
    for i, (x, y) in enumerate(zip(xs, ys)):
        sep = P.separator_lines(x)
        if not sep:
            return []
        col, rs, cs = sep
        rows = P.runs_between(len(x), rs); cols = P.runs_between(len(x[0]), cs)
        if len(rows) < 2 or len(cols) < 2:
            return []
        for a, (r0, r1) in enumerate(rows):
            for b, (c0, c1) in enumerate(cols):
                cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
                if all(y[r][c] == x[r][c] for r, c in cells):
                    v = "KEEP"
                else:
                    vs = {y[r][c] for r, c in cells}
                    if len(vs) != 1:
                        return []
                    v = vs.pop()
                k = (_pos_class(a, len(rows)), _pos_class(b, len(cols)))
                if table.setdefault(k, v) != v:
                    return []
                seen.setdefault(k, set()).add(i); occ[k] += 1
    if all(v == "KEEP" for v in table.values()) or not all(enough(seen[k], occ[k]) for k in table):
        return []
    return [tuple(sorted(table.items(), key=repr))]


def content_shift(xs, ys, inputs=None):
    """Shifts (dr, dc) with |dr|, |dc| <= 4 that map every input onto its output."""
    from .ops import translate
    from . import perception as P
    if any(shape(x) != shape(y) for x, y in zip(xs, ys)):
        return []
    out = []
    for dr in range(-4, 5):
        for dc in range(-4, 5):
            if (dr or dc) and all(translate(x, None, (dr, dc), P.background(x, "most_common")) == y
                                  for x, y in zip(xs, ys)):
                out.append((dr, dc))
    return out[:2]


def hole_colour(xs, ys, inputs=None):
    """Colours marking holes: every cell that changes had this colour in the input (any layout),
    plus the colours that form one filled rectangle (the original rule)."""
    out = list(rectangle_colour(xs, ys))
    if all(shape(x) == shape(y) for x, y in zip(xs, ys)):
        src = set()
        for x, y in zip(xs, ys):
            src |= {a for rx, ry in zip(x, y) for a, b in zip(rx, ry) if a != b}
        if len(src) == 1 and src.pop() not in out:
            out.append(next(iter({a for x, y in zip(xs, ys) for rx, ry in zip(x, y) for a, b in zip(rx, ry) if a != b})))
    return out


def input_colour(xs, ys, inputs=None):
    """Non-background colours present in every input."""
    from . import perception as P
    cs = set.intersection(*({v for r in x for v in r} - {P.background(x, "most_common")} for x in xs))
    return sorted(cs)[:3]


def elbow_spec(xs, ys, inputs=None):
    from .ops import elbow_path
    from . import perception as P
    cols = changed_colour(xs, ys)
    out = []
    for start in input_colour(xs, ys):
        for mode in ("hv", "vh"):
            for col in cols:
                sp = (start, mode, col)
                if all(elbow_path(x, None, sp, P.background(x, "most_common")) == y for x, y in zip(xs, ys)):
                    out.append(sp)
    return out[:2]


def flip_axis(xs, ys, inputs=None):
    return ["h", "v"]


def object_shift(xs, ys, inputs=None):
    """(selector, dr, dc) with |dr|, |dc| <= 6 reproducing every demonstration."""
    from .ops import shift_objects
    from . import perception as P
    if any(shape(x) != shape(y) for x, y in zip(xs, ys)) or all(x == y for x, y in zip(xs, ys)):
        return []
    from .ops import resolve_bg as _rbg
    out = []
    for b in background(xs, ys)[:2]:
        for sel in ("largest", "smallest", "all"):
            for dr in range(-6, 7):
                for dc in range(-6, 7):
                    if (sel, dr, dc) not in out and (dr or dc) and all(
                            shift_objects(x, None, (sel, dr, dc), _rbg(b, x)) == y
                            for x, y in zip(xs, ys)):
                        out.append((sel, dr, dc))
    return out[:2]


RULES = {name: fn for name, fn in globals().items() if callable(fn) and not name.startswith("_")
         and name not in ("shape", "colours", "most_common", "gcd", "Counter")}


# ------------------------------------------------------------------ engine fitters (OPT198)
def _engine_rule(name):
    def rule(xs, ys, inputs=None):
        from .engines import fit
        return fit(name, xs, ys)
    return rule


from .engines import ENGINES as _ENGINES  # noqa: E402
for _name in _ENGINES:
    RULES[_name + "_spec"] = _engine_rule(_name)
