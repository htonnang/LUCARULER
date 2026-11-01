"""
primitives.py
==============
The "ancestral vocabulary" of reusable causal operators (Step 13 /
Section 17 of the spec): reflect, crop, recolor, paint, move, connect,
count, filter, rank, map, construct, repeat, plus a handful of
association/expression-driven laws (Steps 10-16).

Each primitive here is implemented as a *fitter*: given the full set of
training (input, output) pairs, it either

  * returns None (this law cannot explain these demonstrations), or
  * returns a `Law` object: a name, a human-readable description of the
    *derived* parameters it inferred (never bare unexplained constants
    when a relational alternative exists), and a `predict(grid)->grid`
    callable.

search.py composes/ranks these; MDL scoring penalizes laws whose
parameters could not be derived from the input (see Law.constant_bits).
"""

from __future__ import annotations
from .diagnostics import record_exception
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from collections import Counter
from math import gcd
from functools import reduce

from .grid_world import (
    Grid, to_grid, grid_shape, color_counts, guess_backgrounds,
    extract_objects, extract_holes, Entity,
)
from .expressions import synthesize_scalar_expression
from .dsl import perception as _P  # Phase 2 canonical perception layer

Pair = Tuple[Grid, Grid]


@dataclass
class Law:
    name: str
    predict: Callable[[Grid], Grid]
    description: str
    constant_bits: int = 0  # extra MDL penalty for unexplained constants
    op_count: int = 1       # number of primitive operations chained


def _fits_all(law_predict, pairs: List[Pair]) -> bool:
    from .budget import check as _budget_check
    _budget_check()
    try:
        for x, y in pairs:
            if law_predict(x) != y:
                return False
        return True
    except Exception:
        record_exception('primitives.py:53')
        return False


# ------------------------------------------------------------ geometry

def _reflect_h(g: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in g)


def _reflect_v(g: Grid) -> Grid:
    return tuple(reversed(g))


def _rot90(g: Grid) -> Grid:
    return tuple(zip(*g[::-1]))


def _rot180(g: Grid) -> Grid:
    return _rot90(_rot90(g))


def _rot270(g: Grid) -> Grid:
    return _rot90(_rot90(_rot90(g)))


def _transpose(g: Grid) -> Grid:
    return tuple(zip(*g))


def _anti_transpose(g: Grid) -> Grid:
    return _reflect_h(_reflect_v(_transpose(g)))


GEOM_OPS = {
    "identity": lambda g: g,
    "reflect_h": _reflect_h,
    "reflect_v": _reflect_v,
    "rot90": _rot90,
    "rot180": _rot180,
    "rot270": _rot270,
    "transpose": _transpose,
    "anti_transpose": _anti_transpose,
}


def fit_geometry(pairs: List[Pair]) -> List[Law]:
    from .budget import check as _budget_check
    _budget_check()
    laws = []
    for name, op in GEOM_OPS.items():
        if _fits_all(op, pairs):
            laws.append(Law(name, op, f"GEOMETRIC({name})", constant_bits=0, op_count=1))
    return laws


# ------------------------------------------------------------ color mapping

def fit_color_map(pairs: List[Pair]) -> Optional[Law]:
    """Learn a consistent bijective (or many-to-one) color -> color mapping
    applied uniformly across every cell (Step 16 style AssociationTable,
    keyed by raw color)."""
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None
    mapping = {}
    for x, y in pairs:
        for row_x, row_y in zip(x, y):
            for cx, cy in zip(row_x, row_y):
                if cx in mapping and mapping[cx] != cy:
                    return None
                mapping[cx] = cy
    if all(k == v for k, v in mapping.items()):
        return None  # identity, not interesting

    def predict(g: Grid) -> Grid:
        return tuple(tuple(mapping.get(c, c) for c in row) for row in g)

    if _fits_all(predict, pairs):
        return Law("color_map", predict,
                    f"LOOKUP(color -> color) table with {len(mapping)} entries",
                    constant_bits=len(mapping), op_count=1)
    return None


# ------------------------------------------------------------ crop to content

def _bbox_of_nonbg(g: Grid, bg: int):
    return _P.content_bbox(g, bg)  # Phase 2: canonical perception (was 6 lines)


def fit_crop_to_content(pairs: List[Pair]) -> Optional[Law]:
    from .budget import check as _budget_check
    _budget_check()
    def predict(g: Grid) -> Grid:
        bg = guess_backgrounds(g, k=1)[0]
        box = _bbox_of_nonbg(g, bg)
        if box is None:
            return g
        r0, c0, r1, c1 = box
        return tuple(row[c0:c1 + 1] for row in g[r0:r1 + 1])

    if _fits_all(predict, pairs):
        return Law("crop_to_content", predict, "CROP(bbox of non-background cells)",
                    constant_bits=0, op_count=1)
    return None


def fit_crop_largest_object(pairs: List[Pair], smallest: bool = False) -> Optional[Law]:
    from .budget import check as _budget_check
    _budget_check()
    def predict(g: Grid) -> Grid:
        bg = guess_backgrounds(g, k=1)[0]
        objs = extract_objects(g, bg, connectivity=8, multicolor=True)
        if not objs:
            return g
        target = min(objs, key=lambda e: e.area) if smallest else max(objs, key=lambda e: e.area)
        r0, c0, r1, c1 = target.bbox
        return tuple(row[c0:c1 + 1] for row in g[r0:r1 + 1])

    if _fits_all(predict, pairs):
        which = "SMALLEST" if smallest else "LARGEST"
        return Law(f"crop_{which.lower()}_object", predict,
                    f"CROP(bbox of {which} connected object)", constant_bits=0, op_count=2)
    return None


# ------------------------------------------------------------ scaling / tiling

def fit_scale_up(pairs: List[Pair]) -> Optional[Law]:
    from .budget import check as _budget_check
    _budget_check()
    ratios = set()
    for x, y in pairs:
        hx, wx = grid_shape(x)
        hy, wy = grid_shape(y)
        if hx == 0 or wx == 0 or hy % hx or wy % wx:
            return None
        ratios.add((hy // hx, wy // wx))
    if len(ratios) != 1:
        return None
    kh, kw = next(iter(ratios))
    if kh < 1 or kw < 1 or (kh == 1 and kw == 1):
        return None

    def predict(g: Grid) -> Grid:
        out = []
        for row in g:
            new_row = []
            for c in row:
                new_row.extend([c] * kw)
            for _ in range(kh):
                out.append(tuple(new_row))
        return tuple(out)

    if _fits_all(predict, pairs):
        return Law("scale_up", predict, f"SCALE(factor=Distance-derived ({kh},{kw}))",
                    constant_bits=0, op_count=1)
    return None


def fit_fractal_self_tile(pairs: List[Pair]) -> Optional[Law]:
    """Self-template: replace each non-background cell with a full copy of
    the input grid, background cells with blank -- classic ARC fractal task.

    V1 Priority 6 extension: two additional COMPLEMENT placement variants,
    where each placed block is the colour-complement of the input pattern
    (background<->foreground swapped within the block) rather than a
    literal copy. This directly targets a failure diagnosed in the LucaSO
    technical documentation (a task: some real tasks in the same
    visible fractal family place the input's colour COMPLEMENT at every
    non-background cell, which neither of the original two variants
    (copy-at-foreground, copy-at-background) can express."""
    from .budget import check as _budget_check
    _budget_check()
    # Every placement/background/complement variant has exactly (h*h, w*w)
    # output shape. Reject an impossible demonstration BEFORE allocating it.
    # This is a necessary condition, not a heuristic or a new search policy.
    if not pairs or any(grid_shape(y) != (len(x) ** 2, len(x[0]) ** 2)
                        for x, y in pairs if x and x[0]):
        return None
    def make(invert: bool, bg_val: Optional[int], complement: bool):
        def predict(g: Grid) -> Grid:
            bg = bg_val if bg_val is not None else guess_backgrounds(g, k=1)[0]
            h, w = grid_shape(g)
            fg = None
            if complement:
                fg_colors = {c for row in g for c in row if c != bg}
                if len(fg_colors) != 1:
                    raise ValueError("complement placement needs exactly one foreground colour")
                fg = next(iter(fg_colors))
            out = [[bg] * (w * w) for _ in range(h * h)]
            for i in range(h):
                for j in range(w):
                    cell_is_bg = (g[i][j] == bg)
                    place = (not cell_is_bg) if not invert else cell_is_bg
                    if place:
                        for a in range(h):
                            for b in range(w):
                                if complement:
                                    out[i * h + a][j * w + b] = fg if g[a][b] == bg else bg
                                else:
                                    out[i * h + a][j * w + b] = g[a][b]
            return tuple(tuple(row) for row in out)
        return predict

    # Try both the frequency-ranked backgrounds AND an explicit "0" hypothesis,
    # since fractal tasks often use color 0 as background even when it is not
    # the majority color.
    bg_hypotheses = list(guess_backgrounds(pairs[0][0], k=2))
    if 0 not in bg_hypotheses:
        bg_hypotheses.append(0)
    for bg_val in bg_hypotheses:
        for invert in (False, True):
            for complement in (False, True):
                p = make(invert, bg_val, complement)
                if _fits_all(p, pairs):
                    kind = "COMPLEMENT" if complement else "COPY"
                    return Law("self_template_fractal", p,
                                f"SELF_TEMPLATE(place={kind}, invert={invert}, background derived={bg_val})",
                                constant_bits=1, op_count=2)
    return None


def fit_tile_to_output(pairs: List[Pair]) -> Optional[Law]:
    """Repeat/mirror-tile the input to exactly fill a learned output size,
    trying plain repeat and mirrored (wallpaper) tiling."""
    from .budget import check as _budget_check
    _budget_check()
    ratios = set()
    for x, y in pairs:
        hx, wx = grid_shape(x)
        hy, wy = grid_shape(y)
        if hx == 0 or wx == 0 or hy % hx or wy % wx:
            return None
        ratios.add((hy // hx, wy // wx))
    if len(ratios) != 1:
        return None
    kh, kw = next(iter(ratios))
    if kh < 1 or kw < 1 or (kh == 1 and kw == 1):
        return None

    def make(mirror: bool):
        def predict(g: Grid) -> Grid:
            h, w = grid_shape(g)
            out = [[0] * (w * kw) for _ in range(h * kh)]
            for ti in range(kh):
                for tj in range(kw):
                    block = g
                    if mirror:
                        if ti % 2 == 1:
                            block = tuple(reversed(block))
                        if tj % 2 == 1:
                            block = tuple(tuple(reversed(r)) for r in block)
                    for a in range(h):
                        for b in range(w):
                            out[ti * h + a][tj * w + b] = block[a][b]
            return tuple(tuple(row) for row in out)
        return predict

    for mirror in (False, True):
        p = make(mirror)
        if _fits_all(p, pairs):
            return Law("tile_output" + ("_mirror" if mirror else ""), p,
                        f"REPEAT(grid, factor=({kh},{kw}), mirror={mirror})",
                        constant_bits=0, op_count=1)
    return None


# ------------------------------------------------------------ symmetry repair

def fit_symmetry_repair(pairs: List[Pair]) -> Optional[Law]:
    """Fill occluded region (a single 'noise' color present in input but
    absent in output) using detected grid symmetry."""
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None
    # candidate mask colors: colors present in x but not in y
    mask_candidates = set()
    for x, y in pairs:
        cx, cy = set(color_counts(x)), set(color_counts(y))
        diff = cx - cy
        if not diff:
            return None
        mask_candidates |= diff
    for mask_color in mask_candidates:
        def predict(g: Grid, mc=mask_color) -> Grid:
            h, w = grid_shape(g)
            grid = [list(r) for r in g]
            unknown = {(r, c) for r in range(h) for c in range(w) if g[r][c] == mc}
            if not unknown:
                return g
            transforms = [
                lambda r, c: (r, w - 1 - c),
                lambda r, c: (h - 1 - r, c),
                lambda r, c: (h - 1 - r, w - 1 - c),
            ]
            if h == w:
                transforms += [lambda r, c: (c, r), lambda r, c: (w - 1 - c, h - 1 - r)]
            changed = True
            passes = 0
            while changed and passes < 6:
                changed = False
                passes += 1
                for (r, c) in list(unknown):
                    for t in transforms:
                        rr, cc = t(r, c)
                        if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] != mc:
                            grid[r][c] = grid[rr][cc]
                            unknown.discard((r, c))
                            changed = True
                            break
            return tuple(tuple(row) for row in grid)

        if _fits_all(predict, pairs):
            return Law("symmetry_repair", predict,
                        f"SYMMETRY_FILL(mask_color derived={mask_color})",
                        constant_bits=1, op_count=2)
    return None


def fit_symmetry_repair_crop(pairs: List[Pair]) -> Optional[Law]:
    """Like symmetry_repair, but the output is cropped to just the
    previously-occluded region (common ARC pattern)."""
    from .budget import check as _budget_check
    _budget_check()
    mask_candidates = set()
    for x, y in pairs:
        cx = set(color_counts(x))
        mask_candidates |= cx
    for mask_color in mask_candidates:
        def predict(g: Grid, mc=mask_color) -> Grid:
            h, w = grid_shape(g)
            grid = [list(r) for r in g]
            unknown = {(r, c) for r in range(h) for c in range(w) if g[r][c] == mc}
            if not unknown:
                return g
            r0, c0 = min(r for r, _ in unknown), min(c for _, c in unknown)
            r1, c1 = max(r for r, _ in unknown), max(c for _, c in unknown)
            transforms = [
                lambda r, c: (r, w - 1 - c),
                lambda r, c: (h - 1 - r, c),
                lambda r, c: (h - 1 - r, w - 1 - c),
            ]
            if h == w:
                transforms += [lambda r, c: (c, r), lambda r, c: (w - 1 - c, h - 1 - r)]
            changed = True
            passes = 0
            while changed and passes < 6:
                changed = False
                passes += 1
                for (r, c) in list(unknown):
                    for t in transforms:
                        rr, cc = t(r, c)
                        if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] != mc:
                            grid[r][c] = grid[rr][cc]
                            unknown.discard((r, c))
                            changed = True
                            break
            return tuple(tuple(row[c0:c1 + 1]) for row in grid[r0:r1 + 1])

        if _fits_all(predict, pairs):
            return Law("symmetry_repair_crop", predict,
                        f"CROP(SYMMETRY_FILL(mask={mask_color}), region=occluded_bbox)",
                        constant_bits=1, op_count=3)
    return None


# ------------------------------------------------------------ holes / fill

def fit_fill_holes(pairs: List[Pair]) -> Optional[Law]:
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None
    # learn fill color: either constant, or "color of the enclosing object"
    fill_colors = set()
    per_pair_fill_color = []
    contexts = []
    for x, y in pairs:
        bg = guess_backgrounds(x, k=1)[0]
        holes = extract_holes(x, bg)
        objs = extract_objects(x, bg, connectivity=8, multicolor=False)
        for h in holes:
            (r, c) = next(iter(h.cells))
            fc = y[r][c]
            fill_colors.add(fc)
            per_pair_fill_color.append(fc)
            color_freq = Counter(o.color for o in objs)
            most_common_color = color_freq.most_common(1)[0][0] if color_freq else bg
            contexts.append({
                "background": bg,
                "num_objects": len(objs),
                "most_common_object_color": most_common_color,
            })
    if len(fill_colors) == 1 and per_pair_fill_color:
        # Prefer a derived expression over the bare constant when one exists
        # (Section 3.3: "infer expressions rather than constants").
        expr = synthesize_scalar_expression(contexts, per_pair_fill_color)
        fc_default = next(iter(fill_colors))

        def predict(g: Grid) -> Grid:
            bg = guess_backgrounds(g, k=1)[0]
            holes = extract_holes(g, bg)
            objs = extract_objects(g, bg, connectivity=8, multicolor=False)
            color_freq = Counter(o.color for o in objs)
            most_common_color = color_freq.most_common(1)[0][0] if color_freq else bg
            ctx = {"background": bg, "num_objects": len(objs),
                   "most_common_object_color": most_common_color}
            fc = expr.eval(ctx) if expr is not None else fc_default
            if fc is None:
                fc = fc_default
            grid = [list(r) for r in g]
            for h in holes:
                for (r, c) in h.cells:
                    grid[r][c] = fc
            return tuple(tuple(row) for row in grid)

        if _fits_all(predict, pairs):
            if expr is not None and not expr.is_constant:
                desc = f"FILL_ENCLOSED(color = {expr.name})"
                bits = 1
            else:
                desc = f"FILL_ENCLOSED(color derived constant={fc_default})"
                bits = 3
            return Law("fill_holes_constant", predict, desc, constant_bits=bits, op_count=2)

    def predict_same(g: Grid) -> Grid:
        bg = guess_backgrounds(g, k=1)[0]
        holes = extract_holes(g, bg)
        objs = extract_objects(g, bg, connectivity=8, multicolor=True)
        grid = [list(r) for r in g]
        for h in holes:
            (hr, hc) = next(iter(h.cells))
            best = None
            best_d = None
            for o in objs:
                r0, c0, r1, c1 = o.bbox
                if r0 - 1 <= hr <= r1 + 1 and c0 - 1 <= hc <= c1 + 1:
                    d = 0
                    if best_d is None or d < best_d:
                        best, best_d = o, d
            if best is not None:
                fill = Counter(best.multicolor.values()).most_common(1)[0][0]
                for (r, c) in h.cells:
                    grid[r][c] = fill
        return tuple(tuple(row) for row in grid)

    if _fits_all(predict_same, pairs):
        return Law("fill_holes_enclosing_color", predict_same,
                    "FILL_ENCLOSED(color = Color(enclosing object))",
                    constant_bits=0, op_count=2)
    return None


# ------------------------------------------------------------ denoise

def fit_denoise(pairs: List[Pair]) -> Optional[Law]:
    """Remove isolated single-cell 'noise' objects (paint them background)."""
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None

    def predict(g: Grid) -> Grid:
        bg = guess_backgrounds(g, k=1)[0]
        objs = extract_objects(g, bg, connectivity=8, multicolor=False)
        grid = [list(r) for r in g]
        for o in objs:
            if o.area == 1:
                for (r, c) in o.cells:
                    grid[r][c] = bg
        return tuple(tuple(row) for row in grid)

    if _fits_all(predict, pairs):
        return Law("denoise_singletons", predict,
                    "FILTER(objects, area==1) -> PAINT(background)",
                    constant_bits=0, op_count=2)
    return None


# ------------------------------------------------------------ object recolor by rank/association

def fit_recolor_by_size_rank(pairs: List[Pair]) -> Optional[Law]:
    """Learn a mapping from an object's size-rank (0=smallest) to its output
    color -- a RANK-keyed AssociationTable (Steps 6+16)."""
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None
    mapping = {}
    for x, y in pairs:
        bg = guess_backgrounds(x, k=1)[0]
        objs = extract_objects(x, bg, connectivity=8, multicolor=False)
        if not objs:
            return None
        ranked = sorted(objs, key=lambda e: (e.area, e.bbox))
        for rank, o in enumerate(ranked):
            out_colors = {y[r][c] for r, c in o.cells}
            if len(out_colors) != 1:
                return None
            oc = next(iter(out_colors))
            if rank in mapping and mapping[rank] != oc:
                return None
            mapping[rank] = oc

    def predict(g: Grid) -> Grid:
        bg = guess_backgrounds(g, k=1)[0]
        objs = extract_objects(g, bg, connectivity=8, multicolor=False)
        ranked = sorted(objs, key=lambda e: (e.area, e.bbox))
        grid = [list(r) for r in g]
        for rank, o in enumerate(ranked):
            if rank not in mapping:
                continue
            for (r, c) in o.cells:
                grid[r][c] = mapping[rank]
        return tuple(tuple(row) for row in grid)

    if _fits_all(predict, pairs):
        return Law("recolor_by_size_rank", predict,
                    f"MAP(objects, RANK(area) -> LOOKUP(color)) [{len(mapping)} entries]",
                    constant_bits=len(mapping), op_count=2)
    return None


def fit_recolor_by_shape(pairs: List[Pair]) -> Optional[Law]:
    """AssociationTable keyed by canonical object shape -> output color."""
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None
    mapping = {}
    for x, y in pairs:
        bg = guess_backgrounds(x, k=1)[0]
        objs = extract_objects(x, bg, connectivity=8, multicolor=False)
        for o in objs:
            out_colors = {y[r][c] for r, c in o.cells}
            if len(out_colors) != 1:
                return None
            oc = next(iter(out_colors))
            key = o.shape()
            if key in mapping and mapping[key] != oc:
                return None
            mapping[key] = oc

    def predict(g: Grid) -> Grid:
        bg = guess_backgrounds(g, k=1)[0]
        objs = extract_objects(g, bg, connectivity=8, multicolor=False)
        grid = [list(r) for r in g]
        for o in objs:
            key = o.shape()
            if key in mapping:
                for (r, c) in o.cells:
                    grid[r][c] = mapping[key]
        return tuple(tuple(row) for row in grid)

    if _fits_all(predict, pairs) and len(mapping) >= 1:
        return Law("recolor_by_shape", predict,
                    f"MAP(objects, Shape -> LOOKUP(color)) [{len(mapping)} entries]",
                    constant_bits=len(mapping) * 2, op_count=2)
    return None


# ------------------------------------------------------------ gravity / move-until

def fit_gravity(pairs: List[Pair]) -> Optional[Law]:
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None
    directions = {
        "down": (1, 0), "up": (-1, 0), "left": (0, -1), "right": (0, 1),
    }

    def make(dr, dc):
        def predict(g: Grid) -> Grid:
            bg = guess_backgrounds(g, k=1)[0]
            h, w = grid_shape(g)
            objs = extract_objects(g, bg, connectivity=4, multicolor=False)
            occupied = set()
            order = sorted(objs, key=lambda o: -(o.bbox[2] * dr + o.bbox[3] * dc))
            placements = {}
            for o in order:
                shift = 0
                while True:
                    moved = {(r + dr * (shift + 1), c + dc * (shift + 1)) for r, c in o.cells}
                    if all(0 <= r < h and 0 <= c < w for r, c in moved) and not (moved & occupied):
                        shift += 1
                    else:
                        break
                new_cells = {(r + dr * shift, c + dc * shift) for r, c in o.cells}
                placements[o.id] = new_cells
                occupied |= new_cells
            grid = [[bg] * w for _ in range(h)]
            for o in objs:
                for (r, c) in placements[o.id]:
                    grid[r][c] = o.color
            return tuple(tuple(row) for row in grid)
        return predict

    for name, (dr, dc) in directions.items():
        p = make(dr, dc)
        if _fits_all(p, pairs):
            return Law(f"gravity_{name}", p, f"MOVE_UNTIL(objects, direction={name}, stop=blocked)",
                        constant_bits=0, op_count=1)
    return None


# ------------------------------------------------------------ panel logic (split by separator)

def _find_separator_lines(g: Grid):
    h, w = grid_shape(g)
    row_seps = [i for i in range(h) if len(set(g[i])) == 1]
    col_seps = [j for j in range(w) if len(set(g[i][j] for i in range(h))) == 1]
    return row_seps, col_seps


def _split_panels(g: Grid):
    h, w = grid_shape(g)
    row_seps, col_seps = _find_separator_lines(g)
    # find a single separator color splitting grid into exactly 2 panels either
    # horizontally or vertically
    for j in col_seps:
        if 0 < j < w - 1:
            left = tuple(row[:j] for row in g)
            right = tuple(row[j + 1:] for row in g)
            if grid_shape(left) == grid_shape(right) and grid_shape(left)[1] > 0:
                return left, right
    for i in row_seps:
        if 0 < i < h - 1:
            top = g[:i]
            bot = g[i + 1:]
            if grid_shape(top) == grid_shape(bot) and grid_shape(top)[0] > 0:
                return top, bot
    # even split fallback (no separator, panels adjacent)
    if w % 2 == 0:
        left = tuple(row[:w // 2] for row in g)
        right = tuple(row[w // 2:] for row in g)
        return left, right
    if h % 2 == 0:
        return g[:h // 2], g[h // 2:]
    return None


def fit_panel_boolean(pairs: List[Pair]) -> Optional[Law]:
    from .budget import check as _budget_check
    _budget_check()
    split0 = _split_panels(pairs[0][0])
    if split0 is None:
        return None
    out_colors = set()
    for _, y in pairs:
        out_colors |= set(color_counts(y).keys())
    if len(out_colors) > 2:
        pass  # still try; ops below handle multi-color via "paint" color search

    ops = {
        "and": lambda a, b, bg: a != bg and b != bg,
        "or": lambda a, b, bg: a != bg or b != bg,
        "xor": lambda a, b, bg: (a != bg) != (b != bg),
        "diff": lambda a, b, bg: a != bg and b == bg,
    }
    for op_name, op in ops.items():
        for paint in range(10):
            def predict(g: Grid, op=op, paint=paint) -> Grid:
                split = _split_panels(g)
                if split is None:
                    return g
                a, b = split
                bg = guess_backgrounds(g, k=1)[0]
                ha, wa = grid_shape(a)
                out = [[bg] * wa for _ in range(ha)]
                for i in range(ha):
                    for j in range(wa):
                        if op(a[i][j], b[i][j], bg):
                            out[i][j] = paint
                return tuple(tuple(row) for row in out)

            if _fits_all(predict, pairs):
                return Law(f"panel_{op_name}", predict,
                            f"SPLIT(panels) -> {op_name.upper()}(A,B) -> PAINT(color derived={paint})",
                            constant_bits=4, op_count=3)
    return None


# ------------------------------------------------------------ counting

def fit_count_to_solid(pairs: List[Pair]) -> Optional[Law]:
    """Output is a 1x1 (or NxN) solid grid whose color/size encodes a count
    of objects in the input."""
    from .budget import check as _budget_check
    _budget_check()
    def count_objs(g):
        bg = guess_backgrounds(g, k=1)[0]
        return len(extract_objects(g, bg, connectivity=8, multicolor=False))

    # Case A: output size encodes the count (NxN solid of most common color)
    ok = True
    for x, y in pairs:
        hy, wy = grid_shape(y)
        colors = set(color_counts(y).keys())
        if len(colors) != 1 or hy != wy or hy != count_objs(x):
            ok = False
            break
    if ok:
        contexts = [{"num_objects": count_objs(x), "grid_h": grid_shape(x)[0],
                      "grid_w": grid_shape(x)[1]} for x, _ in pairs]
        targets = [next(iter(color_counts(y).keys())) for _, y in pairs]
        expr = synthesize_scalar_expression(contexts, targets)
        color_default = next(iter(color_counts(pairs[0][1]).keys()))

        def predict(g: Grid) -> Grid:
            n = count_objs(g)
            ctx = {"num_objects": n, "grid_h": grid_shape(g)[0], "grid_w": grid_shape(g)[1]}
            color = expr.eval(ctx) if expr is not None else color_default
            if color is None:
                color = color_default
            return tuple(tuple(color for _ in range(n)) for _ in range(n))

        if _fits_all(predict, pairs):
            if expr is not None and not expr.is_constant:
                desc = f"GRID(size = Count(objects), color = {expr.name})"
                bits = 1
            else:
                desc = "GRID(size = Count(objects), color derived constant)"
                bits = 3
            return Law("count_to_solid_size", predict, desc, constant_bits=bits, op_count=2)
    return None


# ------------------------------------------------------------ shape-driven tiling (V9 Item 4)

def fit_tile_by_expression(pairs: List[Pair]) -> Optional[Law]:
    """Generalizes fit_tile_to_output: instead of requiring one constant
    (kh, kw) ratio across every pair, derive kh/kw per-pair from an
    expression over the input (e.g. kh = Count(objects)) via
    expressions.synthesize_scalar_expression. Covers tasks like 'repeat
    the pattern N times, where N is the number of markers'."""
    from .budget import check as _budget_check
    _budget_check()
    contexts = []
    kh_targets, kw_targets = [], []
    for x, y in pairs:
        hx, wx = grid_shape(x)
        hy, wy = grid_shape(y)
        if hx == 0 or wx == 0 or hy % hx or wy % wx:
            return None
        bg = guess_backgrounds(x, k=1)[0]
        objs = extract_objects(x, bg, connectivity=8, multicolor=False)
        contexts.append({"num_objects": len(objs), "grid_h": hx, "grid_w": wx})
        kh_targets.append(hy // hx)
        kw_targets.append(wy // wx)

    if len(set(kh_targets)) == 1 and len(set(kw_targets)) == 1:
        return None  # constant ratio -- already covered by fit_tile_to_output

    expr_h = synthesize_scalar_expression(contexts, kh_targets)
    expr_w = synthesize_scalar_expression(contexts, kw_targets)
    if expr_h is None or expr_w is None:
        return None
    if expr_h.is_constant and expr_w.is_constant:
        return None  # would just be a bare-constant coincidence; reject

    def predict(g: Grid) -> Grid:
        h, w = grid_shape(g)
        bg = guess_backgrounds(g, k=1)[0]
        objs = extract_objects(g, bg, connectivity=8, multicolor=False)
        ctx = {"num_objects": len(objs), "grid_h": h, "grid_w": w}
        kh = expr_h.eval(ctx) or 1
        kw = expr_w.eval(ctx) or 1
        kh, kw = max(1, int(kh)), max(1, int(kw))
        out = [[0] * (w * kw) for _ in range(h * kh)]
        for ti in range(kh):
            for tj in range(kw):
                for a in range(h):
                    for b in range(w):
                        out[ti * h + a][tj * w + b] = g[a][b]
        return tuple(tuple(row) for row in out)

    if _fits_all(predict, pairs):
        return Law("tile_by_expression", predict,
                    f"REPEAT(grid, factor=({expr_h.name},{expr_w.name}))",
                    constant_bits=1, op_count=2)
    return None


# ------------------------------------------------------------ registry

FITTERS: List[Callable[[List[Pair]], object]] = [
    fit_color_map,
    fit_crop_to_content,
    lambda p: fit_crop_largest_object(p, smallest=False),
    lambda p: fit_crop_largest_object(p, smallest=True),
    fit_scale_up,
    fit_fractal_self_tile,
    fit_tile_to_output,
    fit_symmetry_repair,
    fit_symmetry_repair_crop,
    fit_fill_holes,
    fit_denoise,
    fit_recolor_by_size_rank,
    fit_recolor_by_shape,
    fit_gravity,
    fit_panel_boolean,
    fit_count_to_solid,
    fit_tile_by_expression,
]


def fit_all_single(pairs: List[Pair]) -> List[Law]:
    """Run every single-primitive fitter plus the geometry family."""
    from .budget import check as _budget_check
    _budget_check()
    return [law for law, _fitter in fit_all_single_traced(pairs)]


def fit_all_single_traced(pairs: List[Pair]) -> List[Tuple[Law, Optional[Callable]]]:
    """Like fit_all_single, but also returns the exact fitter function that
    produced each law (or None for the parameter-free geometry family,
    which needs no leave-one-out check since constant_bits is always 0).
    search.py uses the fitter reference to re-run leave-one-out validation
    (V9 Item 2) without needing to guess which fitter a Law came from."""
    from .budget import check as _budget_check
    _budget_check()
    traced: List[Tuple[Law, Optional[Callable]]] = []
    for law in fit_geometry(pairs):
        traced.append((law, None))
    for fitter in FITTERS:
        try:
            result = fitter(pairs)
        except Exception:
            record_exception('primitives.py:846')
            result = None
        if isinstance(result, list):
            for law in result:
                traced.append((law, fitter))
        elif isinstance(result, Law):
            traced.append((result, fitter))
    return traced
