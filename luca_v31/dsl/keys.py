"""LUCA DSL - M1 generic key reading (OPT177).

Many ARC grids carry a small KEY that says how to treat the rest of the grid: colour pairs, a
set of colours, a palette in a corner box or behind a separator. A key rule is four choices, all
fitted on the demonstrations:

    reader  : WHERE the key is      comp2 / comp4 / comp8 (small multicolour pieces of <= N cells),
                                    panel (smaller side of a two-panel separator split),
                                    corner_box (corner rectangle closed off by a one-colour frame),
                                    edge_line (a multicolour first/last row or column set off by
                                    a uniform neighbouring line)
    decoder : WHAT it says          pairs_h / pairs_v (adjacent colour pairs, left->right or
                                    top->bottom; *_rev reverses them), colours (set of key colours)
    action  : WHAT to do with it    pairs  -> recolour, swap, fill_holes, outline
                                    colours-> delete, keep_only, paint_all
    output  : WHAT to return        keep_key, erase_key, crop_scene

The key's CONTENT is never learned: it is read from each grid, so a rule fitted on some colours
applies to any colours. This replaces the per-task legend modules removed in OPT176 with one
parameterised mechanism. Pure functions; never raise on ordinary grids.
"""
from . import perception as P

PAIR_ACTIONS = ("recolour", "swap", "fill_holes", "outline")
SET_ACTIONS = ("delete", "keep_only", "paint_all")
DECODERS = ("pairs_h", "pairs_v", "pairs_h_rev", "pairs_v_rev", "colours")
OUTPUTS = ("keep_key", "erase_key", "crop_scene")


# ------------------------------------------------------------------ readers
def _components(g, bg, n):
    comps = [c for c in P.objects(g, bg=bg, conn=4, by_colour=False)
             if len(c) <= n and len(P.object_colours(g, c)) >= 2]
    if not comps:
        return None
    return set().union(*map(set, comps)), comps


def _panel(g, bg):
    pan = P.panels(g)
    if not pan:
        return None
    boxes = [b for row in pan for b in row]
    if len(boxes) != 2:
        return None
    sizes = [(b[2] - b[0] + 1) * (b[3] - b[1] + 1) for b in boxes]
    if sizes[0] == sizes[1]:
        return None
    kb = boxes[sizes.index(min(sizes))]
    _, rs, cs = P.separator_lines(g)
    cells = {(r, c) for r in range(kb[0], kb[2] + 1) for c in range(kb[1], kb[3] + 1)}
    cells |= {(r, c) for r in rs for c in range(len(g[0]))} | {(r, c) for r in range(len(g)) for c in cs}
    inner = [(r, c) for r in range(kb[0], kb[2] + 1) for c in range(kb[1], kb[3] + 1) if g[r][c] != bg]
    if not inner:
        return None
    return cells, P.cell_components(inner, 4)


def _corner_box(g, bg):
    h, w = len(g), len(g[0])
    best = None
    for fc in sorted({v for row in g for v in row} - {bg}):
        for cr, cc in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            for kh in range(1, min(h - 1, 8)):
                for kw in range(1, min(w - 1, 8)):
                    r0 = 0 if cr == 0 else h - kh
                    c0 = 0 if cc == 0 else w - kw
                    fr = kh if cr == 0 else h - kh - 1
                    fcl = kw if cc == 0 else w - kw - 1
                    rows = range(min(r0, fr), max(r0 + kh - 1, fr) + 1)
                    cols = range(min(c0, fcl), max(c0 + kw - 1, fcl) + 1)
                    if not (all(g[fr][c] == fc for c in cols) and all(g[r][fcl] == fc for r in rows)):
                        continue
                    inner = [(r, c) for r in range(r0, r0 + kh) for c in range(c0, c0 + kw)]
                    if any(g[r][c] == fc for r, c in inner):
                        continue
                    cells = {(r, c) for r in rows for c in cols}
                    if best is None or len(cells) < len(best[0]):
                        best = (cells, inner)
    if best is None:
        return None
    nz = [q for q in best[1] if g[q[0]][q[1]] != bg]
    return (best[0], P.cell_components(nz, 8)) if nz else None


def _edge_line(g, bg):
    h, w = len(g), len(g[0])
    if h < 3 or w < 3:
        return None
    for kind, i in (("row", 0), ("row", h - 1), ("col", 0), ("col", w - 1)):
        j = i + 1 if i == 0 else i - 1
        line = [(i, c) for c in range(w)] if kind == "row" else [(r, i) for r in range(h)]
        nxt = [(j, c) for c in range(w)] if kind == "row" else [(r, j) for r in range(h)]
        if len({g[r][c] for r, c in line} - {bg}) >= 2 and len({g[r][c] for r, c in nxt}) == 1:
            return set(line) | set(nxt), [[q for q in line if g[q[0]][q[1]] != bg]]
    return None


READERS = {"comp2": lambda g, bg: _components(g, bg, 2), "comp4": lambda g, bg: _components(g, bg, 4),
           "comp8": lambda g, bg: _components(g, bg, 8), "panel": _panel, "corner_box": _corner_box,
           "edge_line": _edge_line}


def read(g, reader, bg=None):
    """(key cells, list of key pieces) or None."""
    if bg is None:
        bg = P.background(g)
    try:
        return READERS[reader](g, bg)
    except (IndexError, ValueError, TypeError):
        return None


# ------------------------------------------------------------------ decoders
def decode(g, pieces, decoder, bg):
    if decoder == "colours":
        s = {g[r][c] for cells in pieces for r, c in cells} - {bg}
        return frozenset(s) or None
    axis, rev = decoder[6], decoder.endswith("rev")
    m = {}
    for cells in pieces:
        cs = set(cells)
        for r, c in sorted(cs):
            q = (r, c + 1) if axis == "h" else (r + 1, c)
            if q in cs and g[r][c] != g[q[0]][q[1]]:
                a, b = g[r][c], g[q[0]][q[1]]
                if rev:
                    a, b = b, a
                if m.setdefault(a, b) != b:
                    return None
    return m or None


# ------------------------------------------------------------------ actions
def _enclosed(scene, free, h, w):
    stack = [(r, c) for r, c in free if r in (0, h - 1) or c in (0, w - 1)]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for dr, dc in P.DIRS4:
            q = (r + dr, c + dc)
            if q in free and q not in seen:
                seen.add(q); stack.append(q)
    return free - seen


def act(g, bg, key_cells, content, action, output):
    h, w = len(g), len(g[0])
    out = [list(r) for r in g]
    scene = [(r, c) for r in range(h) for c in range(w) if (r, c) not in key_cells]
    if 2 * len(key_cells) >= h * w or all(g[r][c] == bg for r, c in scene):
        return None  # a key is a small part of the grid that speaks about a non-empty rest
    if (action in PAIR_ACTIONS) != isinstance(content, dict):
        return None
    if action == "recolour":
        for r, c in scene:
            out[r][c] = content.get(g[r][c], g[r][c])
    elif action == "swap":
        inv = {**content, **{b: a for a, b in content.items()}}
        for r, c in scene:
            out[r][c] = inv.get(g[r][c], g[r][c])
    elif action == "outline":
        sc = set(scene)
        for a, b in content.items():
            for r, c in scene:
                if g[r][c] == a:
                    for dr, dc in P.DIRS8:
                        q = (r + dr, c + dc)
                        if q in sc and g[q[0]][q[1]] == bg:
                            out[q[0]][q[1]] = b
    elif action == "fill_holes":
        for a, b in content.items():
            free = {(r, c) for r, c in scene if g[r][c] != a}
            if len(free) == len(scene):
                continue
            for r, c in _enclosed(scene, free, h, w):
                if g[r][c] == bg:
                    out[r][c] = b
    elif action == "delete":
        for r, c in scene:
            if g[r][c] in content:
                out[r][c] = bg
    elif action == "keep_only":
        for r, c in scene:
            if g[r][c] not in content:
                out[r][c] = bg
    elif action == "paint_all":
        if len(content) != 1:
            return None
        k = next(iter(content))
        for r, c in scene:
            if g[r][c] != bg:
                out[r][c] = k
    else:
        return None
    if output == "erase_key":
        for r, c in key_cells:
            out[r][c] = bg
    elif output == "crop_scene":
        box = P.bbox(scene)
        if box is None or any((r, c) in key_cells for r in range(box[0], box[2] + 1)
                              for c in range(box[1], box[3] + 1)):
            return None
        return tuple(tuple(row[box[1]:box[3] + 1]) for row in out[box[0]:box[2] + 1])
    return tuple(map(tuple, out))


# ------------------------------------------------------------------ spec = (reader, decoder, action, output)
SPECS = tuple((rd, dec, a, o) for rd in READERS for dec in DECODERS
              for a in (SET_ACTIONS if dec == "colours" else PAIR_ACTIONS) for o in OUTPUTS)


def apply_spec(g, spec):
    reader, decoder, action, output = spec
    bg = P.background(g)
    k = read(g, reader, bg)
    if not k:
        return None
    content = decode(g, k[1], decoder, bg)
    if content is None:
        return None
    return act(g, bg, k[0], content, action, output)


def fit(xs, ys):
    """Every spec that reproduces every demonstration exactly and changes at least one of them."""
    xs = [tuple(map(tuple, x)) for x in xs]
    ys = [tuple(map(tuple, y)) for y in ys]
    bgs = [P.background(x) for x in xs]
    reads = {}
    out = []
    for spec in SPECS:
        reader, decoder, action, output = spec
        good, changed = True, False
        for i, (x, y) in enumerate(zip(xs, ys)):
            if (reader, i) not in reads:
                reads[reader, i] = read(x, reader, bgs[i])
            k = reads[reader, i]
            if not k:
                good = False; break
            content = decode(x, k[1], decoder, bgs[i])
            if content is None:
                good = False; break
            p = act(x, bgs[i], k[0], content, action, output)
            if p != y:
                good = False; break
            changed |= p != x
        if good and changed:
            out.append(spec)
    return out
