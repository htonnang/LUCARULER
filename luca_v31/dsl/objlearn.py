"""LUCA DSL - object-rule learner (OPT200): which objects change, and what they take.

Objects: 8-connected one-colour components on the task background (perception.objects).
Every object gets a feature record (its own properties and its relations to other objects).
A RULE predicts a target per object (a new colour, keep / delete, ...):

    rule = (split, table)
      split  a feature name whose value selects a table entry ('const' = one entry for all)
      table  ((split value, answer), ...)    answer = ('lit', v)  the literal value v
                                                      ('role', f) the object's own value of feature f
                                                      (e.g. 'touch_colour': the colour it touches)

Learning: every demonstration object gives (features, target). For each split feature the objects
are grouped by its value; a group's answer must explain all of its objects (a role is preferred to a
literal; 'keep as is' roles first). Every group that is not 'unchanged' must be seen in >= 2
demonstrations (evidence rule). On a new grid an object whose split value has no entry makes the
rule inapplicable, unless the rule's every entry for the changing case is a role ('default').
"""
from collections import Counter
from . import perception as P
from .evidence import enough

COLOUR_ROLES = ("colour", "majority_other_colour", "minority_other_colour", "same_shape_colour", "same_d8_colour", "touch_colour", "nearest_colour",
                "container_colour", "contains_colour", "row_key_colour", "col_key_colour", "rarest_ink",
                "commonest_ink")
SPLITS = ("const", "colour", "size", "size_rank", "size_rank_rev", "is_largest", "is_smallest", "shape_count",
          "colour_count", "touches_border", "has_hole", "height", "width", "touch_colour", "n_touch",
          "touches_any", "contains_any", "is_contained", "is_line", "is_rect", "shape_unique", "colour_unique",
          "contains_colour", "container_colour", "nearest_colour", "shape_d8", "n_ends", "n_bends", "n_holes", "other_d8")


def _d8(shape):
    """Canonical form of a cell set under rotations and reflections."""
    forms = []
    pts = list(shape)
    for k in range(8):
        q = []
        for r, c in pts:
            for _ in range(k % 4):
                r, c = c, -r
            if k >= 4:
                c = -c
            q.append((r, c))
        r0 = min(r for r, _ in q); c0 = min(c for _, c in q)
        forms.append(tuple(sorted((r - r0, c - c0) for r, c in q)))
    return min(forms)


def _topology(o):
    s = set(o)
    ends = bends = 0
    for r, c in o:
        nb = [(a, b) for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0)) if (r + a, c + b) in s]
        if len(nb) == 1:
            ends += 1
        elif len(nb) == 2 and nb[0][0] != -nb[1][0] and nb[0][1] != -nb[1][1] or \
                len(nb) == 2 and (nb[0][0] == 0) != (nb[1][0] == 0):
            bends += 1
    return min(ends, 4), min(bends, 4)


def _holes(o, box):
    """Background components inside the bounding box not reaching its edge."""
    r0, c0, r1, c1 = box
    s = set(o)
    free = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in s}
    n = 0
    while free:
        st = [free.pop()]; comp = list(st); edge = False
        while st:
            r, c = st.pop()
            if r in (r0, r1) or c in (c0, c1):
                edge = True
            for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                q = (r + a, c + b)
                if q in free:
                    free.discard(q); st.append(q); comp.append(q)
        n += not edge
    return min(n, 3)


def _norm(o):
    r0 = min(r for r, _ in o); c0 = min(c for _, c in o)
    return tuple(sorted((r - r0, c - c0) for r, c in o))


_SCENE = {}


def scene(g, bg, conn=8):
    """Objects and their feature records (cached per grid)."""
    key = (g, bg, conn)
    if key in _SCENE:
        return _SCENE[key]
    if len(_SCENE) > 512:
        _SCENE.clear()
    res = _scene(g, bg, conn)
    _SCENE[key] = res
    return res


def segment(g, bg, conn):
    """Objects: conn 8 / 4 = one-colour connected components; 'colour' = all cells of one colour."""
    if conn == "colour":
        groups = {}
        for r, row in enumerate(g):
            for c, v in enumerate(row):
                if v != bg:
                    groups.setdefault(v, []).append((r, c))
        return [sorted(v) for _, v in sorted(groups.items())]
    return P.objects(g, bg=bg, conn=conn, by_colour=True)


def _scene(g, bg, conn=8):
    H, W = len(g), len(g[0])
    objs = segment(g, bg, conn)
    if not objs or len(objs) > 60:
        return None
    col = [g[o[0][0]][o[0][1]] for o in objs]
    owner = {}
    for i, o in enumerate(objs):
        for rc in o:
            owner[rc] = i
    boxes = [P.bbox(o) for o in objs]
    sizes = [len(o) for o in objs]
    ranks = sorted(set(sizes), reverse=True)
    shapes = Counter(_norm(o) for o in objs)
    colours = Counter(col)
    ink = Counter(v for r in g for v in r if v != bg)
    rarest = min(ink, key=lambda k: (ink[k], k)) if ink else None
    commonest = max(ink, key=lambda k: (ink[k], -k)) if ink else None
    norms = [_norm(o) for o in objs]
    d8s = [_d8(n) for n in norms]
    feats = []
    for i, o in enumerate(objs):
        r0, c0, r1, c1 = boxes[i]
        s = set(o)
        nb = {owner[(r + a, c + b)] for r, c in o for a in (-1, 0, 1) for b in (-1, 0, 1)
              if (r + a, c + b) in owner and owner[(r + a, c + b)] != i}
        tcols = {col[j] for j in nb}
        cont = [j for j in range(len(objs)) if j != i and boxes[j][0] <= r0 and boxes[j][1] <= c0 and
                boxes[j][2] >= r1 and boxes[j][3] >= c1 and boxes[j] != boxes[i]]
        inside = [j for j in range(len(objs)) if j != i and r0 <= boxes[j][0] and c0 <= boxes[j][1] and
                  r1 >= boxes[j][2] and c1 >= boxes[j][3] and boxes[j] != boxes[i]]
        # nearest other object (Chebyshev distance between cells), unique colour
        best, ncols = None, set()
        for j, q in enumerate(objs):
            if j == i:
                continue
            d = min(max(abs(r - a), abs(c - b)) for r, c in o for a, b in q) if len(o) * len(q) <= 4000 else \
                max(0, max(boxes[j][0] - r1, r0 - boxes[j][2], boxes[j][1] - c1, c0 - boxes[j][3]))
            if best is None or d < best:
                best, ncols = d, {col[j]}
            elif d == best:
                ncols.add(col[j])
        cont_col = col[min(cont, key=lambda j: sizes[j])] if cont else None
        me, md8 = _norm(o), d8s[i]
        ss = {col[j] for j in range(len(objs)) if j != i and col[j] != col[i] and norms[j] == me}
        sd = {col[j] for j in range(len(objs)) if j != i and col[j] != col[i] and d8s[j] == md8}
        rk = {col[j] for j in range(len(objs)) if j != i and col[j] != col[i] and
              not (boxes[j][2] < r0 or boxes[j][0] > r1)}
        ck = {col[j] for j in range(len(objs)) if j != i and col[j] != col[i] and
              not (boxes[j][3] < c0 or boxes[j][1] > c1)}
        ends, bends = _topology(o)
        oc = Counter(col[j] for j in range(len(objs)) if j != i and col[j] != col[i])
        maj = [k for k, n in oc.items() if n == max(oc.values())] if oc else []
        mino = [k for k, n in oc.items() if n == min(oc.values())] if oc else []
        od = {d8s[j] for j in range(len(objs)) if j != i}
        ins_cols = {col[j] for j in inside}
        h, w = r1 - r0 + 1, c1 - c0 + 1
        feats.append(dict(
            colour=col[i], size=len(o), size_rank=ranks.index(len(o)), size_rank_rev=len(ranks) - 1 - ranks.index(len(o)),
            is_largest=int(len(o) == ranks[0]), is_smallest=int(len(o) == ranks[-1]),
            shape_count=min(shapes[_norm(o)], 3), colour_count=min(colours[col[i]], 3),
            touches_border=int(r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1),
            has_hole=int(h * w > len(o) and h > 2 and w > 2), height=h, width=w,
            touch_colour=tcols.pop() if len(tcols) == 1 else None, n_touch=min(len(nb), 3), touches_any=int(bool(nb)),
            contains_any=int(bool(inside)), is_contained=int(bool(cont)),
            is_line=int(h == 1 or w == 1), is_rect=int(h * w == len(o)),
            shape_unique=int(shapes[_norm(o)] == 1), colour_unique=int(colours[col[i]] == 1),
            nearest_colour=ncols.pop() if len(ncols) == 1 else None,
            container_colour=cont_col, contains_colour=ins_cols.pop() if len(ins_cols) == 1 else None,
            rarest_ink=rarest, commonest_ink=commonest,
            same_shape_colour=ss.pop() if len(ss) == 1 else None,
            same_d8_colour=sd.pop() if len(sd) == 1 else None,
            row_key_colour=rk.pop() if len(rk) == 1 else None, col_key_colour=ck.pop() if len(ck) == 1 else None,
            majority_other_colour=maj[0] if len(maj) == 1 else None,
            minority_other_colour=mino[0] if len(mino) == 1 else None,
            other_d8=od.pop() if len(od) == 1 else None,
            shape_d8=d8s[i], n_ends=ends, n_bends=bends, n_holes=_holes(o, boxes[i])))
    return objs, feats


def _answers(f, target, roles):
    out = [("role", r) for r in roles if f.get(r) is not None and f[r] == target]
    out.append(("lit", target))
    return out


def learn(items, roles, unchanged_role=None, max_rules=3):
    """items: [(features, target, demo index)]. Rules (split, table) explaining every item."""
    rules = []
    for split in SPLITS:
        groups = {}
        for f, t, i in items:
            v = "*" if split == "const" else f.get(split)
            groups.setdefault(v, []).append((f, t, i))
        if None in groups and split != "const":
            continue
        if split != "const" and len(groups) < 2:
            continue
        table, good = [], True
        for v, its in sorted(groups.items(), key=lambda kv: repr(kv[0])):
            common = None
            for f, t, i in its:
                a = set(_answers(f, t, roles))
                common = a if common is None else common & a
                if not common:
                    break
            if not common:
                good = False; break
            pref = sorted(common, key=lambda a: (a != ("role", unchanged_role), a[0] != "role",
                                                 roles.index(a[1]) if a[0] == "role" else 0, repr(a)))
            ans = pref[0]
            if ans != ("role", unchanged_role) and not enough({i for _, _, i in its}, len(its)):
                good = False; break
            table.append((v, ans))
        if not good:
            continue
        if all(a == ("role", unchanged_role) for _, a in table):
            continue
        rules.append((split, tuple(table)))
    # simplest first: fewest table entries (stable in SPLITS order)
    rules.sort(key=lambda r: len(r[1]))
    return rules[:max_rules]


def learn_selected(items, roles, unchanged, max_rules=3):
    """Two-level rules ('sel', feature, changing values, target rule): objects whose feature value is
    one of the changing values take the target rule's answer (learned on the changed objects only);
    every other object keeps its value. items: [(features, target, demo)]; unchanged(f, t) -> bool."""
    out = []
    for sel in SPLITS:
        if sel == "const":
            continue
        ch_vals, keep_vals, good = set(), set(), True
        for f, t, i in items:
            v = f.get(sel)
            if v is None:
                good = False; break
            (keep_vals if unchanged(f, t) else ch_vals).add(v)
        if not good or not ch_vals or not keep_vals or ch_vals & keep_vals:
            continue
        changed = [(f, t, i) for f, t, i in items if f.get(sel) in ch_vals]
        if not all(enough({i for f, t, i in changed if f.get(sel) == v}, sum(1 for f, _, _ in changed if f.get(sel) == v))
                   for v in ch_vals) and not enough({i for _, _, i in changed}, len(changed)):
            continue
        for tr in learn(changed, roles, unchanged_role=None, max_rules=1):
            out.append(("sel", sel, tuple(sorted(ch_vals, key=repr)), tr))
            break
        if len(out) >= max_rules:
            break
    return out


def predict(rule, f, unchanged_role=None):
    if rule[0] == "sel":
        _, sel, vals, tr = rule
        if f.get(sel) not in vals:
            return f.get(unchanged_role) if unchanged_role else None
        return predict(tr, f)
    split, table = rule
    t = dict(table)
    v = "*" if split == "const" else f.get(split)
    if v not in t:
        return None
    kind, a = t[v]
    return a if kind == "lit" else f.get(a)


# ------------------------------------------------------------------ action: recolour (or erase) objects
def fit_recolour(xs, ys, bgv):
    from .ops import resolve_bg
    if any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)) or all(x == y for x, y in zip(xs, ys)):
        return []
    items = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        bg = resolve_bg(bgv, x)
        sc = scene(x, bg)
        if sc is None:
            return []
        objs, feats = sc
        cells = set()
        for o, f in zip(objs, feats):
            ts = {y[r][c] for r, c in o}
            if len(ts) != 1:
                return []
            f = dict(f, bg=bg)
            items.append((f, ts.pop(), i))
            cells.update(o)
        if any(x[r][c] != y[r][c] for r in range(len(x)) for c in range(len(x[0])) if (r, c) not in cells):
            return []
    rules = learn(items, COLOUR_ROLES + ("bg",), unchanged_role="colour")
    if not rules:
        rules = learn_selected(items, COLOUR_ROLES + ("bg",), lambda f, t: f["colour"] == t)
    out = []
    for rule in rules:
        spec = (bgv, rule)
        if all(apply_recolour(x, spec) == y for x, y in zip(xs, ys)):
            out.append(spec)
    return out


def apply_recolour(g, spec):
    from .ops import resolve_bg
    bgv, rule = spec
    bg = resolve_bg(bgv, g)
    sc = scene(g, bg)
    if sc is None:
        return None
    out = [list(r) for r in g]
    for o, f in zip(*sc):
        t = predict(rule, dict(f, bg=bg), "colour")
        if t is None:
            return None
        for r, c in o:
            out[r][c] = t
    out = tuple(map(tuple, out))
    return out if out != g else None


# ------------------------------------------------------------------ action: move objects until contact
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
DIR_ROLES = ("stay", "dir_nearest", "dir_largest", "dir_border", "dir_same_colour", "dir_other_colour")


def _dir_to(box, other):
    """Direction from box to other when they overlap in rows (L/R) or columns (U/D)."""
    r0, c0, r1, c1 = box
    a0, b0, a1, b1 = other
    if not (a1 < r0 or a0 > r1):
        if b0 > c1:
            return "R"
        if b1 < c0:
            return "L"
    if not (b1 < c0 or b0 > c1):
        if a0 > r1:
            return "D"
        if a1 < r0:
            return "U"
    return None


def move_feats(g, objs, feats):
    H, W = len(g), len(g[0])
    boxes = [P.bbox(o) for o in objs]
    sizes = [len(o) for o in objs]
    cols = [f["colour"] for f in feats]
    out = []
    for i, f in enumerate(feats):
        b = boxes[i]
        def toward(cands):
            ds = {}
            for j in cands:
                d = _dir_to(b, boxes[j])
                if d:
                    gap = max(boxes[j][0] - b[2], b[0] - boxes[j][2], boxes[j][1] - b[3], b[1] - boxes[j][3])
                    ds.setdefault(gap, set()).add(d)
            if not ds:
                return None
            best = ds[min(ds)]
            return best.pop() if len(best) == 1 else None
        others = [j for j in range(len(objs)) if j != i]
        big = [j for j in others if sizes[j] == max(sizes[k] for k in others)] if others else []
        r0, c0, r1, c1 = b
        gaps = {"U": r0, "D": H - 1 - r1, "L": c0, "R": W - 1 - c1}
        m = min(gaps.values())
        bd = [k for k, v in gaps.items() if v == m]
        out.append(dict(f, stay="none", dir_nearest=toward(others), dir_largest=toward(big) if len(big) == 1 else None,
                        dir_border=bd[0] if len(bd) == 1 else None,
                        dir_same_colour=toward([j for j in others if cols[j] == cols[i]]),
                        dir_other_colour=toward([j for j in others if cols[j] != cols[i]])))
    return out


def _slide(o, d, blocked, H, W):
    dr, dc = DIRS[d]
    cur = list(o)
    while True:
        nxt = [(r + dr, c + dc) for r, c in cur]
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in blocked for r, c in nxt):
            return cur
        cur = nxt


def apply_move(g, spec):
    from .ops import resolve_bg
    bgv, rule = spec
    bg = resolve_bg(bgv, g)
    sc = scene(g, bg)
    if sc is None:
        return None
    objs, feats = sc
    mf = move_feats(g, objs, feats)
    H, W = len(g), len(g[0])
    moves = []
    for o, f in zip(objs, mf):
        d = predict(rule, f, "stay")
        if d is None:
            return None
        if d != "none":
            moves.append((o, d))
    if not moves:
        return None
    moving = {rc for o, _ in moves for rc in o}
    blocked = {(r, c) for r in range(H) for c in range(W) if g[r][c] != bg and (r, c) not in moving}
    out = [list(r) for r in g]
    for o, _ in moves:
        for r, c in o:
            out[r][c] = bg
    # movers nearest to their destination go first; a landed mover blocks the ones behind it
    phases = max(len(d.split("+")) for _, d in moves)
    cur = {k: list(o) for k, (o, _) in enumerate(moves)}
    for ph in range(phases):
        def lead(k):
            d = moves[k][1].split("+")
            if ph >= len(d):
                return 0
            dr, dc = DIRS[d[ph]]
            return -max(r * dr + c * dc for r, c in cur[k])
        placed = set()
        for k in sorted(cur, key=lead):
            d = moves[k][1].split("+")
            others = blocked | {rc for j in cur if j != k and j not in placed for rc in cur[j]} | placed
            if ph < len(d):
                cur[k] = _slide(cur[k], d[ph], others - set(cur[k]), H, W)
            placed |= set(cur[k])
    for k, (o, _) in enumerate(moves):
        for (r, c), (a, b) in zip(o, cur[k]):
            out[a][b] = g[r][c]
    out = tuple(map(tuple, out))
    return out if out != g else None


def fit_move(xs, ys, bgv):
    from .ops import resolve_bg
    if any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)) or all(x == y for x, y in zip(xs, ys)):
        return []
    # every object moves the same way (one or two phases)
    for d in list(DIRS) + [a + "+" + b for a in "UD" for b in "LR"] + [b + "+" + a for a in "UD" for b in "LR"]:
        spec = (bgv, ("const", (("*", ("lit", d)),)))
        if all(apply_move(x, spec) == y for x, y in zip(xs, ys)):
            return [spec]
    items = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        bg = resolve_bg(bgv, x)
        sc = scene(x, bg)
        if sc is None:
            return []
        objs, feats = sc
        mf = move_feats(x, objs, feats)
        H, W = len(x), len(x[0])
        for o, f in zip(objs, mf):
            if all(y[r][c] == x[r][c] for r, c in o):
                items.append((f, "none", i)); continue
            # the direction of its displacement: the object found translated along one axis
            found = None
            for d, (dr, dc) in DIRS.items():
                for k in range(1, max(H, W)):
                    end = [(r + dr * k, c + dc * k) for r, c in o]
                    if any(not (0 <= a < H and 0 <= b < W) for a, b in end):
                        break
                    if all(y[a][b] == x[r][c] for (r, c), (a, b) in zip(o, end)):
                        found = d; break
                if found:
                    break
            if found is None:
                # two-phase move: the object found translated along both axes
                for dr, dc in ((a, b) for a in range(-H + 1, H) for b in range(-W + 1, W) if a and b):
                    end = [(r + dr, c + dc) for r, c in o]
                    if all(0 <= a < H and 0 <= b < W and y[a][b] == x[r][c] for (r, c), (a, b) in zip(o, end)):
                        v = "D" if dr > 0 else "U"; h = "R" if dc > 0 else "L"
                        found = v + "+" + h
                        break
            if found is None:
                return []
            items.append((f, found, i))
    if all(t == "none" for _, t, _ in items):
        return []
    rules = learn(items, DIR_ROLES, unchanged_role="stay") or \
        learn_selected(items, DIR_ROLES, lambda f, t: t == "none")
    if not rules and any("+" in t for _, t, _ in items):
        # the reverse phase order explains the same displacement
        items2 = [(f, "+".join(reversed(t.split("+"))) if "+" in t else t, i) for f, t, i in items]
        rules = learn(items2, DIR_ROLES, unchanged_role="stay") or \
            learn_selected(items2, DIR_ROLES, lambda f, t: t == "none")
    out = []
    for rule in rules:
        spec = (bgv, rule)
        if all(apply_move(x, spec) == y for x, y in zip(xs, ys)):
            out.append(spec)
    return out


# ------------------------------------------------------------------ action: rays from objects (OPT201)
DIR8 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1),
        "UL": (-1, -1), "UR": (-1, 1), "DL": (1, -1), "DR": (1, 1)}
OPP = {"U": "D", "D": "U", "L": "R", "R": "L", "UL": "DR", "DR": "UL", "UR": "DL", "DL": "UR"}
RAY_ROLES = ("no_rays", "points", "container_points", "along", "across", "to_nearest", "to_largest", "to_other_colour", "to_same_colour", "away_nearest",
             "to_border")


def _points(o):
    """The side where the object narrows to one cell while the opposite side is wider."""
    rows = [r for r, _ in o]; cols = [c for _, c in o]
    ends = {"U": sum(r == min(rows) for r in rows), "D": sum(r == max(rows) for r in rows),
            "L": sum(c == min(cols) for c in cols), "R": sum(c == max(cols) for c in cols)}
    p = [d for d in ends if ends[d] == 1 and ends[OPP[d]] > 1]
    return p[0] if len(p) == 1 else None


def ray_feats(g, objs, feats):
    mf = move_feats(g, objs, feats)
    boxes = [P.bbox(o) for o in objs]
    pts = [_points(o) for o in objs]
    out = []
    for i, (o, f) in enumerate(zip(objs, mf)):
        t = lambda d: (d,) if d else None
        b = boxes[i]
        cont = [j for j in range(len(objs)) if j != i and boxes[j][0] <= b[0] and boxes[j][1] <= b[1] and
                boxes[j][2] >= b[2] and boxes[j][3] >= b[3] and boxes[j] != b]
        cp = pts[min(cont, key=lambda j: len(objs[j]))] if cont else None
        h, w = b[2] - b[0] + 1, b[3] - b[1] + 1
        along = ("L", "R") if h == 1 and w > 1 else ("D", "U") if w == 1 and h > 1 else None
        across = ("D", "U") if h == 1 and w > 1 else ("L", "R") if w == 1 and h > 1 else None
        out.append(dict(f, no_rays=(), points=t(pts[i]), container_points=t(cp), along=along, across=across, to_nearest=t(f["dir_nearest"]), to_largest=t(f["dir_largest"]),
                        to_other_colour=t(f["dir_other_colour"]), to_same_colour=t(f["dir_same_colour"]),
                        away_nearest=t(OPP[f["dir_nearest"]]) if f["dir_nearest"] else None,
                        to_border=t(f["dir_border"])))
    return out


def _ray_cells(g, o, d, bg, mode):
    """Background cells a ray in direction d from the object's cells covers. mode 'stop': each ray
    stops at the first non-background cell; 'through': continues to the border over background."""
    H, W = len(g), len(g[0])
    s = set(o)
    dr, dc = DIR8[d]
    cells = []
    for r, c in o:
        a, b = r + dr, c + dc
        if (a, b) in s:
            continue
        while 0 <= a < H and 0 <= b < W:
            if g[a][b] != bg:
                if mode == "stop":
                    break
            else:
                cells.append((a, b))
            a, b = a + dr, b + dc
    return cells


def apply_rays(g, spec):
    from .ops import resolve_bg
    bgv, mode, drule, crule = spec
    bg = resolve_bg(bgv, g)
    sc = scene(g, bg)
    if sc is None:
        return None
    objs, feats = sc
    rf = ray_feats(g, objs, feats)
    out = [list(r) for r in g]
    did = False
    for o, f in zip(objs, rf):
        ds = predict(drule, f, "no_rays")
        if ds is None:
            return None
        if not ds:
            continue
        col = predict(crule, dict(f, bg=bg))
        if col is None:
            return None
        for d in ds:
            for a, b in _ray_cells(g, o, d, bg, mode):
                out[a][b] = col; did = True
    return tuple(map(tuple, out)) if did else None


def fit_rays(xs, ys, bgv):
    from .ops import resolve_bg
    if any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)) or all(x == y for x, y in zip(xs, ys)):
        return []
    out = []
    for mode in ("stop", "through"):
        items, citems, ok = [], [], True
        for i, (x, y) in enumerate(zip(xs, ys)):
            bg = resolve_bg(bgv, x)
            if any(x[r][c] != bg and x[r][c] != y[r][c] for r in range(len(x)) for c in range(len(x[0]))):
                ok = False; break   # rays only paint background
            sc = scene(x, bg)
            if sc is None:
                ok = False; break
            objs, feats = sc
            covered = set()
            for o, f in zip(objs, ray_feats(x, objs, feats)):
                ds, cols = [], set()
                for d in DIR8:
                    cells = _ray_cells(x, o, d, bg, mode)
                    vals = {y[a][b] for a, b in cells}
                    if cells and len(vals) == 1 and bg not in vals:
                        ds.append(d); cols |= vals; covered.update(cells)
                if len(cols) > 1:
                    ok = False; break
                items.append((f, tuple(sorted(ds)), i))
                if ds:
                    citems.append((dict(f, bg=bg), cols.pop(), i))
            if not ok:
                break
            changed = {(r, c) for r in range(len(x)) for c in range(len(x[0])) if x[r][c] != y[r][c]}
            if not changed <= covered:
                ok = False; break
        if not ok or not citems:
            continue
        drules = learn(items, RAY_ROLES, unchanged_role="no_rays") or \
            learn_selected(items, RAY_ROLES, lambda f, t: t == ())
        crules = learn(citems, COLOUR_ROLES, unchanged_role=None, max_rules=2)
        for dr_ in drules:
            for cr in crules:
                spec = (bgv, mode, dr_, cr)
                if all(apply_rays(x, spec) == y for x, y in zip(xs, ys)):
                    out.append(spec); break
            if out:
                break
        if out:
            break
    return out


# ------------------------------------------------------------------ action: connect aligned objects (OPT201)
def _centre(o):
    r0, c0, r1, c1 = P.bbox(o)
    if (r0 + r1) % 2 or (c0 + c1) % 2:
        return None
    return (r0 + r1) // 2, (c0 + c1) // 2


def _links(g, objs, bg, pairs, orient):
    """Lines between aligned object centres whose path (outside the two objects) is all background.
    pairs 'same' (same colour) / 'any'; orient 'hv' / 'diag' / 'all'. Only nearest partners link."""
    cs = [_centre(o) for o in objs]
    col = [g[o[0][0]][o[0][1]] for o in objs]
    own = {}
    for i, o in enumerate(objs):
        for rc in o:
            own[rc] = i
    lines = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if cs[i] is None or cs[j] is None or (pairs == "same" and col[i] != col[j]):
                continue
            (a, b), (c, d) = cs[i], cs[j]
            dr, dc = c - a, d - b
            hv = dr == 0 or dc == 0
            dg = abs(dr) == abs(dc) and dr != 0
            if not ((hv and orient in ("hv", "all")) or (dg and orient in ("diag", "all"))):
                continue
            n = max(abs(dr), abs(dc))
            sr, sc = (dr > 0) - (dr < 0), (dc > 0) - (dc < 0)
            path, ok = [], True
            for k in range(1, n):
                p = (a + sr * k, b + sc * k)
                w = own.get(p)
                if w in (i, j):
                    continue
                if g[p[0]][p[1]] != bg:
                    ok = False; break
                path.append(p)
            if ok and path:
                lines.append((i, j, path))
    return lines, col


def apply_connect(g, spec):
    from .ops import resolve_bg
    bgv, pairs, orient, colrule = spec
    bg = resolve_bg(bgv, g)
    sc = scene(g, bg)
    if sc is None:
        return None
    objs, feats = sc
    lines, col = _links(g, objs, bg, pairs, orient)
    if not lines:
        return None
    out = [list(r) for r in g]
    for i, j, path in lines:
        c = col[i] if colrule == ("role", "colour") else colrule[1] if colrule[0] == "lit" else None
        if colrule == ("role", "other"):
            c = col[j] if col[j] != col[i] else None
        if c is None:
            return None
        for r, k in path:
            out[r][k] = c
    return tuple(map(tuple, out))


def fit_connect(xs, ys, bgv):
    from .ops import resolve_bg
    if any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)) or all(x == y for x, y in zip(xs, ys)):
        return []
    out = []
    lit = {v for x, y in zip(xs, ys) for rx, ry in zip(x, y) for a, v in zip(rx, ry) if a != v}
    for pairs in ("same", "any"):
        for orient in ("hv", "diag", "all"):
            for colrule in [("role", "colour")] + [("lit", v) for v in sorted(lit)][:2]:
                spec = (bgv, pairs, orient, colrule)
                if all(apply_connect(x, spec) == y for x, y in zip(xs, ys)):
                    out.append(spec)
                    return out
    return out
