"""LUCA DSL - object mapping engine (OPT202): input objects -> output objects.

1. Segmentation   objects of the input (one colour, 8- or 4-connected) on the task background.
2. Matching       every input object is matched to an output object that has its shape under one of
                  the 8 rotations / reflections (or that shape scaled up by 2 or 3), colours allowed to
                  change; the overlapping / nearest candidate wins. Input objects with no match are
                  DELETED. Every output cell must be accounted for by a kept or moved object.
3. Object program each matched pair gives the transforms that explain it:
                    shape      D8 transform (+ scale)
                    colour     'same' | literal map | role (the colour it touches, its nearest
                               object's, ...: dsl/objlearn.py features)
                    placement  anchor kept ('tl','tr','bl','br','c') | constant offset |
                               slide in a direction until contact
                  A class of objects (a learned property value) shares one program: the
                  intersection of what its members allow. Classes are found with the object-rule
                  learner's features; a class with no change keeps its objects.
4. Verification   the program is applied to every demonstration input and must reproduce its output.

spec = (bgv, conn, split, table)   table: ((value, program), ...); program = ('del',) or
       (transform, scale, colour rule, placement)
"""
from collections import Counter
from . import perception as P
from .evidence import enough

D8 = ("id", "r90", "r180", "r270", "fh", "fv", "tr", "atr")


def _tf(cells, t):
    """Transform (r, c, v) cells by t, normalised to origin."""
    out = []
    for r, c, v in cells:
        if t == "id":
            a, b = r, c
        elif t == "r90":
            a, b = c, -r
        elif t == "r180":
            a, b = -r, -c
        elif t == "r270":
            a, b = -c, r
        elif t == "fh":
            a, b = r, -c
        elif t == "fv":
            a, b = -r, c
        elif t == "tr":
            a, b = c, r
        else:
            a, b = -c, -r
        out.append((a, b, v))
    r0 = min(a for a, _, _ in out); c0 = min(b for _, b, _ in out)
    return tuple(sorted((a - r0, b - c0, v) for a, b, v in out))


def _scale(cells, k):
    if k == 1:
        return cells
    return tuple(sorted((r * k + i, c * k + j, v) for r, c, v in cells for i in range(k) for j in range(k)))


def _shape(cells):
    return tuple((r, c) for r, c, _ in cells)


def _cells(g, o):
    r0 = min(r for r, _ in o); c0 = min(c for _, c in o)
    return tuple(sorted((r - r0, c - c0, g[r][c]) for r, c in o)), (r0, c0)


def _box(o):
    return P.bbox(o)


ANCHORS = ("tl", "tr", "bl", "br", "c")


def _anchor(box, a):
    r0, c0, r1, c1 = box
    return {"tl": (2 * r0, 2 * c0), "tr": (2 * r0, 2 * c1), "bl": (2 * r1, 2 * c0), "br": (2 * r1, 2 * c1),
            "c": (r0 + r1, c0 + c1)}[a]


def _place_at(shape_cells, anchor, a):
    """Top-left position so that the shape's anchor point a sits at `anchor` (doubled coords)."""
    h = max(r for r, _, _ in shape_cells) + 1; w = max(c for _, c, _ in shape_cells) + 1
    dr, dc = {"tl": (0, 0), "tr": (0, 2 * (w - 1)), "bl": (2 * (h - 1), 0), "br": (2 * (h - 1), 2 * (w - 1)),
              "c": (h - 1, w - 1)}[a]
    r2, c2 = anchor[0] - dr, anchor[1] - dc
    if r2 % 2 or c2 % 2:
        return None
    return r2 // 2, c2 // 2


DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _slide(cells_abs, d, blocked, H, W):
    dr, dc = DIRS[d]
    cur = list(cells_abs)
    for _ in range(max(H, W)):
        nxt = [(r + dr, c + dc) for r, c in cur]
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in blocked for r, c in nxt):
            return cur
        cur = nxt
    return cur


# ------------------------------------------------------------------ explanations by containment
def _fits(tc, pos, y, bg, H, W):
    """Colour map under which the transformed cells sit in y at pos (all on ink), else None."""
    cmap = {}
    for r, c, v in tc:
        a, b = pos[0] + r, pos[1] + c
        if not (0 <= a < H and 0 <= b < W):
            return None
        w = y[a][b]
        if w == bg or cmap.setdefault(v, w) != w:
            return None
    return tuple(sorted(cmap.items()))


def _explain(x, o, y, bg, blocked, max_offsets=12):
    """All (transform, scale, colour map, placements) under which object o (moved / transformed /
    recoloured) is contained in y."""
    H, W = len(x), len(x[0])
    ci, (r0, c0) = _cells(x, o)
    bi = _box(o)
    out = []
    for k in (1, 2, 3):
        for t in D8:
            tc = _scale(_tf(ci, t), k)
            if k > 1 and t != "id" and _scale(_tf(ci, "id"), k) == tc:
                continue
            h = max(r for r, _, _ in tc) + 1; w = max(c for _, c, _ in tc) + 1
            if h > H or w > W:
                continue
            found = {}
            for a in ANCHORS:
                pos = _place_at(tc, _anchor(bi, a), a)
                if pos is not None:
                    cm = _fits(tc, pos, y, bg, H, W)
                    if cm is not None:
                        found.setdefault((pos, cm), set()).add(("anchor", a))
            if (t, k) == ("id", 1):
                # a slide: the object reappears further along one direction (the render stacks
                # the sliding objects and checks the exact stopping places)
                for d, (dr, dc) in DIRS.items():
                    for n in range(1, max(H, W)):
                        pos = (r0 + dr * n, c0 + dc * n)
                        if not (0 <= pos[0] and pos[0] + h <= H and 0 <= pos[1] and pos[1] + w <= W):
                            break
                        cm = _fits(tc, pos, y, bg, H, W)
                        if cm is not None:
                            found.setdefault((pos, cm), set()).add(("slide", d))
            cands = []
            for a in range(H - h + 1):
                for b in range(W - w + 1):
                    cands.append((abs(a - r0) + abs(b - c0), a, b))
            cands.sort()
            n = 0
            for _, a, b in cands:
                cm = _fits(tc, (a, b), y, bg, H, W)
                if cm is not None:
                    found.setdefault(((a, b), cm), set()).add(("offset", a - r0, b - c0))
                    n += 1
                    if n >= max_offsets:
                        break
            for (pos, cm), pls in found.items():
                # a moved object must land on at least one cell that was background in the input
                # (landing only on other objects' cells is a coincidence of colours)
                if pos != (r0, c0) or t != "id" or k != 1:
                    if not any(x[pos[0] + r][pos[1] + c] == bg for r, c, _ in tc):
                        continue
                out.append((t, k, cm, frozenset(pls)))
    return out


def _match(x, y, bg, conn):
    """[(input object, explanations or None)]; None explanations = deleted (its cells became
    background) - an object neither explained nor deleted makes the view inapplicable."""
    H, W = len(x), len(x[0])
    from .objlearn import segment
    ins = segment(x, bg, conn)
    if not ins or len(ins) > 40:
        return None
    occupied = {(r, c) for r in range(H) for c in range(W) if x[r][c] != bg}
    pairs = []
    for o in ins:
        ex = _explain(x, o, y, bg, occupied - set(o))
        if ex:
            pairs.append((o, True, ex))
        elif all(y[r][c] == bg for r, c in o):
            pairs.append((o, None, None))
        else:
            return None
    return pairs


def _class_program(members):
    progs = _class_programs(members, 1)
    return progs[0] if progs else None


def _class_programs(members, limit=6):
    """members: [(explanations, features, x, o_in)] all changing; candidate common programs, best first."""
    common = None
    for ex, f, _, _ in members:
        keys = set()
        for t, k, cm, places in ex:
            for pl in places:
                keys.add((t, k, pl))
        common = keys if common is None else common & keys
        if not common:
            return None
    # colour rule: same / one literal map / role
    def cmap_of(ex, t, k):
        for t2, k2, cm, _ in ex:
            if (t2, k2) == (t, k):
                return dict(cm)
        return None
    # a non-identity transform must be the only one of the 8 explaining the class (with that placement)
    by_rest = {}
    for t, k, pl in common:
        by_rest.setdefault((k, pl), set()).add(t)
    common = {(t, k, pl) for t, k, pl in common if t == "id" or len(by_rest[(k, pl)] - {"id"}) == 1 and "id" not in by_rest[(k, pl)]}
    if not common:
        return []
    pref = sorted(common, key=lambda tkp: (tkp[0] != "id", tkp[1], tkp[2][0] != "anchor", tkp[2][0] == "offset", repr(tkp)))
    found = []
    from .objlearn import COLOUR_ROLES
    for t, k, pl in pref:
        maps = [cmap_of(ex, t, k) for ex, _, _, _ in members]
        if all(all(a == b for a, b in m.items()) for m in maps):
            found.append((t, k, ("same",), pl))
        else:
            merged, ok, seen = {}, True, {}
            for m, (_, _, x_, _) in zip(maps, members):
                for a, b in m.items():
                    if merged.setdefault(a, b) != b:
                        ok = False
                    if a != b:
                        seen.setdefault(a, set()).add(id(x_))
            # evidence: every changed colour entry seen in >= 2 demonstrations
            if ok and any(not enough(len(v), len(v)) for v in seen.values()):
                ok = False
            if ok:
                found.append((t, k, ("map", tuple(sorted(merged.items()))), pl))
            else:
                # single-colour objects taking a role colour
                for role in COLOUR_ROLES[1:]:
                    if all(len(m) == 1 and f.get(role) is not None and list(m.values())[0] == f[role]
                           for m, (_, f, _, _) in zip(maps, members)):
                        found.append((t, k, ("role", role), pl)); break
        if len(found) >= limit:
            break
    return found


def _render(x, bg, conn, assign):
    """assign: [(object, program, features)]; returns the output grid or None."""
    H, W = len(x), len(x[0])
    out = [list(r) for r in x]
    occupied = {(r, c) for r in range(H) for c in range(W) if x[r][c] != bg}
    moving = [(o, p, f) for o, p, f in assign if p is not None]
    for o, p, f in moving:
        for r, c in o:
            out[r][c] = bg
    static = occupied - {rc for o, p, f in moving for rc in o}
    placed = set()
    def order(item):
        o, p, f = item
        if p == ("del",) or p[3][0] != "slide":
            return (0, 0)
        dr, dc = DIRS[p[3][1]]
        return (1, -max(r * dr + c * dc for r, c in o))
    for o, p, f in sorted(moving, key=order):
        if p == ("del",):
            continue
        t, k, crule, pl = p
        ci, (r0, c0) = _cells(x, o)
        tc = _scale(_tf(ci, t), k)
        if crule[0] == "map":
            m = dict(crule[1])
            tc = tuple((r, c, m.get(v, v)) for r, c, v in tc)
        elif crule[0] == "role":
            v = f.get(crule[1])
            if v is None:
                return None
            tc = tuple((r, c, v) for r, c, _ in tc)
        if pl[0] == "anchor":
            pos = _place_at(tc, _anchor(_box(o), pl[1]), pl[1])
            if pos is None:
                return None
        elif pl[0] == "offset":
            pos = (r0 + pl[1], c0 + pl[2])
        else:
            end = _slide(o, pl[1], static | placed, H, W)
            pos = (min(r for r, _ in end), min(c for _, c in end))
        for r, c, v in tc:
            a, b = pos[0] + r, pos[1] + c
            if not (0 <= a < H and 0 <= b < W):
                return None
            out[a][b] = v
            placed.add((a, b))
    return tuple(map(tuple, out))


def _features(x, bg, conn, objs):
    from .objlearn import scene
    sc = scene(x, bg, conn)
    if sc is None:
        return None
    o2, feats = sc
    idx = {frozenset(o): f for o, f in zip(o2, feats)}
    return [idx.get(frozenset(o)) for o in objs]


SPLITS = ("const", "is_largest", "is_smallest", "shape_unique", "colour_unique", "shape_count", "colour_count",
          "touches_border", "is_contained", "contains_any", "touches_any", "has_hole", "is_line", "is_rect",
          "n_holes", "shape_d8", "colour", "size_rank", "size")   # structural first, literal values last


def fit(xs, ys, bgv):
    from .ops import resolve_bg
    if all(x == y for x, y in zip(xs, ys)) or any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(xs, ys)):
        return []
    out = []
    for conn in (8, 4, "colour"):
        records, ok = [], True   # (features, kind, member) ; kind 'keep' | 'del' | 'chg'
        for i, (x, y) in enumerate(zip(xs, ys)):
            bg = resolve_bg(bgv, x)
            pairs = _match(x, y, bg, conn)
            if pairs is None:
                ok = False; break
            feats = _features(x, bg, conn, [o for o, _, _ in pairs])
            if feats is None or any(f is None for f in feats):
                ok = False; break
            for (o, q, ex), f in zip(pairs, feats):
                if q is None:
                    records.append((f, "del", None, i))
                elif any(t == "id" and k == 1 and dict(cm) == {v: v for v, _ in cm} and ("offset", 0, 0) in pl
                         for t, k, cm, pl in ex):
                    records.append((f, "keep", None, i))
                else:
                    records.append((f, "chg", (ex, f, x, o), i))
        if not ok or all(kind == "keep" for _, kind, _, _ in records):
            continue
        for split in SPLITS:
            groups = {}
            for f, kind, mem, i in records:
                v = "*" if split == "const" else f.get(split)
                groups.setdefault(v, []).append((kind, mem, i))
            if None in groups:
                continue
            table, good = [], True
            for v, rs in sorted(groups.items(), key=lambda kv: repr(kv[0])):
                kinds = {k for k, _, _ in rs}
                if kinds == {"keep"}:
                    continue
                if not enough({i for _, _, i in rs}, len(rs)):
                    good = False; break
                if kinds == {"del"}:
                    table.append((v, [("del",)])); continue
                if "del" in kinds:
                    good = False; break
                progs = _class_programs([m for k, m, _ in rs if k == "chg"])
                if not progs:
                    good = False; break
                table.append((v, progs))
            if not good or not table:
                continue
            # the first candidate of every class, then each class's alternatives in turn
            base = [(v, ps[0]) for v, ps in table]
            trials = [base] + [base[:n] + [(v, alt)] + base[n + 1:] for n, (v, ps) in enumerate(table) for alt in ps[1:]]
            for tab in trials:
                spec = (bgv, conn, split, tuple(tab))
                if all(apply_spec(x, spec) == y for x, y in zip(xs, ys)):
                    out.append(spec)
                    break
            if out:
                break
        if out:
            break
    return out


def apply_spec(g, spec):
    from .ops import resolve_bg
    bgv, conn, split, table = spec
    bg = resolve_bg(bgv, g)
    from .objlearn import segment
    objs = segment(g, bg, conn)
    if not objs or len(objs) > 40:
        return None
    feats = _features(g, bg, conn, objs)
    if feats is None or any(f is None for f in feats):
        return None
    t = dict(table)
    assign = []
    for o, f in zip(objs, feats):
        v = "*" if split == "const" else f.get(split)
        assign.append((o, t.get(v), f))
    if all(p is None for _, p, _ in assign):
        return None
    res = _render(g, bg, conn, assign)
    return res if res != g else None
