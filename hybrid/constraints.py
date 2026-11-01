"""LUCA hybrid - what the demonstrations imply about an answer (candidate screening).

Each rule describes a property shared by every demonstration output, as a function of its input:
output size, allowed colours, colours whose cells never change, output symmetry, output = a piece
of the input. A rule is kept only if it is
  * true on every demonstration, and
  * predictive: fitted on all demonstrations but one, it holds on the one left out (every fold),
so that a rule fitted by coincidence is not used. A neural candidate that breaks a kept rule is
dropped (or moved behind the candidates that keep all rules).

    rules = fit_rules(task)          # list of (name, check) with check(test_input, grid) -> bool
    ok = passes(rules, test_input, grid)
"""
from collections import Counter


def _shape(g):
    return (len(g), len(g[0]) if g else 0)


def _colours(g):
    return {v for r in g for v in r}


def _bg(g):
    return Counter(v for r in g for v in r).most_common(1)[0][0]


# ---------------------------------------------------------------- size rules: (in) -> allowed (h, w)
def _nonbg_bbox(g):
    b = _bg(g)
    cells = [(i, j) for i, r in enumerate(g) for j, v in enumerate(r) if v != b]
    if not cells:
        return None
    rs = [i for i, _ in cells]; cs = [j for _, j in cells]
    return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)


def _size_rules(pairs):
    """Candidate size laws, each fitted from `pairs` -> function in -> (h, w), or None."""
    laws = {}
    if all(_shape(i) == _shape(o) for i, o in pairs):
        laws["size=input"] = lambda x: _shape(x)
    outs = {_shape(o) for _, o in pairs}
    if len(outs) == 1:
        s = next(iter(outs)); laws["size=constant"] = lambda x, s=s: s
    if all(_shape(i) == _shape(o)[::-1] for i, o in pairs):
        laws["size=transposed"] = lambda x: _shape(x)[::-1]
    rat = {(_shape(o)[0] / _shape(i)[0], _shape(o)[1] / _shape(i)[1]) for i, o in pairs}
    if len(rat) == 1:
        (rh, rw), = rat
        if (rh, rw) != (1.0, 1.0):
            laws["size=ratio"] = lambda x, rh=rh, rw=rw: (round(_shape(x)[0] * rh), round(_shape(x)[1] * rw))
    off = {(_shape(o)[0] - _shape(i)[0], _shape(o)[1] - _shape(i)[1]) for i, o in pairs}
    if len(off) == 1:
        (dh, dw), = off
        if (dh, dw) != (0, 0):
            laws["size=offset"] = lambda x, dh=dh, dw=dw: (_shape(x)[0] + dh, _shape(x)[1] + dw)
    if all(_nonbg_bbox(i) == _shape(o) for i, o in pairs):
        laws["size=content_bbox"] = lambda x: _nonbg_bbox(x)
    return laws


# ---------------------------------------------------------------- other rules
def _new_colours(pairs):
    return set().union(*[_colours(o) - _colours(i) for i, o in pairs])


def _fixed_colours(pairs):
    """Colours whose input cells are unchanged in the output (same-size pairs), seen in >= 2 pairs."""
    if not all(_shape(i) == _shape(o) for i, o in pairs):
        return set()
    seen, broken = Counter(), set()
    for i, o in pairs:
        for r1, r2 in zip(i, o):
            for a, b in zip(r1, r2):
                if a != b:
                    broken.add(a)
        for c in _colours(i):
            seen[c] += 1
    return {c for c, n in seen.items() if n >= FIXED_MIN_SEEN and c not in broken}


SYMS = {
    "lr": lambda g: [r[::-1] for r in g],
    "ud": lambda g: g[::-1],
    "rot180": lambda g: [r[::-1] for r in g[::-1]],
    "transpose": lambda g: [list(r) for r in zip(*g)],
}


def _is_sub(small, big):
    h, w = _shape(small); H, W = _shape(big)
    if h > H or w > W:
        return False
    for a in range(H - h + 1):
        for b in range(W - w + 1):
            if all(big[a + i][b:b + w] == small[i] for i in range(h)):
                return True
    return False


def _candidate_rules(pairs):
    """Every rule fitted from `pairs`: {name: check(x, grid) -> bool}."""
    rules = {}
    for name, law in _size_rules(pairs).items():
        rules[name] = lambda x, g, law=law: law(x) is not None and _shape(g) == tuple(law(x))
    new = _new_colours(pairs)
    rules["palette"] = lambda x, g, new=new: _colours(g) <= _colours(x) | new
    for c in _fixed_colours(pairs):
        rules[f"fixed:{c}"] = lambda x, g, c=c: _shape(g) == _shape(x) and all(
            b == c for r1, r2 in zip(x, g) for a, b in zip(r1, r2) if a == c)
    for s, f in SYMS.items():
        if all(f(o) == [list(r) for r in o] for _, o in pairs) and not all(
                f(i) == [list(r) for r in i] for i, _ in pairs):
            rules[f"symmetric:{s}"] = lambda x, g, f=f: f(g) == [list(r) for r in g]
    if all(_shape(o) != _shape(i) and _is_sub(o, i) for i, o in pairs):
        rules["piece_of_input"] = lambda x, g: _is_sub(g, x)
    return rules


FIXED_MIN_SEEN = 3
STRONG = ("fixed:", "symmetric:", "size=content_bbox")   # need >= 3 demonstrations


def fit_rules(task, min_pairs=2):
    """The kept rules: true on every demonstration and predictive under leave-one-out. All size
    rules are merged into one: a candidate's size must match ANY kept size law (on square grids
    e.g. 'same as input' and 'constant' both fit the demonstrations but can differ on the test)."""
    pairs = [(p["input"], p["output"]) for p in task["train"]]
    if len(pairs) < min_pairs:
        return []
    full = _candidate_rules(pairs)
    kept, size_checks = [], []
    for name, check in full.items():
        if name.startswith(STRONG) and len(pairs) < 3:
            continue
        ok = all(check(i, o) for i, o in pairs)
        for k in range(len(pairs)):
            if not ok:
                break
            rest = pairs[:k] + pairs[k + 1:]
            fold = _candidate_rules(rest).get(name)
            i, o = pairs[k]
            ok = fold is not None and fold(i, o)
        if ok:
            (size_checks if name.startswith("size=") else kept).append((name, check))
    if size_checks:
        names = "|".join(n for n, _ in size_checks)
        kept.insert(0, (names, lambda x, g, cs=tuple(c for _, c in size_checks): any(c(x, g) for c in cs)))
    return kept


def passes(rules, x, grid):
    if not grid or not grid[0] or any(len(r) != len(grid[0]) for r in grid):
        return False
    try:
        return all(check(x, grid) for _, check in rules)
    except Exception:
        return False


def broken(rules, x, grid):
    out = []
    for name, check in rules:
        try:
            if not check(x, grid):
                out.append(name)
        except Exception:
            out.append(name)
    return out
