"""cell_rules.py  (LUCA V31 prototype: learned cell-level rule finder)
====================================================================
For same-shape tasks. Every cell of every demonstration input becomes a
feature vector (colour, 8 neighbours, object size/shape, enclosed-background,
nearest coloured cell in each direction, mirror-image cells, optionally
position). A small decision tree must reproduce every demonstration output
exactly. It is accepted only if it ALSO predicts each demonstration pair
exactly when that pair is held out (leave-one-out): a cheap guard against
memorising the demonstrations.

Standard library only.
"""
from __future__ import annotations
import time
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from .grid_world import Grid
from .primitives import Pair

MAX_DEPTH = 9
NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bg(g: Grid) -> int:
    return _P.background(g, 'most_common_first')  # Phase 2: canonical perception (was 1 lines)


def _components(g: Grid):
    h, w = len(g), len(g[0])
    cid = [[-1] * w for _ in range(h)]
    comps = []
    for i in range(h):
        for j in range(w):
            if cid[i][j] != -1:
                continue
            col = g[i][j]
            k = len(comps)
            stack = [(i, j)]
            cid[i][j] = k
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in DIRS4:
                    x, y = a + da, b + db
                    if 0 <= x < h and 0 <= y < w and cid[x][y] == -1 and g[x][y] == col:
                        cid[x][y] = k
                        stack.append((x, y))
            rs = [a for a, _ in cells]
            cs = [b for _, b in cells]
            touches = int(min(rs) == 0 or max(rs) == h - 1 or min(cs) == 0 or max(cs) == w - 1)
            comps.append((col, len(cells), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1, touches))
    return cid, comps


def _enclosed_bg(g: Grid, bg: int):
    h, w = len(g), len(g[0])
    seen = [[False] * w for _ in range(h)]
    stack = [(i, j) for i in range(h) for j in range(w)
             if (i in (0, h - 1) or j in (0, w - 1)) and g[i][j] == bg]
    for i, j in stack:
        seen[i][j] = True
    while stack:
        a, b = stack.pop()
        for da, db in DIRS4:
            x, y = a + da, b + db
            if 0 <= x < h and 0 <= y < w and not seen[x][y] and g[x][y] == bg:
                seen[x][y] = True
                stack.append((x, y))
    return seen


def cell_features(g: Grid, tier: int) -> List[Tuple[int, ...]]:
    h, w = len(g), len(g[0])
    bg = _bg(g)
    cid, comps = _components(g)
    nonbg_sizes = sorted({c[1] for c in comps if c[0] != bg}, reverse=True)
    enclosed = _enclosed_bg(g, bg)
    sq = h == w
    rows_any = [int(any(v != bg for v in g[i])) for i in range(h)]
    cols_any = [int(any(g[i][j] != bg for i in range(h))) for j in range(w)]
    out = []
    for i in range(h):
        for j in range(w):
            col = g[i][j]
            f = [col]
            for da, db in NEIGH8:
                x, y = i + da, j + db
                f.append(g[x][y] if 0 <= x < h and 0 <= y < w else -1)
            c = comps[cid[i][j]]
            size = c[1]
            rank = nonbg_sizes.index(size) if col != bg and size in nonbg_sizes else -1
            f += [size, rank, int(col != bg and size == nonbg_sizes[0]) if nonbg_sizes else 0,
                  int(col != bg and size == nonbg_sizes[-1]) if nonbg_sizes else 0,
                  c[4], c[2], c[3], int(col == bg), int(enclosed[i][j])]
            for da, db in DIRS4:
                x, y = i + da, j + db
                val = -1
                while 0 <= x < h and 0 <= y < w:
                    if g[x][y] != bg:
                        val = g[x][y]
                        break
                    x += da
                    y += db
                f.append(val)
            n4 = sum(1 for da, db in DIRS4 if 0 <= i + da < h and 0 <= j + db < w and g[i + da][j + db] != bg)
            n8 = sum(1 for v in f[1:9] if v not in (-1, bg))
            f += [n4, n8, g[i][w - 1 - j], g[h - 1 - i][j], g[h - 1 - i][w - 1 - j],
                  g[j][i] if sq else -1, rows_any[i], cols_any[j]]
            if tier >= 2:
                f += [i, j, h - 1 - i, w - 1 - j, i % 2, j % 2, i % 3, j % 3]
                f += [int(k in g[i]) for k in range(10)]
                f += [int(any(g[a][j] == k for a in range(h))) for k in range(10)]
            out.append(tuple(f))
    return out


# ---------------------------------------------------------------- decision tree

def _gini_split(groups_l: Counter, groups_r: Counter) -> float:
    nl, nr = sum(groups_l.values()), sum(groups_r.values())
    if nl == 0 or nr == 0:
        return 1e9
    gl = nl - sum(v * v for v in groups_l.values()) / nl
    gr = nr - sum(v * v for v in groups_r.values()) / nr
    return gl + gr


def _best_test(rows):
    total = Counter()
    for _, lab, n in rows:
        total[lab] += n
    nfeat = len(rows[0][0])
    best = None
    for j in range(nfeat):
        byval: Dict[int, Counter] = defaultdict(Counter)
        for f, lab, n in rows:
            byval[f[j]][lab] += n
        if len(byval) < 2:
            continue
        vals = sorted(byval)
        if len(vals) <= 12:
            for v in vals:
                left = byval[v]
                right = total - left
                s = _gini_split(left, right)
                if best is None or s < best[0] - 1e-12:
                    best = (s, 'eq', j, v)
        left = Counter()
        for v in vals[:-1]:
            left += byval[v]
            s = _gini_split(left, total - left)
            if best is None or s < best[0] - 1e-12:
                best = (s, 'le', j, v)
    return best


def _build(rows, depth, deadline):
    labels = {lab for _, lab, _ in rows}
    if len(labels) == 1:
        return ('leaf', next(iter(labels)))
    if depth == 0 or time.monotonic() > deadline:
        return None
    best = _best_test(rows)
    if best is None:
        return None
    _, kind, j, v = best
    if kind == 'eq':
        l = [r for r in rows if r[0][j] == v]
        r_ = [r for r in rows if r[0][j] != v]
    else:
        l = [r for r in rows if r[0][j] <= v]
        r_ = [r for r in rows if r[0][j] > v]
    if not l or not r_:
        return None
    a = _build(l, depth - 1, deadline)
    if a is None:
        return None
    b = _build(r_, depth - 1, deadline)
    if b is None:
        return None
    return ('node', kind, j, v, a, b)


def _walk(tree, f):
    while tree[0] == 'node':
        _, kind, j, v, a, b = tree
        yes = (f[j] == v) if kind == 'eq' else (f[j] <= v)
        tree = a if yes else b
    return tree[1]


def _labels(pairs: List[Pair], mode: str):
    """Yield (features-agnostic) label per cell: 'kl' = KEEP or literal colour, 'abs' = literal."""
    for x, y in pairs:
        for i, row in enumerate(y):
            for j, v in enumerate(row):
                if mode == 'kl':
                    yield ('K',) if x[i][j] == v else ('L', v)
                else:
                    yield ('L', v)


def _fit(pairs: List[Pair], tier: int, mode: str, deadline: float):
    rows: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
    for (x, _), labs_iter in zip(pairs, _split_labels(pairs, mode)):
        for f, lab in zip(cell_features(x, tier), labs_iter):
            rows[f][lab] += 1
    data = []
    for f, cnt in rows.items():
        if len(cnt) > 1:
            return None  # identical features, different outputs: tier cannot express the rule
        (lab, n), = cnt.items()
        data.append((f, lab, n))
    if not data:
        return None
    tree = _build(data, MAX_DEPTH, deadline)
    if tree is None:
        return None

    def predict(g: Grid) -> Grid:
        h, w = len(g), len(g[0])
        feats = cell_features(g, tier)
        out = []
        for i in range(h):
            row = []
            for j in range(w):
                lab = _walk(tree, feats[i * w + j])
                row.append(g[i][j] if lab[0] == 'K' else lab[1])
            out.append(tuple(row))
        return tuple(out)
    return predict


def _split_labels(pairs, mode):
    it = _labels(pairs, mode)
    for x, _ in pairs:
        yield [next(it) for _ in range(len(x) * len(x[0]))]


def induce(train_pairs: List[Pair], seconds: float = 20.0):
    """Returns (predict, info) if a rule fits all pairs AND survives leave-one-out; else (None, info)."""
    info = {"tried": [], "loo": None}
    if not train_pairs or any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in train_pairs):
        info["reason"] = "not same-shape"
        return None, info
    deadline = time.monotonic() + seconds
    n = len(train_pairs)
    for tier, mode in ((1, 'kl'), (1, 'abs'), (2, 'kl'), (2, 'abs')):
        if time.monotonic() > deadline:
            break
        pred = _fit(train_pairs, tier, mode, deadline)
        if pred is None:
            info["tried"].append((tier, mode, "no_fit"))
            continue
        ok = 0
        for i in range(n):
            if time.monotonic() > deadline:
                break
            sub = train_pairs[:i] + train_pairs[i + 1:]
            p = _fit(sub, tier, mode, deadline) if sub else None
            if p is not None and p(train_pairs[i][0]) == train_pairs[i][1]:
                ok += 1
        info["tried"].append((tier, mode, f"loo {ok}/{n}"))
        if ok == n and n >= 2:
            info["loo"] = (tier, mode, ok, n)
            return pred, info
    return None, info


# ---------------------------------------------------------------- integration stage
import os
from dataclasses import replace
from .primitives import Law
from .dsl import perception as _P  # Phase 2 canonical perception layer

ENABLED = os.environ.get("LUCA_CELLRULES", "1") != "0"
SECONDS = float(os.environ.get("LUCA_CELLRULES_SECONDS", "20"))


def build_candidates(train_pairs: List[Pair], seconds: float = SECONDS):
    """All configurations that fit every demonstration pair and survive leave-one-out on
    at least n-1 pairs, best first: (loo, -tier, keep-mode first)."""
    n = len(train_pairs)
    if n < 2 or any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in train_pairs):
        return []
    deadline = time.monotonic() + seconds
    out = []
    for tier, mode in ((1, 'kl'), (1, 'abs'), (2, 'kl'), (2, 'abs')):
        if time.monotonic() > deadline:
            break
        pred = _fit(train_pairs, tier, mode, deadline)
        if pred is None:
            continue
        ok = 0
        for i in range(n):
            sub = train_pairs[:i] + train_pairs[i + 1:]
            p = _fit(sub, tier, mode, deadline)
            if p is not None and p(train_pairs[i][0]) == train_pairs[i][1]:
                ok += 1
        if ok >= n - 1:
            out.append(dict(loo=ok, n=n, tier=tier, mode=mode, predict=pred))
    out.sort(key=lambda d: (-d["loo"], d["tier"], d["mode"] != 'kl'))
    return out


def upgrade_many(train_pairs: List[Pair], test_inputs: List[Grid], results: list) -> list:
    """Only touches pure fallbacks (failure_category set). Best cell rule becomes attempt 1;
    attempt 2 is a real partial-fit guess if one exists, else the next distinct cell rule,
    else the previous first guess."""
    if not ENABLED or not any(r.failure_category is not None for r in results):
        return results
    try:
        cands = build_candidates(train_pairs)
    except Exception:
        return results
    if not cands:
        return results
    upgraded = []
    for g, r in zip(test_inputs, results):
        if r.failure_category is None:
            upgraded.append(r)
            continue
        cpreds = []
        for c in cands:
            try:
                p = c["predict"](g)
            except Exception:
                continue
            if p != g and all(p != q for q, _ in cpreds):
                cpreds.append((p, c))
        if not cpreds:
            upgraded.append(r)
            continue
        def mk(c):
            return Law(name=f"cell_rule:t{c['tier']}_{c['mode']}", predict=c["predict"],
                       description=f"CELL-RULE(loo {c['loo']}/{c['n']}, tier {c['tier']}, {c['mode']})",
                       constant_bits=8, op_count=1)
        preds = [cpreds[0][0]]; laws = [mk(cpreds[0][1])]
        second = None
        for k, p in enumerate(r.predictions):
            nm = r.laws[k].name if k < len(r.laws) else ''
            if nm.startswith('partial_fit:') and p not in preds:
                second = (p, r.laws[k]); break
        if second is None and len(cpreds) > 1:
            second = (cpreds[1][0], mk(cpreds[1][1]))
        if second is None:
            for k, p in enumerate(r.predictions):
                if p not in preds:
                    second = (p, r.laws[k] if k < len(r.laws) else None); break
        if second is not None:
            preds.append(second[0])
            if second[1] is not None:
                laws.append(second[1])
        diag = dict(r.diagnostics or {})
        diag["cell_rules"] = dict(candidates=[(c["loo"], c["tier"], c["mode"]) for c in cands])
        upgraded.append(replace(r, predictions=preds[:2], laws=laws[:2],
                                explanation=r.explanation + " | cell-rule stage supplied attempt 1.",
                                diagnostics=diag))
    return upgraded
