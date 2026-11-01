"""stateful_swarm.py  (LUCA V31 prototype: stateful agents with signals)
======================================================================
Extends the iterated local-rule swarm (swarm_ca.py) with SIGNALS, the grid
analogue of pheromone trails. Every cell is still an agent applying the same
small rule, but besides its colour and its neighbours it can sense:

  los[d]   colour of the first non-background cell in each of 8 directions
           (a signal that travels through empty cells; it equals the fixed
           point of passing a message from neighbour to neighbour)
  outside  for an empty cell: is it connected to the grid border through empty
           cells? (the signal that separates 'enclosed' from 'open')
  born     has this cell (or a neighbour) been changed by the swarm itself?
           (a mark left in the environment; original cells carry no mark)

The rule is a decision tree over these local signals. It may also COPY a signal
(paint the colour seen in direction d). It is applied to all cells at once,
repeatedly, until nothing changes.

Learning is the same conservative iterative self-training as swarm_ca:
label start and end states with the demonstrated end colours, take the majority
label where identical views disagree (KEEP wins ties), simulate, add the
intermediate states, repeat. A rule is accepted only if it reproduces every
demonstration exactly and survives leave-one-out on at least n-1 pairs.
Standard library only.
"""
from __future__ import annotations
import os
import time
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from .grid_world import Grid
from .primitives import Pair, Law
from .cell_rules import _build, _walk
from .dsl import perception as _P  # Phase 2 canonical perception layer

MAX_DEPTH = 9
ROUNDS = 6
LAST_FAIL = [""]
D8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTH = D8[:4]


def _bg(g: Grid) -> int:
    return _P.background(g, 'most_common_first')  # Phase 2: canonical perception (was 1 lines)


def _los(s: Grid, bg: int):
    """For each direction: grid of the colour of the first non-bg cell met going that way (-1 if none)."""
    h, w = len(s), len(s[0]); out = []
    for dr, dc in D8:
        L = [[-1] * w for _ in range(h)]
        rows = range(h) if dr <= 0 else range(h - 1, -1, -1)
        cols = range(w) if dc <= 0 else range(w - 1, -1, -1)
        for i in rows:
            for j in cols:
                a, b = i + dr, j + dc
                if 0 <= a < h and 0 <= b < w:
                    L[i][j] = s[a][b] if s[a][b] != bg else L[a][b]
        out.append(L)
    return out


def _outside(s: Grid, bg: int):
    h, w = len(s), len(s[0]); seen = [[0] * w for _ in range(h)]
    stack = [(i, j) for i in range(h) for j in range(w) if (i in (0, h - 1) or j in (0, w - 1)) and s[i][j] == bg]
    for i, j in stack:
        seen[i][j] = 1
    while stack:
        a, b = stack.pop()
        for da, db in ORTH:
            x, y = a + da, b + db
            if 0 <= x < h and 0 <= y < w and not seen[x][y] and s[x][y] == bg:
                seen[x][y] = 1; stack.append((x, y))
    return seen


def _feats(s: Grid, x0: Grid, bg: int):
    h, w = len(s), len(s[0]); los = _los(s, bg); out_ = _outside(s, bg); feats = []
    for i in range(h):
        for j in range(w):
            own = s[i][j]
            nb = [s[i + a][j + b] if 0 <= i + a < h and 0 <= j + b < w else -1 for a, b in D8]
            born = [int(s[i + a][j + b] != x0[i + a][j + b]) if 0 <= i + a < h and 0 <= j + b < w else -1 for a, b in ORTH]
            lc = [los[d][i][j] for d in range(8)]
            feats.append(tuple(
                [int(own == bg), own, int(s[i][j] != x0[i][j])] + born
                + [(-1 if v == -1 else int(v == bg)) for v in nb]
                + [(-1 if v == -1 else int(v == own)) for v in nb]
                + [own if False else v for v in nb[:4]]
                + lc + [int(v == own) if v != -1 else -1 for v in lc]
                + [out_[i][j]]))
    return feats, los


def _admissible(s: Grid, y: Grid, los, mode: str) -> List[frozenset]:
    h, w = len(s), len(s[0]); out = []
    for i in range(h):
        for j in range(w):
            own, tgt = s[i][j], y[i][j]
            if own == tgt:
                out.append(frozenset([("K",)])); continue
            labs = {("L", tgt)}
            if mode == "full":
                for k, (a, b) in enumerate(D8):
                    if 0 <= i + a < h and 0 <= j + b < w and s[i + a][j + b] == tgt:
                        labs.add(("N", k))
                for d in range(8):
                    if los[d][i][j] == tgt:
                        labs.add(("S", d))
            out.append(frozenset(labs))
    return out


def _step(s: Grid, x0: Grid, tree, bg: int) -> Grid:
    h, w = len(s), len(s[0]); feats, los = _feats(s, x0, bg); out = []
    for i in range(h):
        row = []
        for j in range(w):
            lab = _walk(tree, feats[i * w + j])
            if lab[0] == "K":
                row.append(s[i][j])
            elif lab[0] == "L":
                row.append(lab[1])
            elif lab[0] == "N":
                a, b = D8[lab[1]]; x, y = i + a, j + b
                row.append(s[x][y] if 0 <= x < h and 0 <= y < w else s[i][j])
            else:
                v = los[lab[1]][i][j]
                row.append(v if v != -1 else s[i][j])
        out.append(tuple(row))
    return tuple(out)


def simulate(x0: Grid, tree, bg: int, max_steps: Optional[int] = None):
    s = x0; T = max_steps or (len(s) + len(s[0]) + 4); visited = []
    for _ in range(T):
        n = _step(s, x0, tree, bg)
        if n == s:
            break
        s = n; visited.append(s)
    return s, visited


def _train(pairs: List[Pair], mode: str, deadline: float):
    rank = {("K",): (0, 0)}
    for k, (a, b) in enumerate(D8):
        rank[("N", k)] = (1, abs(a) + abs(b)); rank[("S", k)] = (2, k)
    occ: Dict[Tuple[int, ...], Counter] = defaultdict(Counter); tot: Counter = Counter()
    bgs = [_bg(x) for x, _ in pairs]

    def add(s, x0, y, bg):
        feats, los = _feats(s, x0, bg)
        for f, adm in zip(feats, _admissible(s, y, los, mode)):
            tot[f] += 1
            for lab in adm:
                occ[f][lab] += 1

    def rows():
        out = []
        for f, cnt in occ.items():
            lab = max(cnt, key=lambda l: (cnt[l] / tot[f], -rank.get(l, (3, 0))[0], -rank.get(l, (3, 0))[1], l))
            out.append((f, lab, 1))
        return out

    for (x, y), bg in zip(pairs, bgs):
        add(x, x, y, bg); add(y, x, y, bg)
    for _ in range(ROUNDS):
        if time.monotonic() > deadline:
            LAST_FAIL[0] = "out of time"; return None
        tree = _build(rows(), MAX_DEPTH, deadline)
        if tree is None:
            LAST_FAIL[0] = "tree could not separate the cases within depth %d" % MAX_DEPTH; return None
        clean = True
        for (x, y), bg in zip(pairs, bgs):
            fin, vis = simulate(x, tree, bg)
            if fin != y:
                clean = False
                for st in vis[:10]:
                    add(st, x, y, bg)
        if clean:
            return tree
    LAST_FAIL[0] = "did not converge within %d rounds" % ROUNDS
    return None


def induce(train_pairs: List[Pair], seconds: float = 8.0):
    if not all(len(x) == len(y) and len(x[0]) == len(y[0]) for x, y in train_pairs) or len(train_pairs) < 2:
        return None
    if all(x == y for x, y in train_pairs):
        return None
    deadline = time.monotonic() + seconds; n = len(train_pairs)
    for mode in ("full", "kl"):
        if time.monotonic() > deadline:
            return None
        tree = _train(train_pairs, mode, deadline)
        if tree is None:
            continue
        ok = 0
        for i in range(n):
            t = _train(train_pairs[:i] + train_pairs[i + 1:], mode, deadline)
            if t is not None:
                x, y = train_pairs[i]
                if simulate(x, t, _bg(x))[0] == y:
                    ok += 1
        if ok >= n - 1:
            return (mode, tree)
    return None


def apply_spec(g: Grid, spec) -> Optional[Grid]:
    out, _ = simulate(g, spec[1], _bg(g))
    return out if out != g else None


# ---------------------------------------------------------------- integration stage
ENABLED = os.environ.get("LUCA_STATEFUL", "1") != "0"


def upgrade_many(train_pairs: List[Pair], test_inputs: List[Grid], results: list) -> list:
    """Only touches pure fallbacks. The emergent prediction becomes attempt 1; the earlier first guess is
    kept as attempt 2 so nothing supplied by an earlier stage is displaced."""
    if not ENABLED or not any(r.failure_category is not None for r in results):
        return results
    try:
        spec = induce(train_pairs)
    except Exception:
        return results
    if spec is None:
        return results
    upgraded = []
    for g, r in zip(test_inputs, results):
        if r.failure_category is None:
            upgraded.append(r); continue
        try:
            pred = apply_spec(g, spec)
        except Exception:
            pred = None
        if pred is None:
            upgraded.append(r); continue
        law = Law(name="stateful_swarm", predict=lambda x, sp=spec: apply_spec(x, sp) or x,
                  description=f"STATEFUL-SWARM(labels={spec[0]}) local rule over signals, iterated to a fixed point; "
                              f"verified on all demonstrations and leave-one-out", constant_bits=12, op_count=2)
        preds, laws = [pred], [law]
        if r.predictions and r.predictions[0] != pred:
            preds.append(r.predictions[0])
            if r.laws:
                laws.append(r.laws[0])
        upgraded.append(replace(r, predictions=preds[:2], laws=laws[:2], failure_category=None,
                                explanation=r.explanation + " | stateful-swarm stage supplied attempt 1."))
    return upgraded
