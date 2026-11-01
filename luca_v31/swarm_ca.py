"""swarm_ca.py  (LUCA V31 prototype: emergent behaviour from an iterated local rule)
=================================================================================
Swarm principles applied to grids:
  * Decentralised   - no global controller: every cell is an agent applying the SAME rule.
  * Local           - an agent sees only its (2r+1)x(2r+1) neighbourhood.
  * Simple          - the rule is a small decision tree over the neighbourhood.
  * Emergent        - the rule is applied to all cells simultaneously, repeatedly, until
                      nothing changes, so long-range structure (rays, fills, growth)
                      arises from local interaction.

Learning. We see only the start x and the end y of each demonstration, never the
intermediate states. So the rule is learned by iterative self-training (DAgger style):
  1. label every cell of x and of y with its target colour in y
     (y itself must be a fixed point);
  2. fit a tree, simulate it from x, and add every intermediate state it visits
     (labelled with the final target y) whenever it fails to converge to y;
  3. repeat until it reproduces y exactly from x, or the data become contradictory.
This assumes each cell converges monotonically to its final colour.

A rule is accepted only if (a) it reproduces every demonstration exactly and (b) it
survives leave-one-out (fit without a pair, then simulate that pair) on at least n-1
pairs. Standard library only.
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


def _bg(g: Grid) -> int:
    return _P.background(g, 'most_common_first')  # Phase 2: canonical perception (was 1 lines)


def _offsets(r: int):
    return [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1) if (a, b) != (0, 0)]


def _feats(s: Grid, bg: int, r: int) -> List[Tuple[int, ...]]:
    h, w = len(s), len(s[0]); offs = _offsets(r); out = []
    for i in range(h):
        for j in range(w):
            own = s[i][j]
            nb = [s[i + a][j + b] if 0 <= i + a < h and 0 <= j + b < w else -1 for a, b in offs]
            out.append(tuple([int(own == bg)] + [(-1 if v == -1 else int(v == bg)) for v in nb]
                             + [(-1 if v == -1 else int(v == own)) for v in nb] + [own] + nb))
    return out


def _admissible(s: Grid, y: Grid, r: int, mode: str) -> List[frozenset]:
    """For every cell the set of labels that would produce its target colour: KEEP if already right,
    otherwise COPY-NEIGHBOUR-k for every neighbour holding the target colour (and a literal colour)."""
    h, w = len(s), len(s[0]); offs = _offsets(r); out = []
    for i in range(h):
        for j in range(w):
            own, tgt = s[i][j], y[i][j]
            if own == tgt:
                out.append(frozenset([("K",)]))
                continue
            labs = {("L", tgt)}
            if mode == "kn":
                for k, (a, b) in enumerate(offs):
                    if 0 <= i + a < h and 0 <= j + b < w and s[i + a][j + b] == tgt:
                        labs.add(("N", k))
            out.append(frozenset(labs))
    return out


def _step(s: Grid, tree, bg: int, r: int) -> Grid:
    h, w = len(s), len(s[0]); offs = _offsets(r); feats = _feats(s, bg, r); out = []
    for i in range(h):
        row = []
        for j in range(w):
            lab = _walk(tree, feats[i * w + j])
            if lab[0] == "K":
                row.append(s[i][j])
            elif lab[0] == "L":
                row.append(lab[1])
            else:
                a, b = offs[lab[1]]
                x, y = i + a, j + b
                row.append(s[x][y] if 0 <= x < h and 0 <= y < w else s[i][j])
        out.append(tuple(row))
    return tuple(out)


def simulate(s: Grid, tree, bg: int, r: int, max_steps: Optional[int] = None):
    """Run the local rule on every cell at once until nothing changes. Returns (final, visited states)."""
    T = max_steps or (len(s) + len(s[0]) + 4)
    visited = []
    for _ in range(T):
        n = _step(s, tree, bg, r)
        if n == s:
            break
        s = n; visited.append(s)
    return s, visited


def _train(pairs: List[Pair], r: int, mode: str, deadline: float):
    """Conservative learning: when identical local views disagree, take the label that is valid for the
    largest share of their occurrences (KEEP wins ties). Whether the resulting rule really converges to the
    demonstrated outputs is decided by simulation, not assumed."""
    offs = _offsets(r)
    rank = {("K",): (0, 0)}
    for k, (a, b) in enumerate(offs):
        rank[("N", k)] = (1, abs(a) + abs(b) + (0 if (a == 0 or b == 0) else 0.5))
    occ: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
    tot: Counter = Counter()
    bgs = [_bg(x) for x, _ in pairs]

    def add(s, y, bg):
        for f, adm in zip(_feats(s, bg, r), _admissible(s, y, r, mode)):
            tot[f] += 1
            for lab in adm:
                occ[f][lab] += 1

    def rows():
        out = []
        for f, cnt in occ.items():
            lab = max(cnt, key=lambda l: (cnt[l] / tot[f], -rank.get(l, (2, 0))[0], -rank.get(l, (2, 0))[1], l))
            out.append((f, lab, 1))
        return out

    for (x, y), bg in zip(pairs, bgs):
        add(x, y, bg); add(y, y, bg)
    for _ in range(ROUNDS):
        if time.monotonic() > deadline:
            LAST_FAIL[0] = "out of time"
            return None
        tree = _build(rows(), MAX_DEPTH, deadline)
        if tree is None:
            LAST_FAIL[0] = "tree could not separate the cases within depth %d" % MAX_DEPTH
            return None
        clean = True
        for (x, y), bg in zip(pairs, bgs):
            fin, vis = simulate(x, tree, bg, r)
            if fin != y:
                clean = False
                for st in vis[:12]:
                    add(st, y, bg)
        if clean:
            return tree
    LAST_FAIL[0] = "did not converge to the demonstrated outputs within %d rounds" % ROUNDS
    return None


def induce(train_pairs: List[Pair], seconds: float = 6.0):
    """(radius, mode, tree) whose iterated application reproduces every demonstration and survives
    leave-one-out on at least n-1 pairs; else None."""
    if not all(len(x) == len(y) and len(x[0]) == len(y[0]) for x, y in train_pairs) or len(train_pairs) < 2:
        return None
    if all(x == y for x, y in train_pairs):
        return None
    deadline = time.monotonic() + seconds
    n = len(train_pairs)
    for r in (1, 2):
        for mode in ("kn", "kl"):
            if time.monotonic() > deadline:
                return None
            tree = _train(train_pairs, r, mode, deadline)
            if tree is None:
                continue
            ok = 0
            for i in range(n):
                sub = train_pairs[:i] + train_pairs[i + 1:]
                t = _train(sub, r, mode, deadline)
                if t is not None:
                    x, y = train_pairs[i]
                    if simulate(x, t, _bg(x), r)[0] == y:
                        ok += 1
            if ok >= n - 1:
                return (r, mode, tree)
    return None


def apply_spec(g: Grid, spec) -> Optional[Grid]:
    r, mode, tree = spec
    out, _ = simulate(g, tree, _bg(g), r)
    return out if out != g else None


# ---------------------------------------------------------------- integration stage
ENABLED = os.environ.get("LUCA_SWARM", "1") != "0"


def upgrade_many(train_pairs: List[Pair], test_inputs: List[Grid], results: list) -> list:
    """Only touches pure fallbacks. The emergent prediction becomes attempt 1; the earlier first guess
    is kept as attempt 2 so nothing an earlier stage supplied is displaced."""
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
            upgraded.append(r)
            continue
        try:
            pred = apply_spec(g, spec)
        except Exception:
            pred = None
        if pred is None:
            upgraded.append(r)
            continue
        law = Law(name="swarm_ca", predict=lambda x, sp=spec: apply_spec(x, sp) or x,
                  description=f"SWARM-CA(radius={spec[0]}, labels={spec[1]}) local rule iterated to a fixed point; "
                              f"verified on all demonstrations and leave-one-out",
                  constant_bits=10, op_count=2)
        preds, laws = [pred], [law]
        if r.predictions and r.predictions[0] != pred:
            preds.append(r.predictions[0])
            if r.laws:
                laws.append(r.laws[0])
        upgraded.append(replace(r, predictions=preds[:2], laws=laws[:2], failure_category=None,
                                explanation=r.explanation + " | swarm-CA stage supplied attempt 1."))
    return upgraded
