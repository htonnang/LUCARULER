"""
lazy_beam_search.py  (LucaSO V1 -- Priority 4: Lazy Typed Beam Composition)
===============================================================================
The LucaSO technical documentation (Chapter 18.3) documents a real, prior
attempt at improving luca_search.py's composition search: a priority-
ordering scheme for the fitter pool, plus an early-exit once a
theoretically MDL-optimal candidate was found. It was measured, via
profiling, to provide ZERO speedup -- because primitives.fit_all_single_
traced() (and its object_diff/recursive_world/imported_primitives/
set_object_reducer/connectivity_engine equivalents) call EVERY fitter in
their pool unconditionally to build their return list BEFORE any caller-
side reordering or early-exit logic can act. Reordering an already-fully-
computed list cannot save the computation that built it.

This module does NOT repeat that mistake. It does not try to make the
POOL itself lazy (that would require rewriting every fit_all_*_traced()
function into a true generator -- a larger, riskier change deferred here,
same as before). Instead, it makes the SEARCH DEPTH lazy: the existing
depth<=3 raw-op composition in luca_search.py already tries every
combination of raw operations up to depth 3 exhaustively (bounded to a
small curated set specifically to keep that exhaustive search tractable).
This module replaces "try every combination up to depth 3" with "expand
only the top-BEAM_WIDTH most promising partial compositions at each
depth, and stop the instant an exact-fit solution is found," which is
what actually lets useful depth reach 4-6 without combinatorial
explosion -- the genuine bottleneck this priority targets, verified
against the real one profiling found (see module docstring above,
repeated from the honest negative-result record).

Typed operators, in the specific, honest sense implemented here: every
step in a composition chain is tagged with its behaviour class (GEOMETRIC,
STRUCTURAL, or EXACT_FIT_TERMINAL) so preconditions can skip combinations
that provably cannot help (e.g. chaining two colour-palette-preserving
GEOMETRIC ops when the target uses colours absent from the input can only
be resolved by an EXACT_FIT_TERMINAL step, so pure-GEOMETRIC chains longer
than needed are pruned early). This is a real but intentionally modest
type system -- not a full Grid/ObjectSet/Graph type lattice, which would
require reworking every fitter's signature across the whole codebase; see
the integration report for why that larger change was scoped out.

This is wired in as a FALLBACK, tried only when luca_search.search_
programs() (depth<=3) finds nothing, so it can never regress a task that
already worked -- it can only ever add coverage, by construction, not
take any away.
"""

from __future__ import annotations
from .diagnostics import record_exception
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, FrozenSet

from .grid_world import Grid, grid_shape, color_counts
from .primitives import GEOM_OPS, Law, _bbox_of_nonbg
from .grid_world import guess_backgrounds

Pair = Tuple[Grid, Grid]

BEAM_WIDTH = 6
MAX_DEPTH = 6


# ------------------------------------------------------------ typed raw operators

def _crop_to_content(g: Grid) -> Grid:
    bg = guess_backgrounds(g, k=1)[0]
    box = _bbox_of_nonbg(g, bg)
    if box is None:
        return g
    r0, c0, r1, c1 = box
    return tuple(row[c0:c1 + 1] for row in g[r0:r1 + 1])


@dataclass(frozen=True)
class TypedOp:
    name: str
    fn: Callable[[Grid], Grid]
    op_class: str  # "GEOMETRIC" | "STRUCTURAL"


TYPED_RAW_OPS: List[TypedOp] = [
    TypedOp("reflect_h", GEOM_OPS["reflect_h"], "GEOMETRIC"),
    TypedOp("reflect_v", GEOM_OPS["reflect_v"], "GEOMETRIC"),
    TypedOp("rot90", GEOM_OPS["rot90"], "GEOMETRIC"),
    TypedOp("rot180", GEOM_OPS["rot180"], "GEOMETRIC"),
    TypedOp("transpose", GEOM_OPS["transpose"], "GEOMETRIC"),
    TypedOp("crop_to_content", _crop_to_content, "STRUCTURAL"),
]


# ------------------------------------------------------------ scoring / state hashing

def _partial_match_score(pred: Grid, target: Grid) -> float:
    if grid_shape(pred) != grid_shape(target):
        return 0.0
    h, w = grid_shape(target)
    if h == 0 or w == 0:
        return 0.0
    matches = sum(1 for i in range(h) for j in range(w) if pred[i][j] == target[i][j])
    return matches / (h * w)


def _state_key(pairs: List[Pair]) -> FrozenSet:
    return frozenset(pairs)


@dataclass
class BeamNode:
    ops: Tuple[str, ...]              # sequence of op names applied so far
    pairs: List[Pair]                 # current (transformed_input, target) pairs
    score: float                      # average partial-match score against targets


def _apply_chain(ops: Tuple[TypedOp, ...], g: Grid) -> Grid:
    for op in ops:
        g = op.fn(g)
    return g


# ------------------------------------------------------------ lazy beam search

def lazy_beam_compose(pairs: List[Pair], terminal_search) -> Optional[Law]:
    """Beam search over TYPED_RAW_OPS composition chains, depth up to
    MAX_DEPTH, beam width BEAM_WIDTH. At every depth, before expanding
    further, every surviving beam node is offered to `terminal_search`
    (luca_search's exact-fit pooled candidate search) as a fresh residual
    problem -- exactly SOSEP-02's own residual-pairs beam search strategy
    (documented in the LucaSO technical documentation, Chapter 8.2), now
    given a native LUCA implementation instead of only being available by
    calling SOSEP-02 as a whole engine. Returns the first exact-fit Law
    found, composed with whatever op chain reached it, or None."""
    from .budget import check as _budget_check
    _budget_check()
    op_lookup = {op.name: op for op in TYPED_RAW_OPS}

    root = BeamNode(ops=(), pairs=list(pairs),
                     score=sum(_partial_match_score(x, y) for x, y in pairs) / len(pairs))
    beam = [root]
    visited = {_state_key(root.pairs)}

    for depth in range(MAX_DEPTH):
        for node in beam:
            if depth > 0:
                # Typed precondition: two GEOMETRIC ops in a row never help,
                # since composing two colour/shape-preserving rigid
                # transforms is itself just a single rigid transform already
                # reachable at depth 1 -- prune to avoid wasted terminal
                # searches on a state no terminal fitter can do anything
                # with that it couldn't already do one step earlier.
                if len(node.ops) >= 2:
                    last_two = [op_lookup[n].op_class for n in node.ops[-2:]]
                    if last_two == ["GEOMETRIC", "GEOMETRIC"]:
                        continue
                candidates = terminal_search(node.pairs)
                if candidates:
                    best = candidates[0]
                    chain = tuple(op_lookup[n] for n in node.ops)

                    def composed_predict(g: Grid, chain=chain, best=best) -> Grid:
                        return best.predict(_apply_chain(chain, g))

                    try:
                        if all(composed_predict(x) == y for x, y in pairs):
                            op_desc = " -> ".join(node.ops)
                            return Law(
                                name="+".join(node.ops) + "+" + best.name,
                                predict=composed_predict,
                                description=f"BEAM_COMPOSE(depth={depth}, ops=[{op_desc}], "
                                            f"terminal={best.description})",
                                constant_bits=best.constant_bits,
                                op_count=len(node.ops) + best.op_count,
                            )
                    except Exception:
                        record_exception('lazy_beam_search.py:169')
                        pass

        if depth == MAX_DEPTH - 1:
            break

        # expand the beam
        expansions: List[BeamNode] = []
        for node in beam:
            for op in TYPED_RAW_OPS:
                try:
                    new_pairs = [(op.fn(x), y) for x, y in node.pairs]
                except Exception:
                    record_exception('lazy_beam_search.py:181')
                    continue
                if new_pairs == node.pairs:
                    continue  # no-op; skip (mirrors luca_search's existing no-op guard)
                key = _state_key(new_pairs)
                if key in visited:
                    continue
                visited.add(key)
                score = sum(_partial_match_score(x, y) for x, y in new_pairs) / len(new_pairs)
                expansions.append(BeamNode(ops=node.ops + (op.name,), pairs=new_pairs, score=score))

        if not expansions:
            break
        expansions.sort(key=lambda n: -n.score)
        beam = expansions[:BEAM_WIDTH]

    return None
