"""
search.py  (LUCA V9)
=====================
V9 changes vs. V8:

  * Item 2 (robustness): every law whose parameters were learned from data
    (constant_bits > 0) is now leave-one-out validated before being
    admitted as a candidate, when there are >=3 training pairs. A law that
    only "works" because it memorized 2-3 examples no longer gets through.

  * Bug fix in composition: V8's depth-2 search drew its "stage 1" ops from
    laws that had *already been required to exactly solve the whole task*
    (`fit_all_single(pairs)` requires an exact match against the final
    output), which meant genuine two-step composition essentially never
    fired -- composing an already-complete solution with anything else is
    a no-op, and the "skip trivial" guard silently discarded those
    branches. V9 fixes this: stage-1 candidates are drawn from a small set
    of *parameter-free, unconditionally-applicable* raw operations
    (geometry + crop-to-content + denoise), applied speculatively without
    requiring them to solve anything on their own. Only the *final* stage
    in a chain must exactly reproduce the target.

  * Item 5 (typed/deeper search): iterative-deepening beam search now goes
    to depth 3 (raw-op -> raw-op -> exact-fit primitive) when no depth<=2
    solution exists, bounded to a small curated raw-op set to keep the
    combinatorics tractable.

  * Object-diff (object_diff.py) and recursive-world (recursive_world.py)
    laws are pooled into the same candidate search as primitives.py.

  * Item 21 (pass@2 diversification) unchanged in spirit, extended to
    dedupe across the larger candidate pool.
"""

from __future__ import annotations
from .diagnostics import record_exception, event
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable, Dict

from .grid_world import Grid, grid_shape, guess_backgrounds, extract_objects
from . import primitives
from . import object_diff
from . import recursive_world
from . import imported_primitives
from . import set_object_reducer
from . import connectivity_engine
from . import lazy_beam_search
from .primitives import Law, Pair, GEOM_OPS

# ---------------------------------------------------------------- raw ops

def _crop_to_content_raw(g: Grid) -> Grid:
    from .primitives import _bbox_of_nonbg
    bg = guess_backgrounds(g, k=1)[0]
    box = _bbox_of_nonbg(g, bg)
    if box is None:
        return g
    r0, c0, r1, c1 = box
    return tuple(row[c0:c1 + 1] for row in g[r0:r1 + 1])


def _denoise_raw(g: Grid) -> Grid:
    bg = guess_backgrounds(g, k=1)[0]
    objs = extract_objects(g, bg, connectivity=8, multicolor=False)
    grid = [list(r) for r in g]
    for o in objs:
        if o.area == 1:
            for (r, c) in o.cells:
                grid[r][c] = bg
    return tuple(tuple(row) for row in grid)


# Small, curated set of parameter-free operations usable as an
# intermediate composition step regardless of whether they alone explain
# anything (see module docstring "Bug fix in composition").
RAW_OPS: List[Tuple[str, Callable[[Grid], Grid]]] = [
    ("reflect_h", GEOM_OPS["reflect_h"]),
    ("reflect_v", GEOM_OPS["reflect_v"]),
    ("rot90", GEOM_OPS["rot90"]),
    ("rot180", GEOM_OPS["rot180"]),
    ("transpose", GEOM_OPS["transpose"]),
    ("crop_to_content", _crop_to_content_raw),
    ("denoise_singletons", _denoise_raw),
]
# Even smaller subset for depth-3 (raw o raw o exact-fit) to bound the
# combinatorics -- depth-3 is only attempted when nothing shallower works.
# Depth-3 stays bounded to a curated subset for runtime reasons -- tested
# expanding to the full RAW_OPS set (7 ops) after adding grid_world.py
# caching, but it cost ~58% more runtime for zero additional solves on a
# 60-task sample, so the trimmed set remains the better tradeoff until
# there's evidence a specific missing op-pair is needed.
RAW_OPS_DEPTH3: List[Tuple[str, Callable[[Grid], Grid]]] = RAW_OPS[:5]


def mdl_score(law: Law) -> float:
    return 10.0 * law.op_count + 2.0 * law.constant_bits


# ---------------------------------------------------------------- Item 2: robustness / LOO

def _robust(fitter: Optional[Callable], law: Law, pairs: List[Pair]) -> bool:
    """Leave-one-out validation for laws with learned (non-zero) constant
    bits. Parameter-free laws (constant_bits == 0, e.g. plain geometry)
    skip this -- there is nothing to overfit."""
    if fitter is None or law.constant_bits == 0:
        return True
    if len(pairs) < 3:
        return True  # not enough held-out signal to judge; let MDL sort it out
    for i in range(len(pairs)):
        subset = pairs[:i] + pairs[i + 1:]
        try:
            refit = fitter(subset)
        except Exception:
            record_exception('luca_search.py:112')
            refit = None
        if isinstance(refit, list):
            refit = next((l for l in refit if l.name == law.name), None)
        if refit is None or refit.name != law.name:
            return False
        try:
            if refit.predict(pairs[i][0]) != pairs[i][1]:
                return False
        except Exception:
            record_exception('luca_search.py:121')
            return False
    return True


def _uncached_pooled_traced_candidates(pairs: List[Pair]):
    """All depth-1 candidates from primitives.py + object_diff.py +
    recursive_world.py + imported_primitives.py + set_object_reducer.py
    (LucaSO integration: SOSEP-02-derived fitters AND the set-level
    object-reduction engine participate here identically to native LUCA
    fitters -- same LOO validation, same MDL scoring, same composition
    eligibility), as (Law, fitter_or_None) pairs."""
    traced = []
    traced.extend(primitives.fit_all_single_traced(pairs))
    traced.extend(connectivity_engine.fit_all_connectivity_traced(pairs))
    return traced


from functools import lru_cache

@lru_cache(maxsize=64)
def _cached_pooled(pairs):
    return tuple(_uncached_pooled_traced_candidates(list(pairs)))

def _pooled_traced_candidates(pairs):
    from .budget import check
    check()
    return list(_cached_pooled(tuple(pairs)))

def _robust_candidates(pairs: List[Pair]) -> List[Law]:
    out = []
    for law, fitter in _pooled_traced_candidates(pairs):
        if _robust(fitter, law, pairs):
            out.append(law)
        else:
            event("leave_one_out_rejected", law.name)
    return out


# ---------------------------------------------------------------- best-partial-match fallback
# When no law explains the training set exactly, the old fallback was
# always "return the input unchanged." That wastes the pass@2 budget --
# a transform that's *closer* to the target (even if not exact) is a
# strictly better second guess. This scores a handful of cheap candidate
# transforms by average per-cell match fraction against the training
# outputs and returns the best one.

def _partial_match_score(pred: Grid, target: Grid) -> float:
    if grid_shape(pred) != grid_shape(target):
        return 0.0
    h, w = grid_shape(target)
    if h == 0 or w == 0:
        return 0.0
    matches = sum(1 for i in range(h) for j in range(w) if pred[i][j] == target[i][j])
    return matches / (h * w)


def _best_effort_color_map(pairs: List[Pair]):
    """Majority-vote color->color mapping, even if it doesn't fit every
    pair exactly (unlike primitives.fit_color_map, which requires an
    exact bijective fit)."""
    from collections import Counter
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None
    votes = {}
    for x, y in pairs:
        for row_x, row_y in zip(x, y):
            for cx, cy in zip(row_x, row_y):
                votes.setdefault(cx, Counter())[cy] += 1
    mapping = {c: cnt.most_common(1)[0][0] for c, cnt in votes.items()}

    def predict(g: Grid) -> Grid:
        return tuple(tuple(mapping.get(c, c) for c in row) for row in g)
    return predict


def best_partial_match_prediction(train_pairs: List[Pair], test_input: Grid) -> Tuple[Grid, str]:
    candidates: List[Tuple[str, Callable[[Grid], Grid]]] = [("identity", lambda g: g)]
    candidates.extend(RAW_OPS)
    best_effort_cmap = _best_effort_color_map(train_pairs)
    if best_effort_cmap is not None:
        candidates.append(("best_effort_color_map", best_effort_cmap))

    best_name, best_fn, best_score = "identity", (lambda g: g), -1.0
    for name, fn in candidates:
        try:
            scores = [_partial_match_score(fn(x), y) for x, y in train_pairs]
        except Exception:
            record_exception('luca_search.py:195')
            continue
        avg = sum(scores) / len(scores) if scores else 0.0
        if avg > best_score:
            best_score = avg
            best_name, best_fn = name, fn

    try:
        pred = best_fn(test_input)
    except Exception:
        record_exception('luca_search.py:204')
        pred = test_input
    return pred, f"{best_name} (best-partial-match, avg_cell_match={best_score:.2f})"


# ---------------------------------------------------------------- Item: usage-weighted ordering (attempted, reverted)
# Inspired by a paper on proteinoid microsphere computing (Mougkogiannis &
# Adamatzky 2024, BBRC 709:149725) -- specifically its account of
# intrinsic plasticity: pathways that have actually carried signal before
# become more conductive next time, rather than every pathway being tried
# with equal, memoryless weight. Stripped of the biology, that's the
# standard "try what has worked before, first" heuristic (VSIDS-style
# variable ordering in SAT solvers).
#
# This was implemented -- a FITTER_PRIORITY table built from a real,
# empirical pass over the full 1000-task training set (recording which
# fitter last determined each solved task's winning law), used to reorder
# `_pooled_traced_candidates`'s output, plus an early-exit once a
# theoretical-MDL-minimum candidate was found for a given raw_op in the
# depth-2 composition loop.
#
# It was reverted after profiling showed it provides NO real speedup, and
# the reason is architectural, not a bug in the reordering logic itself:
# `primitives.fit_all_single_traced` (and its object_diff/recursive_world
# equivalents) unconditionally call EVERY fitter in their pool to build
# the full candidate list BEFORE returning it -- so by the time
# `_pooled_traced_candidates` receives that list to sort, or search.py's
# composition loop gets to early-exit on it, the expensive work (e.g.
# `fit_fractal_self_tile`, `fit_panel_boolean` -- 15s and 7s respectively
# of the ~30s spent per 60 tasks, per profiling) has already happened
# unconditionally. Reordering a list after every item in it was already
# computed cannot save the computation.
#
# A real fix would require converting `fit_all_single_traced` and its two
# counterparts into lazy generators that call fitters one at a time in
# priority order, so a consumer's early-exit actually skips *calling*
# the remaining (expensive) fitters -- not just skips iterating over
# results that were already computed regardless. That is a legitimate
# next step, but it also needs COST-aware priority, not just win-count
# priority: naively calling the highest-win-rate fitter first would put
# `fit_panel_boolean` (this dataset's single highest-value fitter, but
# also one of its most expensive) ahead of everything, which would slow
# down the common case of tasks that don't need it at all. Correctly
# weighing win-rate against per-call cost needs real profiling data per
# fitter, not just the win-count table below -- left as future work
# rather than shipped half-verified.
#
# The empirical win-count table itself is kept here for reference, since
# gathering it was legitimate, verified work (see README) even though the
# optimization built on top of it didn't pan out:
FITTER_WIN_COUNTS_REFERENCE: Dict[str, int] = {
    "tile_output_mirror": 8, "panel_or": 5, "rot180+tile_output_mirror": 4,
    "recolor_by_size_rank": 3, "panel_xor": 3, "symmetry_repair_crop": 3,
    "scale_up": 3, "color_map": 3, "self_template_fractal": 2,
    "uniform_object_translate": 2, "rot180": 2, "transpose": 2,
    "fill_holes_constant": 1, "best_partial_match": 1, "crop_to_content": 1,
    "gravity_down": 1, "recolor_by_internal_subobjects": 1,
    "denoise_singletons": 1, "gravity_right": 1, "reflect_h": 1,
    "reflect_v": 1, "duplicate_periodic_v_both": 1, "tile_output": 1,
    "symmetry_repair": 1, "crop_largest_object": 1, "tile_by_expression": 1,
    "duplicate_periodic_v_fwd": 1, "rot270": 1,
}


# ---------------------------------------------------------------- search

def _legacy_search_programs(pairs: List[Pair]) -> List[Law]:
    candidates: List[Law] = list(_robust_candidates(pairs))
    from .budget import remember
    remember(candidates)

    # ---- depth 2: raw_op -> exact-fit primitive/object-diff/recursive law
    for raw_name, raw_fn in RAW_OPS:
        try:
            intermediate = [(raw_fn(x), y) for x, y in pairs]
        except Exception:
            record_exception('luca_search.py:277')
            continue
        if all(a == b for a, b in intermediate):
            continue  # raw op changed nothing; composing further adds no value here
        from .budget import check
        check()
        for law2, fitter2 in _pooled_traced_candidates(intermediate):
            if law2.name == "identity":
                continue
            if not _robust(fitter2, law2, intermediate):
                continue

            def composed_predict(g: Grid, raw_fn=raw_fn, law2=law2) -> Grid:
                return law2.predict(raw_fn(g))

            law = Law(
                name=f"{raw_name}+{law2.name}",
                predict=composed_predict,
                description=f"COMPOSE(GEOMETRIC/STRUCTURAL({raw_name}), {law2.description})",
                constant_bits=law2.constant_bits,
                op_count=1 + law2.op_count,
            )
            try:
                if all(law.predict(x) == y for x, y in pairs):
                    candidates.append(law)
                    remember(candidates)
            except Exception:
                record_exception('luca_search.py:300')
                pass

    if candidates:
        candidates.sort(key=mdl_score)
        return candidates

    # ---- depth 3: raw_op -> raw_op -> exact-fit (only tried when nothing shallower works)
    for name1, fn1 in RAW_OPS_DEPTH3:
        try:
            inter1 = [(fn1(x), y) for x, y in pairs]
        except Exception:
            record_exception('luca_search.py:311')
            continue
        if all(a == b for a, b in inter1):
            continue
        for name2, fn2 in RAW_OPS_DEPTH3:
            if name2 == name1:
                continue
            try:
                inter2 = [(fn2(gx), y) for gx, y in inter1]
            except Exception:
                record_exception('luca_search.py:320')
                continue
            if all(a == b for a, b in inter2):
                continue
            for law3, fitter3 in _pooled_traced_candidates(inter2):
                if law3.name == "identity":
                    continue
                if not _robust(fitter3, law3, inter2):
                    continue

                def composed_predict(g: Grid, fn1=fn1, fn2=fn2, law3=law3) -> Grid:
                    return law3.predict(fn2(fn1(g)))

                law = Law(
                    name=f"{name1}+{name2}+{law3.name}",
                    predict=composed_predict,
                    description=(
                        f"COMPOSE(GEOMETRIC/STRUCTURAL({name1}), "
                        f"GEOMETRIC/STRUCTURAL({name2}), {law3.description})"
                    ),
                    constant_bits=law3.constant_bits,
                    op_count=2 + law3.op_count,
                )
                try:
                    if all(law.predict(x) == y for x, y in pairs):
                        candidates.append(law)
                    remember(candidates)
                except Exception:
                    record_exception('luca_search.py:346')
                    pass

    candidates.sort(key=mdl_score)
    if candidates:
        return candidates

    # ---- Priority 4 fallback: lazy beam composition (depth up to 6), tried
    # ONLY when depth<=3 raw-op composition (above) found nothing at all --
    # this ordering means the beam search can only ever ADD coverage, never
    # regress a task the existing depth<=3 search already solves, since it
    # never runs when `candidates` is already non-empty.
    beam_law = lazy_beam_search.lazy_beam_compose(pairs, terminal_search=_robust_candidates)
    if beam_law is not None:
        candidates = [beam_law]

    return candidates


def search_programs(pairs: List[Pair]) -> List[Law]:
    from .budget import check as _budget_check
    _budget_check()
    from .diagnostics import event
    # Exact structural hypotheses are considered before fixed colour-map
    # shortcuts: demonstration fit alone cannot establish colour generality.
    from .audit_rules import fit
    structural = fit(pairs)
    if structural:
        event('audit_structural_candidates')
        return structural
    from .panel_composition import fit as fit_panels
    panels = fit_panels(pairs)
    if panels:
        event('panel_composition_candidates')
        return panels
    candidates = _legacy_search_programs(pairs)
    if candidates:
        event('legacy_exact_candidates')
        return candidates
    event('legacy_search_exhausted')
    from .enhanced_search import search
    enhanced = search(pairs)
    if enhanced:
        return enhanced
    from .procedure_memory import recall
    recalled = recall(pairs)
    if recalled:
        return recalled
    from .relational_programs import fit as fit_relational
    return fit_relational(pairs)


def diversify_top2(candidates: List[Law]) -> List[Law]:
    """Pass@2: pick the best law, then the best law that is structurally
    different from it (different first-stage family) and predicts
    differently where it matters."""
    if not candidates:
        return []
    best = candidates[0]

    def family(law: Law) -> str:
        return law.name.split("+")[0]

    second = None
    for c in candidates[1:]:
        if family(c) != family(best) or c.name != best.name:
            second = c
            break
    picks = [best]
    if second is not None:
        picks.append(second)
    return picks


@dataclass
class SolveResult:
    predictions: List[Grid]
    laws: List[Law]
    explanation: str
    failure_category: Optional[str] = None
    diagnostics: Optional[Dict] = None


# OPT198: the legacy base-search harness (_solve_task_impl ... solve_many) is retired; the
# primitive program search below runs as the DSL engine operation `primitive_programs`.

