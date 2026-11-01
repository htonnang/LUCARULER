"""LUCA DSL - engine operations: learners whose parameters are fitted by their own induction.

Each engine is an ordinary DSL operation with one fitted parameter `spec`. The fitter is the
learner's own induction procedure, which already checks itself by leaving one demonstration out,
so the search does not refit it again (search.SELF_VALIDATED). Engines run in the search's
second tier: only when no plain DSL program fits.

A fitted spec may hold structures that are not hashable (decision trees, closures); it is kept in
a per-process store and the program refers to it by key ('<engine>#<n>').
"""
import time

DEADLINE = [None]          # set by search.solve: the search's own deadline
_STORE = {}
_N = [0]


def _seconds(cap):
    if DEADLINE[0] is None:
        return cap
    return max(0.3, min(cap, DEADLINE[0] - time.time() - 0.2))


def _key(name, spec):
    _N[0] += 1
    k = f"{name}#{_N[0]}"
    _STORE[k] = spec
    return k


def _tuples(xs, ys):
    return [(tuple(map(tuple, x)), tuple(map(tuple, y))) for x, y in zip(xs, ys)]


# ------------------------------------------------------------------ fitters: pairs -> list of specs
def _fit_cellular(pairs):
    from .. import swarm_ca as m
    s = m.induce(pairs, seconds=_seconds(6.0))
    return [s] if s is not None else []


def _fit_signal(pairs):
    from .. import stateful_swarm as m
    s = m.induce(pairs, seconds=_seconds(8.0))
    return [s] if s is not None else []


def _fit_signal2(pairs):
    from .. import stateful_swarm2 as m
    s = m.induce(pairs, seconds=_seconds(8.0))
    return [s] if s is not None else []


def _fit_cell_rules(pairs):
    from .. import cell_rules as m
    return [c["predict"] for c in m.build_candidates(pairs, seconds=_seconds(20.0))[:2]]


def _fit_path(pairs):
    from .. import path_draw as m
    s = m.induce(pairs)
    if s is None:
        s = m.induce_compose(pairs)
    return [s] if s is not None else []


def _fit_hole(pairs):
    from .. import hole_decorate as m
    s = m.induce(pairs)
    return [s] if s is not None else []


def _fit_pattern(pairs):
    from .. import pattern_transfer as m
    return list(m.induce(pairs) or [])[:2]


def _fit_crop(pairs):
    from .. import crop_transform as m
    return list(m.induce_all(pairs, limit=4) or [])[:2]


# ------------------------------------------------------------------ appliers: (grid, spec) -> grid
def _apply_module(modname, fn="apply_spec"):
    def run(g, spec):
        import importlib
        m = importlib.import_module(".." + modname, __package__)
        return getattr(m, fn)(g, spec)
    return run


def _apply_cell_rules(g, predict):
    return predict(g)


def _apply_pattern(g, model):
    from .. import pattern_transfer as m
    return m.predict(g, model)


# ------------------------------------------------------------------ law engines: program synthesisers that
# return fitted laws (objects with .predict(grid)); they run under the legacy cooperative budget
def _law_fit(modname, fname, cap, top=4):
    def fitter(pairs):
        import importlib
        from .. import budget
        m = importlib.import_module(".." + modname, __package__)
        token = budget.begin(_seconds(cap))
        try:
            laws = list(getattr(m, fname)(pairs) or [])
        except budget.Exhausted:
            laws = list(budget.candidates())
        except Exception:
            laws = []
        finally:
            budget.end(token)
        out, seen = [], set()
        for law in laws:
            try:
                ok = all(law.predict(x) == y for x, y in pairs)
            except Exception:
                ok = False
            key = (law.name, getattr(law, "description", ""))
            if ok and key not in seen:
                seen.add(key); out.append(law)
            if len(out) >= top:
                break
        return out
    return fitter


def _apply_law(g, law):
    return law.predict(g)


ENGINES = {
    # name: (fitter, applier, description)
    # program synthesisers (formerly the base program search)
    "primitive_programs": (_law_fit("luca_search", "search_programs", 6.0), _apply_law,
                           "primitive transformations and their compositions"),
    "relational_rules": (_law_fit("generalized_rules", "fitted", 3.0), _apply_law,
                         "recolour / select / keep by relational object properties"),
    "drawing_rules": (_law_fit("conditional_drawing", "fit", 3.0), _apply_law, "conditional drawing programs"),
    "object_programs": (_law_fit("object_programs", "fit", 3.0), _apply_law, "per-object edit programs"),
    "object_edits": (_law_fit("object_diff", "fit_all_object_diff", 3.0), _apply_law,
                     "object edits matched between input and output"),
    "panel_tables": (_law_fit("representation_search", "fit", 3.0), _apply_law,
                     "panel tables, frames, region paint, template repeat"),
    "reference_programs": (_law_fit("reference_programs", "fit", 3.0), _apply_law, "named-role branching programs"),
    "composition": (_law_fit("composition", "fit", 4.0), _apply_law, "compositions of unary / binary scene operators"),
    "representation_programs": (_law_fit("representation_programs", "fit", 3.0), _apply_law,
                                "object-set to grid representation programs"),
    "local_frames": (_law_fit("local_frames", "fit", 2.0), _apply_law, "local frame programs"),
    # learners (LEARNERS below)
    "cellular": (_fit_cellular, _apply_module("swarm_ca"), "iterated local rule (cellular automaton)"),
    "cell_rules": (_fit_cell_rules, _apply_cell_rules, "learned per-cell decision rule"),
    "path_draw": (_fit_path, _apply_module("path_draw"), "rays / paths cast from sources"),
    "signal_ca": (_fit_signal, _apply_module("stateful_swarm"), "cellular automaton with signals"),
    "signal_ca2": (_fit_signal2, _apply_module("stateful_swarm2"), "cellular automaton with signals and object info"),
    "pattern_transfer": (_fit_pattern, _apply_pattern, "pattern transfer / marker extension"),
    "hole_decorate": (_fit_hole, _apply_module("hole_decorate"), "outline objects, fill their holes"),
    "crop_transform": (_fit_crop, _apply_module("crop_transform"), "crop a selected region, then rotate / flip"),
}


def fit(name, xs, ys):
    """Keys of the specs the engine fits on these demonstrations (best first)."""
    try:
        specs = ENGINES[name][0](_tuples(xs, ys))
    except Exception:
        return []
    return [_key(name, s) for s in specs]


def apply(name, g, key):
    spec = _STORE.get(key)
    if spec is None:
        return None
    try:
        out = ENGINES[name][1](tuple(map(tuple, g)), spec)
    except Exception:
        return None
    return None if out is None else tuple(map(tuple, out))


SYNTHESISERS = ("primitive_programs", "relational_rules", "drawing_rules", "object_programs", "object_edits",
                "panel_tables", "reference_programs", "composition", "representation_programs", "local_frames")
