"""LUCA DSL v1 - interpreter and typed search.

Program  = tuple of steps; step = (op_name, ((param, value), ...))
run(program, grid) -> grid | None
solve(pairs, test_inputs) -> list of (prediction_per_test, program) ranked best first

Search: breadth-first over chains of up to MAX_DEPTH operations. At each step the parameter
candidates come from the inference rules applied to the CURRENT intermediate grids and the
demonstration outputs. A program is a solution if it reproduces every demonstration exactly.
Validation (when there are >= 3 demonstrations): for each demonstration, the same operation
skeleton is re-fitted on the other demonstrations and must predict the held-out one.
Ranking: total cost (operation costs + parameter complexity), then agreement between programs.
"""
import itertools, time
from collections import defaultdict
from .ops import REGISTRY, Context, ok, most_common, resolve_bg
from .infer import RULES
from .engines import ENGINES as _ENGINES
from . import engines as _engines
from . import lift as _lift

# OPT198: engines (learners with their own induction) form the search's second tier and validate
# themselves by leaving one demonstration out
ENGINE_OPS = set(_ENGINES)
TIER1_SECONDS = 8.0     # the plain DSL tier; engines use the rest of the task's time

MAX_DEPTH = 3
BEAM = [True]
BEAM_SECONDS = 4.0   # at least this long, after the engines
GEOMETRIC = {"rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"}
# drawing operations are final 'render' steps: tried only when the grids already have the output
# shape, never extended by further steps, and only within programs of at most 2 steps
# operations that read the original scene: first step only
FIRST_ONLY = {"select_objects", "connect_points", "move_objects"}
# fitted grammars: settings come from the grammar's own fitter on the raw demonstrations
FITTED = {"obj_map", "obj_connect", "obj_rays", "obj_move", "obj_recolour", "per_region", "elbow_path", "recolour_by_position", "fill_holes_by_property", "interior_by_property", "colour_order", "connect_points", "move_objects", "apply_key", "complete_copies", "panel_logic", "count_render", "stencil", "colour_histogram", "recolour_by_property", "object_rule", "to_boundary", "layers", "copy_to_markers"}
# count_render reads the raw scene

# key reading works on the raw scene: first step only, but (unlike selection) it may override
# cheap first steps after which a scene-reading rule may run (crop / view / palette changes)
PREFIX_OK = {"rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "crop_content", "crop_bbox_of",
             "recolor", "swap_colours"}
INPUT_ONLY = {"interior_by_property", "apply_key", "complete_copies", "count_render", "stencil", "colour_histogram", "recolour_by_property", "object_rule", "to_boundary", "layers", "copy_to_markers"}
# OPT193 scene concepts: a whole-scene transformation is the last step of a program (no op follows),
# used at most after one earlier step - this keeps the search frontier (and the deadline) unchanged
SCENE_FINAL = {"obj_map", "obj_connect", "obj_rays", "obj_move", "obj_recolour", "per_region", "fill_gaps", "edge_pair_lines", "line_uniform", "role_recolour", "fill_pairs", "marker_stripes", "shift_objects", "keep_largest", "flip_objects", "hollow", "extend_to_marker", "elbow_path", "translate", "panel_paint", "recolour_by_position", "fill_holes_by_property", "recolour_to_marker", "panel_reduce", "frame", "recolour_by_contact", "fill_enclosed", "symmetrize", "gravity", "crop_window", "colour_order", "rings", "fill_touching", "mirror_about", "border_lines",
               "fill_largest_rect", "project_markers"}
TERMINAL = {"obj_map", "obj_connect", "obj_rays", "obj_move", "obj_recolour", "per_region", "fill_gaps", "edge_pair_lines", "line_uniform", "role_recolour", "fill_pairs", "marker_stripes", "shift_objects", "keep_largest", "flip_objects", "hollow", "extend_to_marker", "elbow_path", "translate", "panel_paint", "recolour_by_position", "fill_holes_by_property", "recolour_to_marker", "frame", "recolour_by_contact", "fill_enclosed", "symmetrize", "gravity", "rings", "fill_touching", "mirror_about", "border_lines", "fill_largest_rect", "project_markers",
            "box_objects", "connect_points", "move_objects", "fill_bbox_by_colour", "marker_rays", "connect_to_block", "extend_lines", "corner_rays"}
# operations whose output always has the input's shape: useless as a final step when shapes differ
SHAPE_PRESERVING = {"obj_map", "obj_connect", "obj_rays", "obj_move", "obj_recolour", "per_region", "fill_gaps", "edge_pair_lines", "line_uniform", "role_recolour", "fill_pairs", "marker_stripes", "shift_objects", "keep_largest", "flip_objects", "hollow", "extend_to_marker", "elbow_path", "translate", "panel_paint", "recolour_by_position", "fill_holes_by_property", "recolour_to_marker", "frame", "recolour_by_contact", "fill_enclosed", "symmetrize", "gravity", "rings", "fill_touching", "mirror_about", "border_lines", "fill_largest_rect",
                    "project_markers", "box_objects", "rot180", "flip_h", "flip_v", "recolor", "swap_colours", "rays", "repair_period",
                    "fill_bbox_by_colour", "marker_rays", "connect_to_block", "extend_lines", "corner_rays"}


def _resolve(params, g):
    return {k: (resolve_bg(v, g) if k == "bg" or v == "auto" else v) for k, v in params}


def run(program, g):
    ctx = Context(g)
    for op, params in program:
        fn = REGISTRY[op][0]
        try:
            g = fn(g, ctx, **_resolve(params, g))
        except Exception:
            return None
        if not ok(g):
            return None
    return g


def _apply_step(step, gs, inputs):
    op, params = step
    fn = REGISTRY[op][0]
    out = []
    for g, inp in zip(gs, inputs):
        try:
            r = fn(g, Context(inp), **_resolve(params, g))
        except Exception:
            return None
        if not ok(r):
            return None
        out.append(r)
    return tuple(out)


def _param_options(op, gs, ys, inputs=None):
    spec = REGISTRY[op][1]
    lists = []
    for name, rule in spec:
        vals = RULES[rule](list(gs), list(ys), list(inputs) if inputs else None)
        if not vals:
            return []
        lists.append([(name, v) for v in vals])
    return [tuple(c) for c in itertools.product(*lists)] if lists else [()]


def _complexity(v):
    if isinstance(v, tuple) and v and isinstance(v[0], tuple) and len(v[0]) == 2 and isinstance(v[0][1], (tuple, int)):
        return len(v)  # mappings / tables cost per entry
    return 0


def cost(program):
    return sum(REGISTRY[op][2] + sum(_complexity(v) for _, v in params) for op, params in program)


def _allowed(prev, op):
    if prev is None:
        return True
    if prev in GEOMETRIC and op in GEOMETRIC:
        return False
    if prev == op and op not in ("rays",):
        return False
    if op == "crop_window" and prev in ("select_objects", "crop_content", "crop_bbox_of", "object_rule", "pool",
                                        "pool_cells", "top_left_quadrant"):
        return False   # a fixed window inside an already selected / cropped region is a coincidence-maker
    return True


def fit(pairs, deadline, skeleton=None, max_depth=MAX_DEPTH):
    """All programs (up to max_depth, or following `skeleton`) reproducing every pair."""
    inputs = tuple(x for x, _ in pairs)
    ys = tuple(y for _, y in pairs)
    frontier = [((), inputs)]
    seen = {inputs}
    sols = []
    depth_limit = len(skeleton) if skeleton else max_depth
    for depth in range(depth_limit):
        nxt = []
        ops = [skeleton[depth]] if skeleton else [o for o in REGISTRY if o not in ENGINE_OPS and
                                                  not (o == "per_region" and _lift.INNER[0])]
        for prog, gs in frontier:
            prev = prog[-1][0] if prog else None
            last = depth == depth_limit - 1
            same_shape = all(len(g) == len(y) and len(g[0]) == len(y[0]) for g, y in zip(gs, ys))
            for op in ops:
                if not skeleton and not _allowed(prev, op):
                    continue
                if last and not same_shape and op in SHAPE_PRESERVING:
                    continue
                if op in TERMINAL and (not same_shape or (depth > 1 and not skeleton)):
                    continue
                if op in FIRST_ONLY and depth > 0:
                    continue
                if op in SCENE_FINAL and depth > 0 and not skeleton and not (op == "crop_window" and prev in GEOMETRIC):
                    continue
                if op in INPUT_ONLY and depth > 0 and not (depth == 1 and prog[0][0] in PREFIX_OK):
                    continue  # a scene-reading rule may follow ONE cheap view change (OPT184)
                if prog and prog[0][0] in FIRST_ONLY and depth >= 2 and not skeleton:
                    continue  # evidence rule: a high-branching selection supports at most one more step
                for params in _param_options(op, gs, ys, inputs):
                    if time.time() > deadline:
                        return sols
                    step = (op, params)
                    new = _apply_step(step, gs, inputs)
                    if new is None:
                        continue
                    p2 = prog + (step,)
                    if new == ys:
                        sols.append(p2)
                    elif (op in TERMINAL or op in SCENE_FINAL) and not skeleton:
                        continue
                    elif new not in seen:
                        seen.add(new); nxt.append((p2, new))
        frontier = nxt
        if sols and not skeleton:
            break  # shortest programs only
    return sols


def _form(v):
    """How a parameter value was derived: vocabulary choices keep their value, tagged sizes keep
    their tag, 'auto' stays 'auto', literals and learned tables collapse to their kind."""
    if isinstance(v, str):
        return v
    if isinstance(v, tuple) and len(v) == 3 and v[0] in ("factor", "cells"):
        return v[0]
    if isinstance(v, tuple) and v and isinstance(v[0], tuple):
        return "learned"
    if isinstance(v, tuple) and len(v) == 3 and isinstance(v[0], str) and v[0].startswith("bg"):
        return ("bg",) + v[1:]   # OPT199 task-background object view: keeps connectivity / multicolour
    if isinstance(v, tuple) and all(isinstance(x, (int, bool)) for x in v):
        return v  # vocabulary choices (views, directions) and shape ratios keep their identity
    return "literal"


def signature(program):
    # a fitted grammar's spec IS its derivation: refits must use the same spec
    return tuple((op, tuple((k, v if op in FITTED else _form(v)) for k, v in params)) for op, params in program)


# ------------------------------------------------------------------ guided composition (OPT200)
BEAM_WIDTH = 24
BEAM_DEPTH = 4


def _progress(gs, ys):
    """How close intermediate grids are to the outputs: per pair 0.3 for the right shape plus
    0.7 x the share of equal cells (0 when the shape differs)."""
    tot = 0.0
    for g, y in zip(gs, ys):
        if len(g) != len(y) or len(g[0]) != len(y[0]):
            hs = min(len(g), len(y)) / max(len(g), len(y)); ws = min(len(g[0]), len(y[0])) / max(len(g[0]), len(y[0]))
            tot += 0.1 * hs * ws
            continue
        n = len(y) * len(y[0])
        tot += 0.3 + 0.7 * sum(a == b for ra, rb in zip(g, y) for a, b in zip(ra, rb)) / n
    return tot / len(ys)


def beam_fit(pairs, deadline, width=BEAM_WIDTH, max_depth=BEAM_DEPTH):
    """Guided search for longer compositions: keeps the `width` partial programs whose grids are
    closest to the outputs and extends them, up to max_depth steps. Scene operations may follow
    other steps here; a step must change the grids. Returns the programs of the first depth that
    reproduces every pair."""
    inputs = tuple(x for x, _ in pairs)
    ys = tuple(y for _, y in pairs)
    beam = [((), inputs, _progress(inputs, ys))]
    seen = {inputs}
    ops_all = [o for o in REGISTRY if o not in ENGINE_OPS and o != "per_region"]
    for depth in range(max_depth):
        children, sols = [], []
        for prog, gs, sc in beam:
            prev = prog[-1][0] if prog else None
            same_shape = all(len(g) == len(y) and len(g[0]) == len(y[0]) for g, y in zip(gs, ys))
            for op in ops_all:
                if not _allowed(prev, op) or (op in FIRST_ONLY and depth > 0):
                    continue
                if op in INPUT_ONLY and depth > 0:
                    continue
                if op in TERMINAL and not same_shape:
                    continue
                if depth == max_depth - 1 and not same_shape and op in SHAPE_PRESERVING:
                    continue
                for params in _param_options(op, gs, ys, inputs):
                    if time.time() > deadline:
                        return sols
                    step = (op, params)
                    new = _apply_step(step, gs, inputs)
                    if new is None or new == gs or new in seen:
                        continue
                    seen.add(new)
                    p2 = prog + (step,)
                    if new == ys:
                        sols.append(p2)
                    elif op not in TERMINAL:
                        children.append((p2, new, _progress(new, ys)))
        if sols:
            return sols
        children.sort(key=lambda c: (-c[2], cost(c[0])))
        beam = children[:width]
        if not beam:
            break
    return []


def _sound_gain(gs, new, ys):
    gain = 0
    for g, n, y in zip(gs, new, ys):
        if len(n) != len(y) or len(n[0]) != len(y[0]):
            return None
        for rg, rn, ry in zip(g, n, y):
            for a, b, c in zip(rg, rn, ry):
                if a != b:
                    if b != c:
                        return None
                    gain += 1
    return gain


SELECT_VIEW = ("obj", 8, True)
_RF = {}


def _selected_steps(step, gs, new, ys, inputs, bgv):
    """Restrictions of a step to the objects of one property value that make it sound."""
    from .lift import regions, _features, PROPS
    key = (gs, bgv)
    if key not in _RF:
        _RF.clear()
        rf = []
        for g in gs:
            regs = regions(g, SELECT_VIEW, resolve_bg(bgv, g))
            if not regs:
                rf = None; break
            rf.append((regs, _features(g, regs)))
        _RF[key] = rf
    regs_feats = _RF[key]
    if not regs_feats:
        return []
    # which property values are 'good' (every changed cell inside correct) / 'bad' per region
    out = []
    for prop in PROPS:
        good, bad = set(), set()
        for (regs, feats), g, n, y in zip(regs_feats, gs, new, ys):
            for (cells, _, _), f in zip(regs, feats):
                ch = [(r, c) for r, c in cells if n[r][c] != g[r][c]]
                if not ch:
                    continue
                (good if all(n[r][c] == y[r][c] for r, c in ch) else bad).add(f[prop])
        for v in sorted(good - bad, key=repr):
            spec = (SELECT_VIEW, bgv, prop, v, step)
            res = _apply_step(("where", (("spec", spec),)), gs, inputs)
            if res is None:
                continue
            gain = _sound_gain(gs, res, ys)
            if gain:
                out.append((gain, ("where", (("spec", spec),)), res))
    return out


def greedy_compose(pairs, deadline, max_steps=5, validate=True):
    """OPT200 goal-directed composition (same-shape tasks). A step is SOUND when every cell it
    changes, in every demonstration, takes its output value. Among sound steps - plain steps, and
    steps restricted to the objects with one property value ('where') - the one fixing the most
    still-wrong cells is applied, and the search continues from the result until every
    demonstration is reproduced. With >= 3 demonstrations the composition is re-run leaving each
    demonstration out and must predict it."""
    inputs = tuple(x for x, _ in pairs)
    ys = tuple(y for _, y in pairs)
    if any(len(x) != len(y) or len(x[0]) != len(y[0]) for x, y in zip(inputs, ys)) or inputs == ys:
        return []
    from .infer import background as _bgrule
    bgv = _bgrule(inputs, ys)[0]
    gs, prog = inputs, ()
    ops_all = [o for o in REGISTRY if o not in ENGINE_OPS and o not in ("where", "per_region")]
    for depth in range(max_steps):
        best = None
        prev = prog[-1][0] if prog else None
        for op in ops_all:
            if not _allowed(prev, op) or (op in FIRST_ONLY and depth > 0) or (op in INPUT_ONLY and depth > 0):
                continue
            for params in _param_options(op, gs, ys, inputs):
                if time.time() > deadline:
                    return []
                new = _apply_step((op, params), gs, inputs)
                if new is None or new == gs:
                    continue
                gain = _sound_gain(gs, new, ys)
                cands = [(gain, (op, params), new)] if gain else \
                    (_selected_steps((op, params), gs, new, ys, inputs, bgv)
                     if all(len(n) == len(g) and len(n[0]) == len(g[0]) for n, g in zip(new, gs)) else [])
                for g_, st, res in cands:
                    key = (g_, -cost((st,)))
                    if best is None or key > best[0]:
                        best = (key, st, res)
        if best is None:
            return []
        prog, gs = prog + (best[1],), best[2]
        if gs == ys:
            break
    else:
        return []
    if len(prog) < 2 and prog[0][0] != "where":
        return []
    if validate and len(pairs) >= 3:
        for i in range(len(pairs)):
            sub = greedy_compose(pairs[:i] + pairs[i + 1:], deadline, max_steps, validate=False)
            if not sub or run(sub[0], pairs[i][0]) != pairs[i][1]:
                return []
    return [prog]


def _validated(program, pairs, deadline, cache=None):
    """Leave-one-demonstration-out: re-fit the same skeleton on the other demonstrations; refits
    whose parameters are derived the same way must all predict the held-out output.
    `cache` memoises refits per (skeleton, held-out index) within one solve."""
    if any(op == "where" for op, _ in program):
        return True   # built and validated by greedy_compose itself
    if len(pairs) < 3 or all(op in ENGINE_OPS for op, _ in program):
        return True
    skeleton, sig = tuple(op for op, _ in program), signature(program)
    for i, (x, y) in enumerate(pairs):
        key = (skeleton, i)
        if cache is None or key not in cache:
            from .evidence import REFIT
            REFIT[0] = True
            try:
                fitted = fit(pairs[:i] + pairs[i + 1:], deadline, skeleton=skeleton)
            finally:
                REFIT[0] = False
            if cache is not None:
                cache[key] = fitted
        else:
            fitted = cache[key]
        refits = [p for p in fitted if signature(p) == sig]
        if {run(p, x) for p in refits} != {y}:
            return False
    return True


def solve(pairs, test_inputs, seconds=6.0):
    """Ranked list of (predictions for every test input, program)."""
    pairs = [(tuple(map(tuple, x)), tuple(map(tuple, y))) for x, y in pairs]
    tests = [tuple(map(tuple, t)) for t in test_inputs]
    start = time.time()
    deadline = start + seconds
    _engines.DEADLINE[0] = deadline
    sols = fit(pairs, min(deadline, start + TIER1_SECONDS))
    engine_sols = []
    # second tier: engines, in order. When the plain tier already fits, the engines still run (time
    # permitting) to offer a different answer for attempt 2 - scoring counts either attempt.
    # synthesisers (fast) all run; learners run while fewer than two answers exist. Answers are
    # ranked by how many engine programs agree on them, then by engine order.
    votes, first = {}, {}
    for n_i, name in enumerate(_ENGINES):
        if time.time() > deadline:
            break
        if name not in _engines.SYNTHESISERS and len(votes) >= 2:
            continue
        for k, p in enumerate(fit(pairs, deadline, skeleton=(name,))):
            preds = tuple(run(p, t) for t in tests)
            if any(pr is None for pr in preds):
                continue
            votes[preds] = votes.get(preds, 0) + 1
            first.setdefault(preds, ((n_i, k), p))
    engine_sols = [first[pr][1] for pr in sorted(votes, key=lambda pr: (-votes[pr], first[pr][0]))]
    if not sols and not engine_sols and BEAM[0]:
        # OPT200: nothing fits - goal-directed composition with the time that is left
        sols = greedy_compose(pairs, max(deadline, time.time() + BEAM_SECONDS))
    if not sols and not engine_sols:
        return []
    if not sols:
        sols, engine_sols = engine_sols, []
    sols.sort(key=cost)
    checked = {}
    refit_cache = {}
    ranked = []
    # cheapest tier first (cost <= best + 1); if nothing in it survives validation, the next tier is
    # tried, so an invalid cheap coincidence cannot hide a valid slightly dearer program (OPT181)
    rest = list(sols)
    while rest and not ranked:
        best = cost(rest[0])
        tier = [p for p in rest if cost(p) <= best + 1]
        rest = [p for p in rest if cost(p) > best + 1]
        ranked = _rank_tier(tier, pairs, tests, deadline, checked, refit_cache)
    # group identical predictions; order by best cost then number of agreeing programs
    groups = defaultdict(list)
    for c, preds, p in ranked:
        groups[preds].append((c, p))
    order = sorted(groups.items(), key=lambda kv: (min(c for c, _ in kv[1]), -len(kv[1])))
    ranked_out = [(preds, min(progs, key=lambda cp: cp[0])[1]) for preds, progs in order]
    # attempt 2 from the engine tier, when it predicts something the plain tier's best does not
    for p in engine_sols:
        preds = tuple(run(p, t) for t in tests)
        if any(pr is None for pr in preds) or (ranked_out and preds == ranked_out[0][0]):
            continue
        ranked_out.insert(1, (preds, p))
        break
    return ranked_out


def _rank_tier(tier, pairs, tests, deadline, checked, refit_cache):
    ranked = []
    for p in tier:
        sig = signature(p)
        if sig not in checked:
            checked[sig] = _validated(p, pairs, deadline + 4.0, refit_cache)
        if not checked[sig]:
            continue
        preds = tuple(run(p, t) for t in tests)
        if any(pr is None for pr in preds):
            continue
        ranked.append((cost(p), preds, p))
    return ranked


def show(program):
    def fmt(v):
        if isinstance(v, tuple) and len(v) == 3 and v[0] in ("factor", "cells"):
            return f"{v[0]}{v[1]}x{v[2]}"
        if isinstance(v, tuple) and v and all(isinstance(r, tuple) and r and all(isinstance(t, str) for t in r) for r in v):
            return "/".join("-".join(r) for r in v)  # transform layouts
        if isinstance(v, tuple) and v and isinstance(v[0], tuple):
            return "{" + ",".join(f"{a}" for a, _ in v) + "}" if not isinstance(v[0][1], int) else \
                   "{" + ",".join(f"{a}>{b}" for a, b in v) + "}"
        if isinstance(v, tuple):
            return "x".join(map(str, v))
        return str(v)
    def safe(v):
        try:
            return fmt(v)
        except (ValueError, TypeError, IndexError):
            return repr(v).replace(" ", "")
    return " | ".join(op + ("(" + ",".join(f"{k}={safe(v)}" for k, v in params) + ")" if params else "")
                      for op, params in program)


def _tup(v):
    return tuple(_tup(x) for x in v) if isinstance(v, list) else v


def run_spec(g, spec):
    """Run a stored program_spec (schema luca-dsl-1) on a grid."""
    program = tuple((st["op"], tuple((k, _tup(v)) for k, v in st["params"].items())) for st in spec["steps"])
    return run(program, tuple(map(tuple, g)))


def overrides(program):
    """May this program replace an answer an earlier verified rule already gave?
    Selection programs search a large hypothesis space, so they only fill gaps or take attempt 2."""
    return not any(op in FIRST_ONLY for op, _ in program)



def near_miss(pairs, test_inputs, seconds=6.0, top=2):
    """Last tier, when no program fits every demonstration: programs found on each leave-one-out
    subset (plain DSL tier, then the primitive-program engine), ranked by exact demonstrations,
    then cell agreement, then cost. Returns ranked (predictions, program) like solve()."""
    pairs = [(tuple(map(tuple, x)), tuple(map(tuple, y))) for x, y in pairs]
    tests = [tuple(map(tuple, t)) for t in test_inputs]
    if len(pairs) < 2:
        return []
    end = time.time() + seconds
    pool = {}
    for i in range(len(pairs)):
        sub = pairs[:i] + pairs[i + 1:]
        left = end - time.time()
        if left <= 0:
            break
        _engines.DEADLINE[0] = end
        progs = fit(sub, time.time() + left / (len(pairs) - i))
        if not progs and time.time() < end:
            progs = fit(sub, end, skeleton=("primitive_programs",))
        for p in progs:
            outs = [run(p, x) for x, _ in pairs]
            if any(o is None for o in outs):
                continue
            exact = sum(o == y for o, (_, y) in zip(outs, pairs))
            cells = sum(sum(a == b for ra, rb in zip(o, y) for a, b in zip(ra, rb)) / max(1, len(y) * len(y[0]))
                        if len(o) == len(y) and len(o[0]) == len(y[0]) else 0 for o, (_, y) in zip(outs, pairs))
            pool.setdefault(show(p), ((-exact, -cells, cost(p)), p))
    ranked, seen = [], set()
    for _, p in sorted(pool.values(), key=lambda kv: kv[0]):
        preds = tuple(run(p, t) for t in tests)
        if any(pr is None for pr in preds) or preds in seen or any(pr == t for pr, t in zip(preds, tests)):
            continue
        seen.add(preds); ranked.append((preds, p))
        if len(ranked) >= top:
            break
    return ranked
