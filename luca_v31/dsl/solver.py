"""LUCA general solver: one language (the typed DSL) plus one search.

solve_task(task) -> one result per test input: dict(predictions=[attempt1, attempt2], laws=[...])
  1. DSL search on the demonstrations; the two best distinct predictions are the two attempts.
  2. If nothing fits: colours relabelled by role (luca_v31.canon), searched again, mapped back.
  3. If still nothing: the test input itself (identity) as the only guess.
"""
from .search import solve, show

SECONDS = 30.0
NEAR_MISS_SECONDS = 10.0   # OPT199: last tier (only for tasks with no fitting program); was 6


def _inner(prog):
    """For an engine operation: the name of the rule its fitter produced, e.g. '{tile_output_mirror}'."""
    from .engines import ENGINES, _STORE
    if len(prog) != 1 or prog[0][0] not in ENGINES:
        return ""
    spec = _STORE.get(dict(prog[0][1]).get("spec"))
    name = getattr(spec, "name", None)
    if name is None and isinstance(spec, tuple) and spec and isinstance(spec[0], str):
        name = spec[0]
    return "{" + str(name) + "}" if name else ""


def _attempts(ranked, i):
    preds, laws = [], []
    for p, prog in ranked:
        if p[i] is None:
            continue
        g = [list(r) for r in p[i]]
        if g not in preds:
            preds.append(g); laws.append("dsl:" + show(prog) + _inner(prog))
        if len(preds) == 2:
            break
    return preds, laws


def _search(train, tests, seconds):
    pairs = [(p["input"], p["output"]) for p in train]
    try:
        return solve(pairs, tests, seconds=seconds)
    except Exception:
        return []


def _fill_second(task, tests, out, seconds=5.0):
    """Attempt 2 is empty: offer the best near-miss program's answer there (attempt 1 unchanged)."""
    from .search import near_miss
    try:
        nm = near_miss([(p["input"], p["output"]) for p in task["train"]], tests, seconds=seconds)
    except Exception:
        return
    for i, r in enumerate(out):
        if len(r["predictions"]) >= 2:
            continue
        for p, prog in nm:
            g = [list(x) for x in p[i]] if p[i] is not None else None
            if g is not None and g not in r["predictions"]:
                r["predictions"].append(g); r["laws"].append("near_miss:dsl:" + show(prog)); break


def solve_task(task, seconds=SECONDS):
    tests = [t["input"] for t in task["test"]]
    ranked = _search(task["train"], tests, seconds)
    out = []
    if ranked:
        for i in range(len(tests)):
            preds, laws = _attempts(ranked, i)
            out.append(dict(predictions=preds, laws=laws, explanation="DSL program fits every demonstration."))
        if all(r["predictions"] for r in out):
            if any(len(r["predictions"]) < 2 for r in out):
                _fill_second(task, tests, out)
            return out
    # colour roles: relabel every pair's colours by frequency rank and search again
    from ..canon import normalise_task, denormalise
    norm = normalise_task({"train": task["train"], "test": [{"input": t} for t in tests]})
    if norm is not None:
        canon, invs = norm
        cr = _search(canon["train"], [t["input"] for t in canon["test"]], seconds)
        if cr:
            res = []
            for i, inv in enumerate(invs):
                preds, laws = _attempts(cr, i)
                res.append(dict(predictions=[denormalise(p, inv) for p in preds],
                                laws=[l + " (colour roles)" for l in laws],
                                explanation="DSL program found with colours relabelled by role."))
            if all(r["predictions"] for r in res):
                return res
    # near miss: programs that fit all but one demonstration, ranked by agreement
    from .search import near_miss
    try:
        nm = near_miss([(p["input"], p["output"]) for p in task["train"]], tests, seconds=NEAR_MISS_SECONDS)
    except Exception:
        nm = []
    if nm:
        out = []
        for i, t in enumerate(tests):
            preds, laws = _attempts(nm, i)
            if len(preds) < 2 and [list(r) for r in t] not in preds:
                preds.append([list(r) for r in t]); laws.append("identity")
            out.append(dict(predictions=preds, laws=["near_miss:" + l if l != "identity" else l for l in laws],
                            explanation="No program fits every demonstration; near-miss programs."))
        return out
    return [dict(predictions=[[list(r) for r in t]], laws=["identity"], explanation="No program fits.")
            for t in tests]
