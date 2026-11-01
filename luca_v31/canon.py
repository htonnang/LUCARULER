"""Per-pair colour normalisation (colours by role, not by literal value).

For each grid pair (demo input + output, or a test input) the INPUT colours are ranked by frequency
(most frequent = background -> 0, then 1, 2, ...). Output colours that do not occur in the input are
'new' colours: if a new colour is the same literal colour in every demonstration it is kept as a
literal (mapped to 10 + k so it cannot collide), otherwise the pair cannot be normalised.
Ties in frequency are broken by first appearance in reading order.

normalise_task(task) -> (canonical_task, per_test_inverse_maps) or None
denormalise(grid, inverse_map) -> grid in the test's own colours
"""
from collections import Counter


def _rank(g):
    """Colours ranked by frequency; equal counts are ordered by first appearance in reading order
    (OPT187: on the fresh dev half this recovers 20 of 96 otherwise refused tasks)."""
    c = Counter(v for r in g for v in r)
    first = {}
    for i, v in enumerate(v for r in g for v in r):
        first.setdefault(v, i)
    items = sorted(c.items(), key=lambda kv: (-kv[1], first[kv[0]]))
    return {col: i for i, (col, _) in enumerate(items)}


def normalise_task(task):
    demos = task["train"]
    new_cols = []
    maps = []
    for p in demos:
        m = _rank(p["input"])
        if m is None:
            return None
        maps.append(m)
        new_cols.append({v for r in p["output"] for v in r} - set(m))
    lit = set().union(*new_cols) if new_cols else set()
    # every new colour must be a literal shared by all demos that introduce new colours
    if any(nc and nc != lit for nc in new_cols if nc):
        return None
    lit_map = {c: 10 + k for k, c in enumerate(sorted(lit))}
    canon = {"train": [], "test": []}
    for p, m in zip(demos, maps):
        full = {**m, **lit_map}
        canon["train"].append({"input": [[full[v] for v in r] for r in p["input"]],
                               "output": [[full[v] for v in r] for r in p["output"]]})
    invs = []
    for t in task["test"]:
        m = _rank(t["input"])
        if m is None:
            return None
        full = {**m, **lit_map}
        canon["test"].append({"input": [[full[v] for v in r] for r in t["input"]]})
        invs.append({v: k for k, v in full.items()})
    # canonical codes must stay within 0..9 for the solver: remap 10+ literals onto unused codes
    used = {v for p in canon["train"] for g in (p["input"], p["output"]) for r in g for v in r} | \
           {v for t in canon["test"] for r in t["input"] for v in r}
    if any(v > 9 for v in used):
        free = [c for c in range(10) if c not in used]
        big = sorted(v for v in used if v > 9)
        if len(free) < len(big):
            return None
        re = dict(zip(big, free[::-1]))
        def fix(g):
            return [[re.get(v, v) for v in r] for r in g]
        canon["train"] = [{"input": fix(p["input"]), "output": fix(p["output"])} for p in canon["train"]]
        canon["test"] = [{"input": fix(t["input"])} for t in canon["test"]]
        invs = [{re.get(k, k): v for k, v in inv.items()} for inv in invs]
    return canon, invs


def denormalise(grid, inv):
    return [[inv.get(v, v) for v in r] for r in grid]
