"""LUCA hybrid - combine LUCA with a neural model's candidate answers.

The neural model runs separately (e.g. on a GPU machine) and writes its candidates to a JSON file:
    {"<task_id>": [ [ {"grid": [[...]], "score": 0.93}, ... ],      # test input 0
                    [ ... ] ]}                                         # test input 1 ...
Scores are any monotone confidence (higher = better), e.g. vote counts over augmentations.

Policy, per test input (two attempts):
  1. LUCA runs first. If it has an EXACT answer (a program reproducing every demonstration, not a
     near miss or the identity guess), that is attempt 1.
  2. Neural candidates are checked against what the demonstrations imply (output size rule, colour
     palette); candidates that contradict them are dropped.
  3. The attempts are filled in order: LUCA's best exact answer, the best surviving neural
     candidate, LUCA's other exact answers, other neural candidates, LUCA's near misses, identity.

Usage:
    python3 hybrid/solve_hybrid.py TASK_DIR NEURAL.json OUT_DIR [--seconds 30] [--workers 2] [--luca-dir DIR]
    python3 hybrid/solve_hybrid.py --score TASK_DIR OUT_DIR        # tasks with test outputs only
"""
import argparse, json, os, sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constraints as C                       # noqa: E402
SCREEN = os.environ.get("HYBRID_SCREEN", "1") != "0"   # HYBRID_SCREEN=0: the old order (for comparison)


def size_rule(task):
    """What the demonstrations imply about the output size: a function test_input -> set of
    allowed (h, w), or None when nothing is implied."""
    sh = [((len(p["input"]), len(p["input"][0])), (len(p["output"]), len(p["output"][0]))) for p in task["train"]]
    if all(a == b for a, b in sh):
        return lambda x: {(len(x), len(x[0]))}
    outs = {b for _, b in sh}
    if len(outs) == 1:
        return lambda x: set(outs)
    ratios = {(b[0] / a[0], b[1] / a[1]) for a, b in sh}
    if len(ratios) == 1:
        (rh, rw), = ratios
        return lambda x: {(round(len(x) * rh), round(len(x[0]) * rw))}
    return None


def palette_rule(task):
    """Colours an answer may use: colours of the demonstration outputs and of the test input."""
    demo = {v for p in task["train"] for r in p["output"] for v in r}
    return lambda x: demo | {v for r in x for v in r}


def consistent(grid, x, sizes, palette):
    if not grid or not grid[0] or any(len(r) != len(grid[0]) for r in grid):
        return False
    if sizes is not None and (len(grid), len(grid[0])) not in sizes(x):
        return False
    return {v for r in grid for v in r} <= palette(x)


def combine(task, luca, neural):
    """luca: solve_task() result; neural: per test input list of {grid, score} (may be empty)."""
    sizes, palette = size_rule(task), palette_rule(task)
    rules = C.fit_rules(task) if SCREEN else []
    out = []
    for i, t in enumerate(task["test"]):
        x = t["input"]
        r = luca[i] if i < len(luca) else {"predictions": [], "laws": []}
        laws = r.get("laws") or []
        preds = r.get("predictions") or []
        luca_exact = [(p, l) for p, l in zip(preds, laws) if l.startswith("dsl:")]
        luca_other = [(p, l) for p, l in zip(preds, laws) if not l.startswith("dsl:") and l != "identity"]
        # screening: candidates breaking fewer of the demonstrations' rules first, then by confidence
        cands = sorted((c for c in (neural[i] if i < len(neural) else [])
                        if consistent(c["grid"], x, sizes, palette)),
                       key=lambda c: (len(C.broken(rules, x, c["grid"])) if rules else 0, -c.get("score", 0)))
        neural_c = [(c["grid"], "neural:%.3g" % c.get("score", 0)) for c in cands]
        lucx = [(p, "luca:" + l) for p, l in luca_exact]
        # interleave: LUCA's best exact answer, the model's best answer, then the rest
        order = lucx[:1] + neural_c[:1] + lucx[1:] + neural_c[1:] + \
                [(p, "luca:" + l) for p, l in luca_other] + [(x, "identity")]
        attempts, sources = [], []
        for g, src in order:
            if g not in attempts:
                attempts.append(g); sources.append(src)
            if len(attempts) == 2:
                break
        out.append({"predictions": attempts, "sources": sources})
    return out


def _luca_from(prev_dir, tid):
    """LUCA's results recovered from an earlier hybrid output folder (its 'luca:' sources)."""
    p = os.path.join(prev_dir, tid + ".json")
    if not os.path.exists(p):
        return None
    res = []
    for r in json.load(open(p)):
        preds, laws = [], []
        for g, src in zip(r.get("predictions", []), r.get("sources", [])):
            if src.startswith("luca:"):
                preds.append(g); laws.append(src[5:])
        res.append({"predictions": preds, "laws": laws})
    return res


def _job(args):
    path, neural, out_dir, seconds, luca_dir = args
    from luca_v31.dsl.solver import solve_task
    tid = os.path.basename(path)[:-5]
    task = json.load(open(path))
    luca = _luca_from(luca_dir, tid) if luca_dir else None
    if luca is None:
        try:
            luca = solve_task(task, seconds)
        except Exception:
            luca = []
    res = combine(task, luca, neural.get(tid, []))
    json.dump(res, open(os.path.join(out_dir, tid + ".json"), "w"))
    return tid


def score(task_dir, out_dir):
    n = solved = 0
    by = {}
    for f in sorted(os.listdir(task_dir)):
        if not f.endswith(".json") or not os.path.exists(os.path.join(out_dir, f)):
            continue
        t = json.load(open(os.path.join(task_dir, f))); r = json.load(open(os.path.join(out_dir, f)))
        n += 1
        ok = all(any(p == tt["output"] for p in rr["predictions"][:2]) for rr, tt in zip(r, t["test"]))
        solved += ok
        if ok:
            src = next(s for p, s in zip(r[0]["predictions"], r[0]["sources"]) if p == t["test"][0]["output"])
            key = src.split(":")[0]
            by[key] = by.get(key, 0) + 1
    print(f"solved {solved}/{n}", by)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--luca-dir", help="reuse LUCA's answers from an earlier output folder (e.g. luca_only_out)")
    a = ap.parse_args()
    if a.score:
        return score(*a.paths)
    task_dir, neural_p, out_dir = a.paths
    os.makedirs(out_dir, exist_ok=True)
    neural = json.load(open(neural_p)) if os.path.exists(neural_p) else {}
    files = sorted(f for f in os.listdir(task_dir) if f.endswith(".json"))
    jobs = [(os.path.join(task_dir, f), neural, out_dir, a.seconds, a.luca_dir) for f in files
            if not os.path.exists(os.path.join(out_dir, f))]
    with Pool(a.workers) as p:
        for tid in p.imap_unordered(_job, jobs):
            print(tid, flush=True)


if __name__ == "__main__":
    main()
