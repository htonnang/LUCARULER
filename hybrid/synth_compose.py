"""LUCA hybrid - synthetic tasks from NEW compositions of LUCA operations.

generate.py turns programs LUCA found for ARC training tasks into variants of those tasks. This
generator instead draws random programs - 1 to 3 LUCA operations with random settings (gravity
direction, ray direction, symmetry mode, colours, scale factors, ...) - so the tasks combine
transformations in ways the ARC training set does not contain. Input grids come from ARC
training tasks (demonstration and test inputs, and perturbed copies of them), so they look like
ARC grids. A task is kept only when:
  * the program applies to every grid and changes every one of them non-trivially;
  * the outputs differ from each other and fit in 30 x 30;
  * every operation matters: dropping any single operation changes at least one output
    (no redundant steps such as rot180 twice);
  * the same program has not produced a task from the same source task before.
Each task: 3-4 demonstrations + 1 test pair, correct by construction.

Usage:
    python3 hybrid/synth_compose.py SOURCE_DIR OUT.json [--tasks 30000] [--workers 2] [--seed 0]
SOURCE_DIR: ARC training tasks (one JSON per task). Never point it at evaluation tasks.
"""
import argparse, hashlib, json, os, random, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COL = list(range(1, 10))


def _c(rng):
    return rng.choice(COL)


# op name -> sampler(rng) -> params (backgrounds are resolved per grid: 'auto' = most frequent colour)
SAMPLERS = {
    "rot90": lambda r: {}, "rot180": lambda r: {}, "rot270": lambda r: {},
    "flip_h": lambda r: {}, "flip_v": lambda r: {}, "transpose": lambda r: {},
    "crop_content": lambda r: {"bg": "auto"},
    "gravity": lambda r: {"direction": r.choice(["down", "up", "left", "right"]), "bg": "auto"},
    "translate": lambda r: {"shift": r.choice([(a, b) for a in range(-3, 4) for b in range(-3, 4) if (a, b) != (0, 0)]),
                            "bg": "auto"},
    "keep_largest": lambda r: {"conn": r.choice([4, 8]), "bg": "auto"},
    "hollow": lambda r: {"bg": "auto"},
    "flip_objects": lambda r: {"axis": r.choice(["h", "v"]), "bg": "auto"},
    "fill_enclosed": lambda r: {"colour": _c(r), "conn": r.choice([4, 8]), "bg": "auto", "outside": "keep"},
    "frame": lambda r: {"colour": _c(r)},
    "extend_lines": lambda r: {"bg": "auto", "cross": _c(r)},
    "symmetrize": lambda r: {"mode": r.choice(["h", "v", "both", "r180", "tr"]), "bg": "auto"},
    "mirror_about": lambda r: {"mode": r.choice(["both", "h", "v", "h_move", "v_move", "both_move"]), "bg": "auto"},
    "rays": lambda r: {"direction": r.choice([(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 1), (1, 0), (0, -1), (-1, 0)]),
                       "bg": "auto"},
    "scale_up": lambda r: {"factor": r.choice([(2, 2), (3, 3), (1, 2), (2, 1)])},
    "tile": lambda r: {"reps": r.choice([(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)])},
    "box_objects": lambda r: {"conn": r.choice([4, 8]), "mode": r.choice(["fill", "ring"]), "ink": _c(r), "bg": "auto"},
    "fill_bbox_by_colour": lambda r: {"bg": "auto", "mode": r.choice(["fill", "outline"])},
    "rings": lambda r: {"mode": r.choice(["reverse", "inward", "outward"])},
    "fill_touching": lambda r: {"colour": _c(r), "conn": r.choice([4, 8]), "bg": "auto"},
    "swap_colours": lambda r: {},
    "dedup_rows": lambda r: {}, "dedup_cols": lambda r: {},
}
GEOM = {"rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"}
NAMES = sorted(SAMPLERS)


def T(g):
    return tuple(tuple(r) for r in g)


def _recolor_params(rng, grids):
    cols = sorted({v for g in grids for r in g for v in r})
    if len(cols) < 2:
        return None
    a = rng.choice(cols)
    b = rng.choice([c for c in range(10) if c != a])
    return {"mapping": ((a, b),)}


def sample_program(rng, grids):
    n = rng.choices([1, 2, 3], weights=[3, 5, 2])[0]
    prog = []
    for _ in range(n):
        if rng.random() < 0.08:
            p = _recolor_params(rng, grids)
            if p:
                prog.append(("recolor", p)); continue
        name = rng.choice(NAMES)
        prog.append((name, SAMPLERS[name](rng)))
    if any(a[0] in GEOM and b[0] in GEOM for a, b in zip(prog, prog[1:])):
        return None                                # two symmetries in a row collapse into one
    if all(op in GEOM for op, _ in prog):          # pure geometry is well covered already: mostly skip
        if rng.random() < 0.8:
            return None
    # LUCA programs carry their settings as sorted (name, value) pairs
    return [(op, tuple(sorted(p.items()))) for op, p in prog]


def _key(prog):
    return json.dumps([[op, [(k, str(v)) for k, v in p]] for op, p in prog])


def _trivial(y):
    return len({v for r in y for v in r}) < 2 and len(y) * len(y[0]) > 1


def make_task(prog, inputs, rng, run):
    rng.shuffle(inputs)
    pairs = []
    for x in inputs:
        if len(x) > 30 or len(x[0]) > 30:
            continue
        y = run(prog, x)
        if y is None or y == x or _trivial(y) or len(y) > 30 or len(y[0]) > 30:
            continue
        pairs.append((x, y))
        if len(pairs) == 5:
            break
    if len(pairs) < 4 or len({y for _, y in pairs}) < len(pairs):
        return None
    # every operation matters
    for i in range(len(prog)):
        rest = prog[:i] + prog[i + 1:]
        if rest and all(run(rest, x) == y for x, y in pairs):
            return None
    tr, te = pairs[:-1], pairs[-1]
    L = lambda g: [list(r) for r in g]
    return {"train": [{"input": L(x), "output": L(y)} for x, y in tr],
            "test": [{"input": L(te[0]), "output": L(te[1])}]}


def _job(args):
    paths, n_target, seed = args
    from luca_v31.dsl import search as S
    from generate import perturb
    rng = random.Random(seed)
    out, seen = [], set()
    t0 = time.time()
    tries = 0
    while len(out) < n_target and tries < n_target * 60:
        tries += 1
        p = rng.choice(paths)
        try:
            task = json.load(open(p))
        except Exception:
            continue
        base = [T(e["input"]) for e in task["train"]] + [T(e["input"]) for e in task["test"]]
        grids = list(base)
        for _ in range(6):
            g = perturb(rng.choice(base), rng)
            if g is not None and g not in grids:
                grids.append(g)
        prog = sample_program(rng, grids)
        if prog is None:
            continue
        key = (os.path.basename(p), _key(prog))
        if key in seen:
            continue
        t = make_task(prog, grids, rng, S.run)
        if t is None:
            continue
        seen.add(key)
        t["meta"] = {"source": os.path.basename(p)[:-5], "program": S.show(prog)}
        out.append(t)
    return out, tries, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source"); ap.add_argument("out")
    ap.add_argument("--tasks", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    paths = sorted(os.path.join(a.source, f) for f in os.listdir(a.source) if f.endswith(".json"))
    chunks = max(1, a.tasks // 250)
    jobs = [(paths, -(-a.tasks // chunks), a.seed * 100003 + k) for k in range(chunks)]
    tasks, tries = [], 0
    t0 = time.time()
    with Pool(a.workers) as pool:
        for out, n, _ in pool.imap_unordered(_job, jobs):
            tasks += out; tries += n
            print(f"{len(tasks)} tasks ({tries} programs tried, {time.time() - t0:.0f}s)", flush=True)
    # de-duplicate identical tasks
    uniq, hs = {}, set()
    for t in tasks:
        h = hashlib.sha1(json.dumps(t["train"] + t["test"]).encode()).hexdigest()[:16]
        if h not in hs:
            hs.add(h); uniq[f"cmp{len(uniq):06d}"] = t
    json.dump(uniq, open(a.out, "w"))
    from collections import Counter
    ops = Counter(s.split("(")[0] for t in uniq.values() for s in t["meta"]["program"].split(" | "))
    lens = Counter(t["meta"]["program"].count(" | ") + 1 for t in uniq.values())
    print(f"wrote {a.out}: {len(uniq)} tasks | program lengths {dict(sorted(lens.items()))}")
    print("operations used:", dict(ops.most_common()))


if __name__ == "__main__":
    main()
