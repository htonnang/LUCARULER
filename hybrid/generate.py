"""LUCA hybrid - synthetic task generator.

For every source task LUCA solves, the fitted program becomes a generator: new input grids are
derived from the task's demonstration inputs (colour permutation of the non-background colours,
objects moved to free positions, a rotation / reflection, size change by padding or cropping the
background) and the program computes their outputs. Pairs are kept only when the program applies,
the output differs from the input and is not a trivial grid. A synthetic task = 3-4 such pairs for
training plus 1 for testing, all from one program, so its answer is correct by construction.

Usage:
    python3 hybrid/generate.py SOURCE_DIR OUT_DIR [--per-task 20] [--seconds 30] [--workers 2] [--ids FILE]
"""
import argparse, json, os, random, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

D8 = [lambda g: g, lambda g: tuple(zip(*g[::-1])), lambda g: tuple(r[::-1] for r in g[::-1]),
      lambda g: tuple(zip(*g))[::-1], lambda g: tuple(r[::-1] for r in g), lambda g: g[::-1],
      lambda g: tuple(zip(*g)), lambda g: tuple(r[::-1] for r in tuple(zip(*g))[::-1])]


def T(g):
    return tuple(tuple(r) for r in g)


def _bg(g):
    from collections import Counter
    return Counter(v for r in g for v in r).most_common(1)[0][0]


def perturb(g, rng):
    """A new input grid derived from demonstration input g."""
    from luca_v31.dsl import perception as P
    bg = _bg(g)
    g = [list(r) for r in g]
    H, W = len(g), len(g[0])
    kind = rng.random()
    if kind < 0.35:
        # move objects to random free positions
        objs = P.objects(T(g), bg=bg, conn=8, by_colour=False)
        if 1 <= len(objs) <= 20:
            out = [[bg] * W for _ in range(H)]
            taken = set()
            for o in sorted(objs, key=len, reverse=True):
                r0, c0, r1, c1 = P.bbox(o)
                h, w = r1 - r0 + 1, c1 - c0 + 1
                for _ in range(30):
                    a, b = rng.randrange(0, H - h + 1), rng.randrange(0, W - w + 1)
                    cells = [(a + r - r0, b + c - c0) for r, c in o]
                    halo = {(x + i, y + j) for x, y in cells for i in (-1, 0, 1) for j in (-1, 0, 1)}
                    if not halo & taken:
                        for (r, c), (x, y) in zip(o, cells):
                            out[x][y] = g[r][c]
                        taken.update(cells)
                        break
                else:
                    return None
            g = out
    elif kind < 0.55 and H + 2 <= 30 and W + 2 <= 30:
        # grow the canvas with background on random sides
        top, left = rng.randint(0, 2), rng.randint(0, 2)
        bottom, right = rng.randint(0, 2), rng.randint(0, 2)
        g = [[bg] * (W + left + right) for _ in range(top)] + \
            [[bg] * left + r + [bg] * right for r in g] + [[bg] * (W + left + right) for _ in range(bottom)]
    # colour permutation of the non-background colours (always)
    cols = sorted({v for r in g for v in r} - {bg})
    free = [c for c in range(10) if c != bg]
    rng.shuffle(free)
    m = dict(zip(cols, free[:len(cols)]))
    g = [[m.get(v, v) for v in r] for r in g]
    return T(D8[rng.randrange(8)](T(g)))


def _trivial(y):
    return len({v for r in y for v in r}) < 2 and len(y) * len(y[0]) > 1


def generate(task, per_task, seconds, seed):
    """Synthetic tasks from one source task (list of ARC task dicts)."""
    from luca_v31.dsl import search as S
    rng = random.Random(seed)
    pairs = [(T(p["input"]), T(p["output"])) for p in task["train"]]
    ranked = S.solve(pairs, [T(t["input"]) for t in task["test"]], seconds=seconds)
    if not ranked:
        return [], None
    prog = ranked[0][1]
    # the program must reproduce the source's own demonstrations (it does by construction) and
    # its test answer is unknown to us here, so it is never used as a filter
    inputs = [x for x, _ in pairs] + [T(t["input"]) for t in task["test"]]
    out_tasks = []
    tries = 0
    while len(out_tasks) < per_task and tries < per_task * 40:
        tries += 1
        n = rng.choice((3, 4, 5))
        made, seen = [], set()
        for _ in range(n * 6):
            x = perturb(rng.choice(inputs), rng)
            if x is None or x in seen:
                continue
            y = S.run(prog, x)
            if y is None or y == x or _trivial(y) or len(y) > 30 or len(y[0]) > 30:
                continue
            seen.add(x)
            made.append((x, y))
            if len(made) == n:
                break
        if len(made) < 3 or len({y for _, y in made}) < len(made) - 1:
            continue
        tr, te = made[:-1], made[-1]
        out_tasks.append({"train": [{"input": [list(r) for r in x], "output": [list(r) for r in y]} for x, y in tr],
                          "test": [{"input": [list(r) for r in te[0]], "output": [list(r) for r in te[1]]}]})
    return out_tasks, S.show(prog)


def _job(args):
    path, out_dir, per_task, seconds, seed = args
    tid = os.path.basename(path)[:-5]
    try:
        task = json.load(open(path))
        tasks, prog = generate(task, per_task, seconds, seed)
    except Exception as e:
        return tid, 0, "error: " + repr(e)[:120]
    for k, t in enumerate(tasks):
        json.dump(t, open(os.path.join(out_dir, f"{tid}_syn{k:03d}.json"), "w"))
    return tid, len(tasks), prog


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source"); ap.add_argument("out")
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ids", help="JSON list of task ids to use (default: all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids = json.load(open(a.ids)) if a.ids else sorted(f[:-5] for f in os.listdir(a.source) if f.endswith(".json"))
    jobs = [(os.path.join(a.source, i + ".json"), a.out, a.per_task, a.seconds, k) for k, i in enumerate(ids)]
    log = []
    with Pool(a.workers) as p:
        for tid, n, prog in p.imap_unordered(_job, jobs):
            log.append(dict(task=tid, generated=n, program=prog))
            print(tid, n, (prog or "")[:90], flush=True)
    json.dump(log, open(os.path.join(a.out, "_generation_log.json"), "w"), indent=1)
    print("source tasks", len(ids), "with output", sum(1 for l in log if l["generated"]),
          "synthetic tasks", sum(l["generated"] for l in log))


if __name__ == "__main__":
    main()
