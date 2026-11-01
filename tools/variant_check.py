"""LUCA variant checker - does a family that solves a training task also solve variants of it?

For every training task the solver solves, the tool builds VARIANTS whose correct answers are known
exactly, runs the solver on them and reports how many it still solves - per variant kind and per
family. A rule that captures the real transformation solves its variants; a rule that fits
accidents of one task (its direction, its colours, its particular test input) does not.

Variant kinds (all label-safe: the answer of a variant is known exactly):
  rot / flip  the whole task rotated or reflected (7 transforms of the square's symmetry group),
              demonstrations and test alike - a gravity-down task becomes gravity-left, etc.
  colour      one permutation of colours applied to the whole task: 'fg' keeps colour 0 fixed,
              'all' permutes all ten colours
  holdout     each demonstration in turn becomes the test pair and the original test pair joins
              the demonstrations: the same rule, new test inputs (closest to fresh instances)

The solver never sees task ids: variant files are numbered (v000001.json) and carry no outputs.
This tool uses ARC TRAINING tasks only, with their known test outputs; keep evaluation and fresh
sets out of development.

Usage:
    python3 variant_check.py --solver ROOT --tasks TRAINING_DIR --out OUT_DIR
        [--ids ids.txt] [--kinds d8,colour,holdout] [--colour-perms 2] [--workers 4]
        [--timeout 120] [--seed 0] [--solver-args "--some-flag"]
  ROOT          folder that contains the solver's run.py  (run.py TASK.json --output PRED.json)
  TRAINING_DIR  ARC training tasks, one JSON per task, test outputs included
  --ids         only these task ids (one per line); default: every task the solver solves
Outputs: OUT_DIR/report.txt (summary), OUT_DIR/report.json (per task, kind and family).
"""
import argparse, json, os, random, subprocess, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- grids
D8 = {
    "rot90": lambda g: [list(r) for r in zip(*g[::-1])],
    "rot180": lambda g: [list(r)[::-1] for r in g[::-1]],
    "rot270": lambda g: [list(r) for r in zip(*g)][::-1],
    "flip_lr": lambda g: [list(r)[::-1] for r in g],
    "flip_ud": lambda g: [list(r) for r in g[::-1]],
    "transpose": lambda g: [list(r) for r in zip(*g)],
    "anti_transpose": lambda g: [list(r)[::-1] for r in zip(*g)][::-1],
}


def map_task(task, f):
    return {"train": [{"input": f(p["input"]), "output": f(p["output"])} for p in task["train"]],
            "test": [{"input": f(p["input"]), "output": f(p["output"])} for p in task["test"]]}


def recolour(perm):
    return lambda g: [[perm[v] for v in r] for r in g]


def variants(task, kinds, n_perm, rng):
    """[(kind, variant task with test outputs)]"""
    out = []
    if "d8" in kinds:
        for name, f in D8.items():
            out.append((name, map_task(task, f)))
    if "colour" in kinds:
        for k in range(n_perm):
            fg = list(range(1, 10)); rng.shuffle(fg)
            out.append((f"colour_fg{k}", map_task(task, recolour([0] + fg))))
            al = list(range(10)); rng.shuffle(al)
            out.append((f"colour_all{k}", map_task(task, recolour(al))))
    if "holdout" in kinds:
        pairs = task["train"] + task["test"]
        n_test = len(task["test"])
        if len(pairs) - 1 >= 2:
            for i in range(len(task["train"])):
                test = [pairs[i]]
                train = [p for j, p in enumerate(pairs) if j != i]
                out.append((f"holdout{i}", {"train": train, "test": test}))
    return out


# --------------------------------------------------------------------------- running the solver
RUNNER = r'''
import sys, runpy
root, task, out, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path.insert(0, root)
sys.argv = [root + "/run.py", task, "--output", out] + extra
runpy.run_path(root + "/run.py", run_name="__main__")
'''


def run_solver(root, task_path, out_path, timeout, extra):
    if os.path.exists(out_path):
        return
    try:
        r = subprocess.run([sys.executable, "-I", "-c", RUNNER, root, task_path, out_path] + extra,
                           capture_output=True, timeout=timeout)
        if r.returncode and not os.path.exists(out_path):
            open(out_path + ".err", "w").write(r.stderr.decode()[-2000:])
    except subprocess.TimeoutExpired:
        open(out_path + ".err", "w").write("TIMEOUT")


def read_prediction(path):
    """(predictions per test input or None, family name)"""
    try:
        r = json.load(open(path))
    except Exception:
        return None, None
    if isinstance(r, dict) and "tests" in r:
        preds = [t.get("prediction") if t.get("status") == "unique" else None for t in r["tests"]]
        fam = (r.get("supporting_families") or [None])[0]
        return preds, fam
    if isinstance(r, list):                                     # LUCA hybrid format
        return [(x.get("predictions") or [None])[0] for x in r], None
    return None, None


def correct(path, outputs):
    preds, fam = read_prediction(path)
    if not preds or len(preds) != len(outputs) or any(p is None for p in preds):
        return False, fam
    return all(p == o for p, o in zip(preds, outputs)), fam


def blind(task):
    return {"train": task["train"], "test": [{"input": p["input"]} for p in task["test"]]}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ids")
    ap.add_argument("--kinds", default="d8,colour,holdout")
    ap.add_argument("--colour-perms", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver-args", default="", help="extra arguments for run.py, e.g. a family-only flag")
    a = ap.parse_args()
    root = os.path.abspath(a.solver)
    if not os.path.exists(os.path.join(root, "run.py")):
        sys.exit(f"no run.py in {root}")
    extra = a.solver_args.split() if a.solver_args else []
    kinds = set(a.kinds.split(","))
    os.makedirs(a.out, exist_ok=True)
    vdir = os.path.join(a.out, "variants"); pdir = os.path.join(a.out, "predictions")
    os.makedirs(vdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
    rng = random.Random(a.seed)
    ids = sorted(f[:-5] for f in os.listdir(a.tasks) if f.endswith(".json"))
    if a.ids:
        want = {l.strip() for l in open(a.ids) if l.strip()}
        ids = [i for i in ids if i in want]
    tasks = {i: json.load(open(os.path.join(a.tasks, i + ".json"))) for i in ids}
    t0 = time.time()

    # 1. originals (numbered, no outputs) -> which tasks the solver solves
    jobs, meta, n = [], {}, 0
    for tid, t in tasks.items():
        n += 1; name = f"v{n:06d}.json"
        json.dump(blind(t), open(os.path.join(vdir, name), "w"))
        meta[name] = (tid, "original", [p["output"] for p in t["test"]])
        jobs.append(name)
    run_all(root, vdir, pdir, jobs, a, extra)
    solved, family = set(), {}
    for name in jobs:
        tid, _, outs = meta[name]
        ok, fam = correct(os.path.join(pdir, name), outs)
        if ok:
            solved.add(tid); family[tid] = fam or "unknown"
    print(f"originals: {len(solved)}/{len(tasks)} solved ({time.time() - t0:.0f}s)", flush=True)

    # 2. variants of the solved tasks
    jobs = []
    for tid in sorted(solved):
        for kind, v in variants(tasks[tid], kinds, a.colour_perms, rng):
            n += 1; name = f"v{n:06d}.json"
            json.dump(blind(v), open(os.path.join(vdir, name), "w"))
            meta[name] = (tid, kind, [p["output"] for p in v["test"]])
            jobs.append(name)
    print(f"variants: {len(jobs)} for {len(solved)} solved tasks", flush=True)
    run_all(root, vdir, pdir, jobs, a, extra)

    # 3. report
    per_task = defaultdict(dict)
    for name in jobs:
        tid, kind, outs = meta[name]
        per_task[tid][kind] = correct(os.path.join(pdir, name), outs)[0]

    def group(kind):
        return "rotation/flip" if kind in D8 else ("colour" if kind.startswith("colour") else "holdout")
    by_kind = defaultdict(lambda: [0, 0]); by_fam = defaultdict(lambda: [0, 0, 0, 0])
    for tid, res in per_task.items():
        for kind, ok in res.items():
            by_kind[group(kind)][0] += ok; by_kind[group(kind)][1] += 1
        allok = all(res.values())
        f = family.get(tid, "unknown")
        by_fam[f][0] += 1; by_fam[f][1] += allok
        by_fam[f][2] += sum(res.values()); by_fam[f][3] += len(res)
    tot_ok = sum(v[0] for v in by_kind.values()); tot = sum(v[1] for v in by_kind.values())
    lines = [f"solver: {root}", f"originals solved: {len(solved)} / {len(tasks)}",
             f"variants solved: {tot_ok} / {tot} ({100 * tot_ok / max(1, tot):.0f}%)", "", "by variant kind:"]
    for k in ("rotation/flip", "colour", "holdout"):
        if k in by_kind:
            ok, m = by_kind[k]
            lines.append(f"  {k:<14} {ok:5d} / {m:<5d} {100 * ok / max(1, m):5.0f}%")
    lines += ["", "by family (tasks | tasks passing ALL their variants | variants solved):"]
    for f, (nt, na, vo, vm) in sorted(by_fam.items(), key=lambda kv: kv[1][2] / max(1, kv[1][3])):
        lines.append(f"  {f[:58]:<58} {nt:4d} | {na:4d} | {vo:4d}/{vm:<4d} {100 * vo / max(1, vm):4.0f}%")
    weak = sorted(((sum(r.values()) / len(r), tid) for tid, r in per_task.items() if not all(r.values())))
    lines += ["", f"tasks failing some variant: {len(weak)} (worst first; details in report.json)"]
    for frac, tid in weak[:40]:
        failed = [k for k, ok in per_task[tid].items() if not ok]
        lines.append(f"  {tid}  {100 * frac:3.0f}% solved  [{family.get(tid)}]  fails: {', '.join(failed[:8])}")
    txt = "\n".join(lines)
    open(os.path.join(a.out, "report.txt"), "w").write(txt + "\n")
    json.dump({"solved": sorted(solved), "family": family, "per_task": per_task,
               "by_kind": dict(by_kind)}, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(txt)
    print(f"\ntotal {time.time() - t0:.0f}s")


def run_all(root, vdir, pdir, jobs, a, extra):
    done = [0]
    t0 = time.time()

    def one(name):
        run_solver(root, os.path.join(vdir, name), os.path.join(pdir, name), a.timeout, extra)
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"  {done[0]}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
