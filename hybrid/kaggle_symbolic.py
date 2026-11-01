"""LUCA symbolic solver for Kaggle: LUCA (one language, one search) + the V160 family library.

CPU only - no GPU and no neural model, so the notebook can run with Accelerator: None.
Per test input the two attempts are filled in this order:
  1. LUCA's best exact answer (a program that reproduces every demonstration)
  2. V160's unique answer (a family fit that passed V160's own validation)
  3. LUCA's other exact answers, then LUCA's near misses, then the identity.
Both solvers abstain when nothing fits, so their answers are rarely wrong.

Notebook cell (internet off, accelerator None):
    !python $(find /kaggle/input -name kaggle_symbolic.py | head -1)
Settings (environment): LUCA_TOTAL_MINUTES (600), LUCA_SECONDS (30, LUCA per task),
V160_TIMEOUT (120 s per task), LUCA_WORKERS (CPU count - 1), CHALLENGES (explicit path).
"""
import glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
WORK = os.environ.get("LUCA_WORK", "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work"))
INPUT = os.environ.get("LUCA_INPUT", "/kaggle/input")
END = T0 + float(os.environ.get("LUCA_TOTAL_MINUTES", "600")) * 60
RUNNER = r'''
import sys, runpy
root, task, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
sys.argv = [root + "/run.py", task, "--output", out]
runpy.run_path(root + "/run.py", run_name="__main__")
'''


def log(m):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)


def find_challenges():
    ch = os.environ.get("CHALLENGES")
    if ch:
        return ch
    hits = sorted(glob.glob(os.path.join(INPUT, "**", "*test_challenges.json"), recursive=True))
    if not hits:
        hits = sorted(glob.glob(os.path.join(INPUT, "**", "*evaluation*challenges.json"), recursive=True))
    return hits[0] if hits else None


def v160_root():
    hits = sorted(glob.glob(os.path.join(KIT, "v160", "**", "run.py"), recursive=True), key=len)
    return os.path.dirname(hits[0]) if hits else None


def write_submission(challenges, attempts, path):
    sub = {}
    for tid, task in challenges.items():
        per = []
        for i, t in enumerate(task["test"]):
            preds = [p for p in ((attempts.get(tid) or [])[i] if i < len(attempts.get(tid) or []) else []) if p]
            preds = preds or [t["input"]]
            if len(preds) < 2:
                preds = preds + [preds[0]]
            per.append({"attempt_1": preds[0], "attempt_2": preds[1]})
        sub[tid] = per
    json.dump(sub, open(path, "w"))


def v160_answers(path):
    """V160 run.py output -> unique prediction per test input (None = abstained)."""
    try:
        r = json.load(open(path))
        return [t.get("prediction") if t.get("status") == "unique" else None for t in r.get("tests", [])]
    except Exception:
        return []


def combine(luca_res, v_preds, task):
    out = []
    for i, t in enumerate(task["test"]):
        r = luca_res[i] if i < len(luca_res) else {}
        pairs = list(zip(r.get("predictions", []), r.get("sources", [])))
        exact = [p for p, s in pairs if s.startswith("luca:dsl:")]
        other = [p for p, s in pairs if s.startswith("luca:") and not s.startswith("luca:dsl:")]
        v = v_preds[i] if i < len(v_preds) else None
        order = exact[:1] + ([v] if v else []) + exact[1:] + other + [t["input"]]
        att = []
        for g in order:
            if g and g not in att:
                att.append(g)
            if len(att) == 2:
                break
        out.append(att)
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    ch_path = find_challenges()
    log(f"challenges {ch_path}")
    if not ch_path:
        sys.exit("no challenges file found under " + INPUT)
    challenges = json.load(open(ch_path))
    sub_path = os.path.join(WORK, "submission.json")
    write_submission(challenges, {}, sub_path)
    log(f"fallback submission written ({len(challenges)} tasks)")
    tasks_dir = os.path.join(WORK, "tasks"); os.makedirs(tasks_dir, exist_ok=True)
    for tid, t in challenges.items():
        json.dump(t, open(os.path.join(tasks_dir, tid + ".json"), "w"))
    workers = int(os.environ.get("LUCA_WORKERS") or max(1, (os.cpu_count() or 2) - 1))

    # 1. LUCA (its own worker pool), no neural candidates
    empty = os.path.join(WORK, "none.json"); json.dump({}, open(empty, "w"))
    luca_out = os.path.join(WORK, "luca_out")
    luca = subprocess.Popen([sys.executable, os.path.join(HERE, "solve_hybrid.py"), tasks_dir, empty, luca_out,
                             "--seconds", os.environ.get("LUCA_SECONDS", "30"), "--workers", str(workers)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, cwd=KIT)
    log(f"LUCA started ({workers} workers)")
    try:
        luca.wait(timeout=max(60, (END - time.time()) * 0.6))
    except subprocess.TimeoutExpired:
        luca.kill(); log("LUCA stopped at its time share (finished tasks are kept)")
    log(f"LUCA done: {len(os.listdir(luca_out)) if os.path.isdir(luca_out) else 0} tasks")

    # 2. V160, one process per task
    root = v160_root()
    v_out = os.path.join(WORK, "v160_out"); os.makedirs(v_out, exist_ok=True)
    tmo = float(os.environ.get("V160_TIMEOUT", "120"))
    if root:
        def one(tid):
            o = os.path.join(v_out, tid + ".json")
            if os.path.exists(o) or time.time() > END - 300:
                return
            try:
                subprocess.run([sys.executable, "-I", "-c", RUNNER, root, os.path.join(tasks_dir, tid + ".json"), o],
                               capture_output=True, timeout=tmo, cwd=v_out)
            except subprocess.TimeoutExpired:
                pass
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, sorted(challenges)))
        log(f"V160 done ({root})")
    else:
        log("V160 not found in the kit: LUCA only")

    # 3. combine
    attempts, src = {}, {"luca": 0, "v160": 0, "none": 0}
    for tid, task in challenges.items():
        try:
            luca_res = json.load(open(os.path.join(luca_out, tid + ".json")))
        except Exception:
            luca_res = []
        vp = v160_answers(os.path.join(v_out, tid + ".json"))
        attempts[tid] = combine(luca_res, vp, task)
        has_l = any(s.startswith("luca:dsl:") for r in luca_res for s in r.get("sources", []))
        src["luca" if has_l else ("v160" if any(vp) else "none")] += 1
    write_submission(challenges, attempts, sub_path)
    log(f"submission.json written: {len(attempts)} tasks | tasks with an exact answer from {src}")


if __name__ == "__main__":
    main()
