"""LUCA + TRM hybrid - Kaggle submission entry point (offline notebook, ARC Prize format).

Paste into a Kaggle notebook cell:   !python /kaggle/input/<luca-hybrid-dataset>/LUCA_hybrid/hybrid/kaggle_main.py
(or `%run` it). Internet must be OFF. Attach three Kaggle datasets (see KAGGLE.md):
  * LUCA_hybrid      this kit (contains luca_v31/ and hybrid/)
  * trm-code         a clone of github.com/TrelisResearch/TinyRecursiveModels (contains puzzle_dataset.py)
  * trm-model        the checkpoint step_XXXX and its all_config.yaml (from Trelis/TRM-pretrain_arc2_rearc_200k)
  * (optional) wheels   pip wheels for packages missing from the Kaggle image (argdantic, coolname, ...)

Pipeline (all within the time budget):
  0. a valid submission.json is written at once (identity guesses), so a timeout never leaves nothing
  1. LUCA solves every task on the CPU, in the background
  2. TRM test-time training on the GPU: new task embeddings, embeddings-then-full training on the
     demonstrations, augmented prediction and voting (hybrid/ttt.py)
  3. hybrid combination (hybrid/solve_hybrid.py rules) -> /kaggle/working/submission.json
     {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...one per test input]}

Settings: environment variables LUCA_TOTAL_MINUTES (default 690 = 11.5 h), LUCA_SECONDS (LUCA per-task
budget, default 30), LUCA_WORKERS (default: CPU count), TTT_NUM_AUG (256), TTT_MAX_AUG (128),
TTT_BATCH (96, halved automatically on out-of-memory), TTT_INFER_BATCH (128), CHALLENGES (explicit path to the challenges JSON).
"""
import glob, json, os, subprocess, sys, time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
WORK = os.environ.get("LUCA_WORK", "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work"))
INPUT = os.environ.get("LUCA_INPUT", "/kaggle/input")
TOTAL_MIN = float(os.environ.get("LUCA_TOTAL_MINUTES", "690"))
END = T0 + TOTAL_MIN * 60


def log(m):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)


def find(pattern, want_dir=False):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", pattern), recursive=True))
    hits = [h for h in hits if os.path.isdir(h) == want_dir]
    return hits


def locate():
    ch = os.environ.get("CHALLENGES")
    if not ch:
        tests = find("*test_challenges.json")
        ch = tests[0] if tests else (find("*evaluation*challenges.json") or [None])[0]
    trm = os.environ.get("TRM_REPO") or next((os.path.dirname(p) for p in find("puzzle_dataset.py")), None)
    steps = [p for p in find("step_*") if not os.path.basename(p).startswith("evaluator")]
    ckpt = os.environ.get("TRM_CHECKPOINT") or (max(steps, key=lambda p: int(os.path.basename(p).split("_")[-1]))
                                                if steps else None)
    return ch, trm, ckpt


def install_wheels():
    wheel_dirs = sorted({os.path.dirname(p) for p in find("*.whl")})
    for d in wheel_dirs:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", d, "-q"] +
                       sorted({os.path.basename(p).split("-")[0] for p in glob.glob(os.path.join(d, "*.whl"))}),
                       check=False)


def write_submission(challenges, attempts_by_task, path):
    sub = {}
    for tid, task in challenges.items():
        per = []
        for i, t in enumerate(task["test"]):
            atts = (attempts_by_task.get(tid) or [])
            preds = atts[i] if i < len(atts) else []
            preds = [p for p in preds if p] or [t["input"]]
            if len(preds) < 2:
                preds = preds + [preds[0]]
            per.append({"attempt_1": preds[0], "attempt_2": preds[1]})
        sub[tid] = per
    json.dump(sub, open(path, "w"))
    return sub


def main():
    os.makedirs(WORK, exist_ok=True)
    sub_path = os.path.join(WORK, "submission.json")
    ch_path, trm_repo, ckpt = locate()
    log(f"challenges {ch_path}\n  TRM code {trm_repo}\n  checkpoint {ckpt}")
    if not ch_path:
        sys.exit("no challenges file found under " + INPUT)
    challenges = json.load(open(ch_path))
    write_submission(challenges, {}, sub_path)                       # 0. a valid file at once
    log(f"fallback submission written ({len(challenges)} tasks)")
    install_wheels()

    # 1. LUCA in the background (CPU)
    tasks_dir = os.path.join(WORK, "tasks"); luca_out = os.path.join(WORK, "luca_out")
    subprocess.run([sys.executable, os.path.join(HERE, "split_tasks.py"), ch_path, tasks_dir], check=True)
    empty = os.path.join(WORK, "none.json"); json.dump({}, open(empty, "w"))
    try:                       # leave one CPU core per GPU process (data loading) besides LUCA's workers
        import torch
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_gpu = 0
    workers = os.environ.get("LUCA_WORKERS") or str(max(1, (os.cpu_count() or 2) - 1 - n_gpu))
    luca = subprocess.Popen([sys.executable, os.path.join(HERE, "solve_hybrid.py"), tasks_dir, empty, luca_out,
                             "--seconds", os.environ.get("LUCA_SECONDS", "30"), "--workers", workers],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, cwd=KIT)
    log(f"LUCA started in the background ({workers} workers)")

    # 2. TRM test-time training (GPU), leaving 20 minutes for the combination
    cand_path = os.path.join(WORK, "candidates.json")
    cands = {}
    if trm_repo and ckpt:
        try:
            sys.path.insert(0, HERE)
            import ttt
            minutes = (END - time.time()) / 60 - 20
            log(f"GPUs: {ttt.gpu_count()} -> one task shard per GPU")
            cands = ttt.run_sharded(trm_repo, ckpt, ch_path, os.path.join(WORK, "ttt"), cand_path, minutes=minutes,
                                    num_aug=int(os.environ.get("TTT_NUM_AUG", "256")),
                                    max_aug=int(os.environ.get("TTT_MAX_AUG", "128")),
                                    batch_size=int(os.environ.get("TTT_BATCH", "96")),
                                    infer_batch=int(os.environ.get("TTT_INFER_BATCH", "128")), log=log)
        except Exception as e:                                         # LUCA alone still gives a submission
            import traceback
            log("TRM stage failed: " + repr(e)); traceback.print_exc()
    else:
        log("TRM code or checkpoint not found: LUCA only")
    if not cands:
        json.dump({}, open(cand_path, "w"))

    # 3. combine
    wait = max(60, END - time.time() - 300)
    try:
        luca.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        luca.kill(); log("LUCA stopped at the time budget (finished tasks are kept)")
    hyb_out = os.path.join(WORK, "hybrid_out")
    subprocess.run([sys.executable, os.path.join(HERE, "solve_hybrid.py"), tasks_dir, cand_path, hyb_out,
                    "--luca-dir", luca_out, "--seconds", "5", "--workers", workers], check=False, cwd=KIT,
                   stdout=subprocess.DEVNULL)
    attempts = {}
    for f in os.listdir(hyb_out) if os.path.isdir(hyb_out) else []:
        if f.endswith(".json"):
            attempts[f[:-5]] = [r.get("predictions", []) for r in json.load(open(os.path.join(hyb_out, f)))]
    write_submission(challenges, attempts, sub_path)
    srcs = {}
    for f in os.listdir(hyb_out) if os.path.isdir(hyb_out) else []:
        for r in json.load(open(os.path.join(hyb_out, f))):
            for s in r.get("sources", [])[:1]:
                srcs[s.split(":")[0]] = srcs.get(s.split(":")[0], 0) + 1
    log(f"submission.json written: {len(attempts)} tasks combined | attempt-1 sources {srcs}")


if __name__ == "__main__":
    main()
