"""LUCA hybrid - Kaggle notebook: continued pre-training of TRM on LUCA tasks + before/after test.

One Kaggle session (GPU L4 x4, internet off) does everything:
  * builds the training dataset once (LUCA tasks + ARC training tasks, data_v2/);
  * GPU 0 and GPU 2: two training variants side by side, same data, different network learning rate
    (variant a: 3e-5, careful; variant b: 1e-4, bolder), checkpoints every hour;
  * GPU 1, at the start: the ORIGINAL checkpoint's before-score on ConceptARC (test-time training);
  * at the end: each variant's final checkpoint takes the same ConceptARC test (GPU 0 and GPU 2);
  * prints the totals: original vs variant a vs variant b.
ConceptARC was used by neither the original training nor the task generator.

Paste in a notebook cell:
    !LUCA_TOTAL_MINUTES=690 python $(find /kaggle/input -name kaggle_pretrain.py | head -1)
Outputs (/kaggle/working): trm-model-v2a/, trm-model-v2b/ (step_N + all_config.yaml), ab/ (results).
Settings: LUCA_TOTAL_MINUTES (690), AB_MINUTES (90: test-time training per ConceptARC test),
PRETRAIN_BATCH (96), LR_A (3e-5), LR_B (1e-4), NUM_AUG (2).
"""
import glob, json, os, subprocess, sys, time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.environ.get("LUCA_INPUT", "/kaggle/input")
WORK = os.environ.get("LUCA_WORK", "/kaggle/working")
TOTAL = float(os.environ.get("LUCA_TOTAL_MINUTES", "690")) * 60
AB_MIN = float(os.environ.get("AB_MINUTES", "90"))
END = T0 + TOTAL
CPU_TEST = os.environ.get("PRETRAIN_CPU_TEST") == "1"      # local testing without GPUs


def log(m):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)


def find(pattern, want_dir=False):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", pattern), recursive=True))
    return [h for h in hits if os.path.isdir(h) == want_dir]


def gpu_env(i):
    env = dict(os.environ)
    if not CPU_TEST:
        env["CUDA_VISIBLE_DEVICES"] = str(i)
    return env


def dev():
    return "cpu" if CPU_TEST else "cuda"


def main():
    os.makedirs(WORK, exist_ok=True)
    trm = next((os.path.dirname(p) for p in find("puzzle_dataset.py")), None)
    steps = [p for p in find("step_*") if os.path.basename(p).split("_")[-1].isdigit()]
    ckpt = os.environ.get("TRM_CHECKPOINT") or (max(steps, key=lambda p: int(p.split("_")[-1])) if steps else None)
    data = {n: next(iter(find(n + ".json")), None) for n in ("composed", "variants", "arc_training",
                                                              "conceptarc_challenges", "conceptarc_solutions")}
    log(f"TRM code {trm}\n  checkpoint {ckpt}\n  data {json.dumps(data, indent=1)}")
    if not (trm and ckpt and all(data.values())):
        sys.exit("missing input: attach the TRM code, the TRM model and the luca-pretrain dataset")
    for d in sorted({os.path.dirname(p) for p in find("*.whl")}):      # optional wheels
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", d, "-q"] +
                       sorted({os.path.basename(p).split("-")[0] for p in glob.glob(os.path.join(d, "*.whl"))}),
                       check=False, capture_output=True)
    py = sys.executable
    pre = os.path.join(HERE, "pretrain_more.py"); ttt = os.path.join(HERE, "ttt.py")
    tasks = [data["composed"], data["variants"], data["arc_training"]]
    wd = os.path.join(WORK, "pretrain_work")
    ab = os.path.join(WORK, "ab"); os.makedirs(ab, exist_ok=True)
    batch = os.environ.get("PRETRAIN_BATCH", "96"); num_aug = os.environ.get("NUM_AUG", "2")
    extra = ["--max-tasks", os.environ["MAX_TASKS"]] if os.environ.get("MAX_TASKS") else []

    # 1. build the dataset once (both variants read it)
    log("building the training dataset")
    subprocess.run([py, pre, "--trm-repo", trm, "--checkpoint", ckpt, "--tasks", *tasks, "--workdir", wd,
                    "--out-dir", os.path.join(WORK, "unused"), "--hours", "0", "--num-aug", num_aug,
                    "--device", "cpu", *extra], check=True)
    import shutil; shutil.rmtree(os.path.join(WORK, "unused"), ignore_errors=True)

    def ab_cmd(ck, name, gpu, minutes):
        out = os.path.join(ab, name)
        os.makedirs(out, exist_ok=True)
        cmd = [py, ttt, "--trm-repo", trm, "--checkpoint", ck, "--challenges", data["conceptarc_challenges"],
               "--workdir", os.path.join(out, "work"), "--out", os.path.join(out, "candidates.json"),
               "--minutes", f"{minutes:.1f}", "--num-aug", os.environ.get("AB_NUM_AUG", "64"),
               "--max-aug", os.environ.get("AB_MAX_AUG", "32"), "--batch-size", os.environ.get("AB_BATCH", "96"),
               "--infer-batch", os.environ.get("AB_INFER_BATCH", "128"), "--device", dev()]
        return subprocess.Popen(cmd, env=gpu_env(gpu), stdout=open(os.path.join(out, "log.txt"), "w"),
                                stderr=subprocess.STDOUT)

    # 2. the original checkpoint's before-score (GPU 1), while training runs
    procs = {"original": ab_cmd(ckpt, "original", 1, AB_MIN)}
    log(f"ConceptARC test of the original checkpoint started on GPU 1 ({AB_MIN:.0f} min)")

    # 3. two training variants (GPU 0 and GPU 2), until there is time left for their own tests
    train_hours = max(0.05, (END - time.time() - (AB_MIN + 15) * 60) / 3600)
    trainers = {}
    for name, gpu, lr in (("v2a", 0, os.environ.get("LR_A", "3e-5")), ("v2b", 2, os.environ.get("LR_B", "1e-4"))):
        out = os.path.join(WORK, f"trm-model-{name}")
        cmd = [py, pre, "--trm-repo", trm, "--checkpoint", ckpt, "--tasks", *tasks, "--workdir", wd,
               "--out-dir", out, "--hours", f"{train_hours:.3f}", "--lr", lr, "--batch-size", batch,
               "--num-aug", num_aug, "--device", dev(), *extra]
        trainers[name] = (subprocess.Popen(cmd, env=gpu_env(gpu), stdout=open(os.path.join(WORK, f"train_{name}.log"), "w"),
                                           stderr=subprocess.STDOUT), gpu, out)
        log(f"training {name} (network learning rate {lr}) on GPU {gpu} for {train_hours:.1f} h -> {out}")

    last = 0
    while any(p.poll() is None for p, _, _ in trainers.values()):
        time.sleep(10)
        if time.time() - last > 900:
            last = time.time()
            for name in trainers:
                try:
                    tail = [l for l in open(os.path.join(WORK, f"train_{name}.log")).read().splitlines() if l.strip()][-1:]
                except Exception:
                    tail = []
                log(f"{name}: {tail[0][:170] if tail else ''}")
        if time.time() > END - (AB_MIN + 5) * 60:
            for name, (p, _, _) in trainers.items():
                if p.poll() is None:
                    p.kill(); log(f"{name}: stopped for its test (the hourly checkpoints are kept)")
            break

    # 4. each variant's latest checkpoint takes the same test
    left = (END - time.time()) / 60 - 8
    for name, (p, gpu, out) in trainers.items():
        cks = [c for c in glob.glob(os.path.join(out, "step_*")) if c.split("_")[-1].isdigit()]
        if not cks:
            log(f"{name}: no checkpoint saved - training failed, see train_{name}.log")
            continue
        ck = max(cks, key=lambda c: int(c.split("_")[-1]))
        procs[name] = ab_cmd(ck, name, gpu, min(AB_MIN, left))
        log(f"ConceptARC test of {name} ({os.path.basename(ck)}) started on GPU {gpu}")
    for name, p in procs.items():
        try:
            p.wait(timeout=max(60, END - time.time() - 120))
        except subprocess.TimeoutExpired:
            p.kill(); log(f"{name}: test stopped at the time limit")

    # 5. totals
    sys.path.insert(0, HERE)
    from ab_test import score
    sols = json.load(open(data["conceptarc_solutions"]))
    lines = []
    for name in ("original", "v2a", "v2b"):
        f = os.path.join(ab, name, "candidates.json")
        if os.path.exists(f):
            tasks_n, right, ni, ri = score(json.load(open(f)), sols)
            lines.append(f"{name:<9} ConceptARC tasks {right}/{tasks_n} | test inputs {ri}/{ni}")
        else:
            lines.append(f"{name:<9} no result (see ab/{name}/log.txt)")
    open(os.path.join(ab, "RESULT.txt"), "w").write("\n".join(lines) + "\n")
    log("RESULT (two attempts per test input)\n  " + "\n  ".join(lines))


if __name__ == "__main__":
    main()
