"""LUCA hybrid - fair before/after test of TRM checkpoints on tasks NEITHER has seen.

Runs test-time training (hybrid/ttt.py) with each checkpoint, with the same settings and time
budget, on one challenges file, then scores the two attempts per test input against the
solutions. Prints totals only.

    python3 LUCA_hybrid/hybrid/ab_test.py --trm-repo trm-code \\
        --checkpoints trm-model/step_737320 trm-model-v2/step_XXXXXX \\
        --challenges LUCA_hybrid/data_v2/conceptarc_challenges.json --minutes 90 [--device auto]

The solutions file is found next to the challenges (..._solutions.json) and is read only for
scoring, after each checkpoint has finished.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def score(cands, solutions):
    tasks = right = inputs = right_inputs = 0
    for tid, sols in solutions.items():
        tasks += 1
        per = cands.get(tid, [])
        ok_all = True
        for i, y in enumerate(sols):
            inputs += 1
            top2 = [c["grid"] for c in (per[i] if i < len(per) else [])][:2]
            ok = y in top2
            right_inputs += ok
            ok_all &= ok
        right += ok_all
    return tasks, right, inputs, right_inputs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trm-repo", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--minutes", type=float, default=90.0, help="test-time training + prediction per checkpoint")
    ap.add_argument("--workdir", default="ab_work")
    ap.add_argument("--num-aug", type=int, default=64)
    ap.add_argument("--max-aug", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--infer-batch", type=int, default=128)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    import ttt
    sol_path = a.challenges.replace("_challenges.json", "_solutions.json")
    results = []
    for k, ck in enumerate(a.checkpoints):
        name = f"{k}_{os.path.basename(os.path.dirname(os.path.abspath(ck)))}_{os.path.basename(ck)}"
        wd = os.path.join(a.workdir, name); out = os.path.join(wd, "candidates.json")
        os.makedirs(wd, exist_ok=True)
        t0 = time.time()
        if not os.path.exists(out):
            print(f"=== {ck}: test-time training for {a.minutes:.0f} min", flush=True)
            ttt.run(a.trm_repo, ck, a.challenges, os.path.join(wd, "ttt"), out, minutes=a.minutes, num_aug=a.num_aug,
                    max_aug=a.max_aug, batch_size=a.batch_size, infer_batch=a.infer_batch, device_name=a.device,
                    log=lambda m: print("   ", m, flush=True))
        cands = json.load(open(out))
        tasks, right, inputs, right_inputs = score(cands, json.load(open(sol_path)))
        results.append((ck, right, tasks, right_inputs, inputs, time.time() - t0))
    print("\nRESULT (two attempts per test input; a task counts when all its test inputs are right)")
    for ck, right, tasks, ri, ni, sec in results:
        print(f"  {ck}: tasks {right}/{tasks} | test inputs {ri}/{ni} | {sec / 60:.0f} min")
    if len(results) == 2:
        d = results[1][1] - results[0][1]
        print(f"  change: {d:+d} tasks ({results[1][3] - results[0][3]:+d} test inputs)")


if __name__ == "__main__":
    main()
