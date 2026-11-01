"""LUCA hybrid - continue pre-training the TRM checkpoint on LUCA-generated tasks.

The pre-trained TRM learns a hidden task at test time from 2-4 demonstrations. This script gives
the network more practice first: it continues training the checkpoint on synthetic tasks LUCA
generated (new compositions of LUCA operations + variants of solved training tasks), mixed with
the ARC training tasks so the network keeps what it knew. Every task gets a fresh task embedding
(the mean of the checkpoint's rows, as in test-time training); the network weights carry over.

Within the time budget it saves a checkpoint every --save-every minutes:
    OUT_DIR/step_<N>  +  OUT_DIR/all_config.yaml      (N = the checkpoint's step + steps trained)
Each saved file can be used exactly like the original checkpoint (ttt.py, kaggle_main.py).

Usage (Mac, from the folder that holds LUCA_hybrid):
    python3 LUCA_hybrid/hybrid/pretrain_more.py --trm-repo trm-code --checkpoint trm-model/step_737320 \\
        --tasks data_v2/composed.json data_v2/variants.json data_v2/arc_training.json \\
        --out-dir trm-model-v2 --hours 8 [--num-aug 2] [--batch-size 96] [--lr 3e-5] [--device auto]
"""
import argparse, json, os, random, shutil, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trm_lib as T


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def _fits(g):
    return 0 < len(g) <= 30 and 0 < len(g[0]) <= 30 and all(len(r) == len(g[0]) for r in g)


def merge_tasks(paths, out_prefix, max_tasks=None, seed=0):
    """All task files -> one challenges/solutions pair for TRM's builder. Every known pair becomes a
    training example; the last pair is also kept as the (unused) test example."""
    rng = random.Random(seed)
    ch, sol = {}, {}
    skipped = 0
    for p in paths:
        d = json.load(open(p))
        items = list(d.items())
        for tid, t in items:
            pairs = [e for e in t["train"] + t.get("test", []) if "output" in e]
            if len(pairs) < 2 or not all(_fits(e["input"]) and _fits(e["output"]) for e in pairs):
                skipped += 1
                continue
            key = f"{os.path.basename(p).split('.')[0]}_{tid}"
            ch[key] = {"train": [{"input": e["input"], "output": e["output"]} for e in pairs],
                       "test": [{"input": pairs[-1]["input"]}]}
            sol[key] = [pairs[-1]["output"]]
    keys = sorted(ch)
    rng.shuffle(keys)
    if max_tasks:
        keys = keys[:max_tasks]
    if skipped:
        log(f"skipped {skipped} tasks with a grid larger than 30x30 or fewer than 2 pairs")
    json.dump({k: ch[k] for k in keys}, open(out_prefix + "_challenges.json", "w"))
    json.dump({k: sol[k] for k in keys}, open(out_prefix + "_solutions.json", "w"))
    return len(keys)


def build(trm_repo, task_paths, workdir, num_aug, max_tasks):
    src = os.path.join(workdir, "src"); out = os.path.join(workdir, "data")
    if os.path.exists(os.path.join(out, "identifiers.json")):
        log("dataset already built: " + out)
        return out
    os.makedirs(src, exist_ok=True)
    n = merge_tasks(task_paths, os.path.join(src, "arc-agi_pre"), max_tasks)
    log(f"{n} tasks -> TRM dataset ({num_aug} augmented copies each) -> {out}")
    from dataset.build_arc_dataset import DataProcessConfig, convert_dataset
    convert_dataset(DataProcessConfig(input_file_prefix=os.path.join(src, "arc-agi"), output_dir=out,
                                      subsets=["pre"], test_set_names=["pre"], num_aug=num_aug))
    return out


def save(model, out_dir, step0, steps, arch_src):
    """Network weights + one task-embedding row (the mean): what test-time training needs."""
    import torch
    sd = {}
    for k, v in model.state_dict().items():
        if "puzzle_emb.local" in k:
            continue                                   # per-batch buffers of the sparse embedding
        if k.endswith("puzzle_emb.weights"):
            v = v.to(torch.float32).mean(0, keepdim=True)
        sd[k] = v.detach().cpu().clone()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step_{step0 + steps}")
    torch.save(sd, path + ".tmp"); os.replace(path + ".tmp", path)
    if not os.path.exists(os.path.join(out_dir, "all_config.yaml")):
        shutil.copy(arch_src, os.path.join(out_dir, "all_config.yaml"))
    log(f"saved {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trm-repo", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tasks", nargs="+", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workdir", default="pretrain_work")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--save-every", type=float, default=60.0, help="minutes between checkpoints")
    ap.add_argument("--num-aug", type=int, default=2)
    ap.add_argument("--max-tasks", type=int, default=None, help="use only this many tasks (smoke test)")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-5, help="network learning rate (low: keep what it knew)")
    ap.add_argument("--emb-lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--device", default="auto"); ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    T.setup(a.trm_repo)
    import torch
    from torch.utils.data import DataLoader
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
    t_end = time.time() + a.hours * 3600
    device = T.pick_device(a.device)
    T.float32_loss_for(device)
    data = build(a.trm_repo, a.tasks, a.workdir, a.num_aug, a.max_tasks)
    arch_path = T.find_arch(a.checkpoint, a.trm_repo)
    arch = T.load_arch(arch_path, a.fp32 or device.type == "cpu")
    step0 = int(os.path.basename(a.checkpoint).split("_")[-1]) if os.path.basename(a.checkpoint).split("_")[-1].isdigit() else 0
    sd = T.load_state(a.checkpoint)
    ek = T.emb_key(sd)
    if ek is not None:                                      # keep only the mean row (memory)
        sd[ek] = sd[ek].to(torch.float32).mean(0, keepdim=True).to(sd[ek].dtype)
    bs = a.batch_size
    while True:
        ds = PuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=[data], global_batch_size=bs, test_set_mode=False,
                                               epochs_per_iter=1, rank=0, num_replicas=1), split="train")
        meta = ds.metadata
        model = T.build_model(arch, meta, bs, meta.num_puzzle_identifiers, device)
        missing, unexpected = T.load_into(model, sd, new_task_embeddings=True)
        model.to(device)
        log(f"device {device} | {meta.total_puzzles} training puzzles | task embeddings {meta.num_puzzle_identifiers} "
            f"| batch {bs} | missing {len(missing)} unexpected {len(unexpected)}")
        loader = DataLoader(ds, batch_size=None, num_workers=0)
        emb_opt = CastedSparseEmbeddingSignSGD_Distributed(model.model.puzzle_emb.buffers(), lr=0, weight_decay=0.1,
                                                           world_size=1)
        opt = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=0.1, betas=(0.9, 0.95))
        model.train()
        carry, step, t0, last_save, ema = None, 0, time.time(), time.time(), None
        try:
            while time.time() < t_end:
                for _, batch, gbs in loader:
                    if time.time() >= t_end:
                        break
                    step += 1
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if carry is None:
                        with torch.device(device):
                            carry = model.initial_carry(batch)
                    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
                    (loss / gbs).backward()
                    scale = min(1.0, step / max(1, a.warmup))
                    for g in emb_opt.param_groups:
                        g["lr"] = a.emb_lr * scale
                    for g in opt.param_groups:
                        g["lr"] = a.lr * scale
                    emb_opt.step(); emb_opt.zero_grad()
                    opt.step(); opt.zero_grad()
                    cnt = max(1, int(metrics.get("count", torch.tensor(1)).item()))
                    acc = float(metrics.get("exact_accuracy", torch.tensor(0)).item()) / cnt
                    ema = acc if ema is None else 0.98 * ema + 0.02 * acc
                    if step % 100 == 0 or step == 1:
                        rate = step / max(1e-6, time.time() - t0)
                        log(f"step {step} loss {float(loss.item()) / gbs:.4f} exact-acc {acc:.3f} (avg {ema:.3f}) "
                            f"| {rate:.2f} steps/s | {(t_end - time.time()) / 3600:.1f} h left")
                    if time.time() - last_save > a.save_every * 60:
                        save(model, a.out_dir, step0, step, arch_path); last_save = time.time()
            break
        except (RuntimeError, MemoryError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(e, MemoryError) or bs <= 4:
                raise
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            bs //= 2
            log(f"out of memory: retrying with batch {bs}")
    path = save(model, a.out_dir, step0, step, arch_path)
    log(f"done: {step} steps. Final checkpoint: {path}")


if __name__ == "__main__":
    main()
