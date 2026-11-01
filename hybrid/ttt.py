"""LUCA hybrid - test-time training (TTT) of a pre-trained TRM on NEW tasks, then candidate answers.

For tasks the checkpoint has never seen (e.g. Kaggle's hidden test set) TRM has no task embedding.
This script, following the TRM authors' post-training recipe, within a time budget:
  1. builds TRM's augmented dataset from a challenges JSON (unknown test outputs become placeholders);
  2. loads the pre-trained network; every new task gets a task-embedding row initialised to the mean
     of the checkpoint's learned rows;
  3. trains on the tasks' demonstration pairs: first the task embeddings only (sign-SGD, as in TRM),
     then the whole network as well (AdamW);
  4. predicts every augmented copy of every test input and votes -> candidates JSON (as trm_infer.py).

Usage:
    python3 hybrid/ttt.py --trm-repo TRM_CODE --checkpoint CKPT --challenges CHALLENGES.json \\
        --workdir WORK --out candidates.json [--minutes 240] [--infer-share 0.35] [--embed-share 0.25]
        [--num-aug 256] [--max-aug 128] [--batch-size 96] [--infer-batch 128] [--device auto]
"""
import argparse, json, os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trm_lib as T


def build_dataset(trm_repo, challenges, workdir, num_aug, log=print):
    """TRM's dataset builder on one challenges file -> <workdir>/data (train = demos, test = test inputs)."""
    src = os.path.join(workdir, "src")
    os.makedirs(src, exist_ok=True)
    shutil.copy(challenges, os.path.join(src, "arc-agi_ttt_challenges.json"))
    sol = challenges.replace("_challenges.json", "_solutions.json")
    if os.environ.get("TTT_USE_SOLUTIONS") == "1" and os.path.exists(sol):
        shutil.copy(sol, os.path.join(src, "arc-agi_ttt_solutions.json"))   # only for local scoring runs
    out = os.path.join(workdir, "data")
    if os.path.exists(os.path.join(out, "identifiers.json")):
        log("dataset already built: " + out)
        return out
    log(f"building dataset (TRM builder, {num_aug} augmented copies per task) -> {out}")
    from dataset.build_arc_dataset import DataProcessConfig, convert_dataset   # called directly: no CLI
    convert_dataset(DataProcessConfig(input_file_prefix=os.path.join(src, "arc-agi"), output_dir=out,
                                      subsets=["ttt"], test_set_names=["ttt"], num_aug=num_aug))
    return out


def train(model, data_path, device, batch_size, deadline, embed_until, lr=1e-4, emb_lr=1e-2, warmup=200, log=print):
    """Time-budgeted TRM training on the demonstration pairs."""
    import torch
    from torch.utils.data import DataLoader
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
    ds = PuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=[data_path], global_batch_size=batch_size,
                                           test_set_mode=False, epochs_per_iter=1, rank=0, num_replicas=1),
                       split="train")
    loader = DataLoader(ds, batch_size=None, num_workers=0)
    inner = model.model
    emb_opt = CastedSparseEmbeddingSignSGD_Distributed(inner.puzzle_emb.buffers(), lr=0, weight_decay=0.1,
                                                       world_size=1)
    trunk_opt = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=0.1, betas=(0.9, 0.95))
    model.train()
    carry, step, t0 = None, 0, time.time()
    while time.time() < deadline:
        for _, batch, gbs in loader:
            if time.time() >= deadline:
                break
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            if carry is None:
                with torch.device(device):
                    carry = model.initial_carry(batch)
            carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
            (loss / gbs).backward()
            scale = min(1.0, step / max(1, warmup))
            for g in emb_opt.param_groups:
                g["lr"] = emb_lr * scale
            emb_opt.step(); emb_opt.zero_grad()
            if time.time() >= embed_until:            # phase 2: the whole network learns too
                for g in trunk_opt.param_groups:
                    g["lr"] = lr * scale
                trunk_opt.step()
            trunk_opt.zero_grad()
            if step % 25 == 0 or step == 1:
                cnt = max(1, int(metrics.get("count", torch.tensor(1)).item()))
                acc = float(metrics.get("exact_accuracy", torch.tensor(0)).item()) / cnt
                phase = "embeddings" if time.time() < embed_until else "full"
                log(f"train step {step} [{phase}] loss {float(loss.item()) / gbs:.4f} exact-acc {acc:.3f} "
                    f"{time.time() - t0:.0f}s")
    log(f"training done: {step} steps in {time.time() - t0:.0f}s")
    return step


def run(trm_repo, checkpoint, challenges, workdir, out_path, minutes=240.0, infer_share=0.35, embed_share=0.25,
        num_aug=256, max_aug=128, batch_size=96, infer_batch=128, device_name="auto", fp32=False, log=print,
        halt_max_steps=None):
    T.setup(trm_repo)
    import torch
    t_start = time.time()
    end = t_start + minutes * 60
    os.makedirs(workdir, exist_ok=True)
    device = T.pick_device(device_name)
    T.float32_loss_for(device)
    data = build_dataset(trm_repo, challenges, workdir, num_aug, log)
    arch_path = T.find_arch(checkpoint, trm_repo)
    arch = T.load_arch(arch_path, fp32 or device.type == "cpu")
    log(f"device {device} | settings {arch_path}")
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    meta = PuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=[data], global_batch_size=batch_size,
                                             test_set_mode=False, epochs_per_iter=1, rank=0, num_replicas=1),
                         split="train").metadata
    sd = T.load_state(checkpoint)
    # the new tasks only need the MEAN of the checkpoint's task-embedding table: keep that one row and
    # free the table (large for checkpoints pre-trained on many tasks; matters with one process per GPU)
    ek = T.emb_key(sd)
    if ek is not None:
        sd[ek] = sd[ek].to(torch.float32).mean(0, keepdim=True).to(sd[ek].dtype)
        import gc; gc.collect()
    # time budget: training gets (1 - infer_share) of what is left, the first embed_share of it for the
    # task embeddings alone
    left = end - time.time()
    train_end = time.time() + left * (1 - infer_share)
    embed_until = time.time() + (train_end - time.time()) * embed_share
    while True:
        model = T.build_model(arch, meta, batch_size, meta.num_puzzle_identifiers, device)
        missing, unexpected = T.load_into(model, sd, new_task_embeddings=True)
        log(f"loaded trunk | missing {len(missing)} unexpected {len(unexpected)} | new tasks "
            f"{meta.num_puzzle_identifiers} | training batch {batch_size}")
        model.to(device)
        try:
            train(model, data, device, batch_size, train_end, embed_until, log=log)
            break
        except (RuntimeError, MemoryError) as e:          # too large a batch for this GPU: halve it
            if "out of memory" not in str(e).lower() and not isinstance(e, MemoryError) or batch_size <= 2:
                raise
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            batch_size //= 2
            log(f"out of memory: retrying with training batch {batch_size}")
    # inference uses its own batch size: rebuild the carry-free model view with the trained weights
    trained = {k: v for k, v in model.state_dict().items()}
    model = T.build_model(arch, meta, infer_batch, meta.num_puzzle_identifiers, device, halt_max_steps)
    model.load_state_dict(trained, strict=False)
    model.to(device)
    # pass 1 (always completes): every test input once, without augmentation - every task gets an
    # answer; it also measures the speed. Pass 2: augmented copies for the time that is left.
    n_inputs = sum(len(p["test"]) for p in json.load(open(os.path.join(data, "test_puzzles.json"))).values())
    t1 = time.time()
    state = T.vote_pass(model, data, device, infer_batch, 0, log=log)
    per_item = (time.time() - t1) / max(1, n_inputs)
    budget = max(0.0, (end - 60 - time.time()) * 0.9)
    affordable = int(budget / max(1e-6, per_item * n_inputs))
    use_aug = max(0, min(max_aug, affordable))
    log(f"pass 1 done: {per_item:.2f}s per input, {budget / 60:.1f} min left -> pass 2 with {use_aug} augmented "
        f"copies per input (requested {max_aug})")
    if use_aug > 0:
        T.vote_pass(model, data, device, infer_batch, use_aug, deadline=end - 60, state=state, skip_original=True,
                    log=log)
    cands = T.finalize(state, data)
    json.dump(cands, open(out_path, "w"))
    log(f"wrote {out_path}: {len(cands)} tasks, {sum(1 for v in cands.values() if v and v[0])} with candidates, "
        f"total {time.time() - t_start:.0f}s")
    return cands


def gpu_count():
    forced = os.environ.get("TTT_SHARDS")
    if forced:
        return max(1, int(forced))
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    except Exception:
        return 1


def run_sharded(trm_repo, checkpoint, challenges, workdir, out_path, minutes=240.0, shards=None, log=print,
                **kw):
    """Multi-GPU test-time training: the tasks are split into one shard per GPU and each GPU trains and
    predicts its own shard in a separate process for the whole time budget (no communication between
    GPUs, so a failing shard costs only its own tasks). With one GPU this is run() itself."""
    shards = shards or gpu_count()
    if shards <= 1:
        return run(trm_repo, checkpoint, challenges, workdir, out_path, minutes=minutes, log=log, **kw)
    t_start = time.time()
    deadline = t_start + minutes * 60
    tasks = json.load(open(challenges))
    # balance the shards by the number of test inputs (prediction work) and demonstrations (training work)
    load = [0.0] * shards
    parts = [dict() for _ in range(shards)]
    for tid in sorted(tasks, key=lambda t: -(len(tasks[t]["train"]) + 2 * len(tasks[t]["test"]))):
        i = min(range(shards), key=lambda j: load[j])
        parts[i][tid] = tasks[tid]
        load[i] += len(tasks[tid]["train"]) + 2 * len(tasks[tid]["test"])
    os.makedirs(workdir, exist_ok=True)
    procs = []
    for i, part in enumerate(parts):
        if not part:
            continue
        sdir = os.path.join(workdir, f"shard{i}")
        os.makedirs(sdir, exist_ok=True)
        ch = os.path.join(sdir, "shard_challenges.json")
        json.dump(part, open(ch, "w"))
        out = os.path.join(sdir, "candidates.json")
        env = dict(os.environ)
        cmd = [sys.executable, os.path.abspath(__file__), "--trm-repo", trm_repo, "--checkpoint", checkpoint,
               "--challenges", ch, "--workdir", os.path.join(sdir, "work"), "--out", out,
               "--minutes", f"{max(1.0, (deadline - time.time()) / 60 - 2):.2f}"]
        for k, flag in (("infer_share", "--infer-share"), ("embed_share", "--embed-share"), ("num_aug", "--num-aug"),
                        ("max_aug", "--max-aug"), ("batch_size", "--batch-size"), ("infer_batch", "--infer-batch"),
                        ("halt_max_steps", "--halt-max-steps")):
            if kw.get(k) is not None:
                cmd += [flag, str(kw[k])]
        if os.environ.get("TTT_SHARDS"):        # forced shard count (testing without GPUs): keep the device
            cmd += ["--device", kw.get("device_name", "auto")]
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(i)
            cmd += ["--device", "cuda"]
        logf = open(os.path.join(sdir, "log.txt"), "w")
        procs.append((i, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), out, logf, len(part)))
        log(f"shard {i}: {len(part)} tasks -> GPU {i} (log {os.path.join(sdir, 'log.txt')})")
    # wait, echoing each shard's latest log line now and then; stop a shard that overruns the budget
    last = 0.0
    while any(p.poll() is None for _, p, _, _, _ in procs):
        time.sleep(5)
        if time.time() > deadline + 60:
            for i, p, _, _, _ in procs:
                if p.poll() is None:
                    p.kill(); log(f"shard {i} overran the time budget: stopped")
            break
        if time.time() - last > 600:
            last = time.time()
            for i, p, _, logf, _ in procs:
                try:
                    tail = open(logf.name).read().strip().splitlines()[-1:]
                except Exception:
                    tail = []
                log(f"shard {i} {'running' if p.poll() is None else 'done'}: {tail[0][:160] if tail else ''}")
    cands = {}
    for i, p, out, logf, n in procs:
        logf.close()
        if os.path.exists(out):
            part = json.load(open(out))
            cands.update(part)
            log(f"shard {i}: {sum(1 for v in part.values() if v and v[0])}/{n} tasks with candidates "
                f"(exit {p.returncode})")
        else:
            tail = open(logf.name).read()[-1500:]
            log(f"shard {i} FAILED (exit {p.returncode}); its tasks keep LUCA's answers. Log tail:\n{tail}")
    json.dump(cands, open(out_path, "w"))
    log(f"wrote {out_path}: {len(cands)} tasks from {len(procs)} shards, total {time.time() - t_start:.0f}s")
    return cands


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trm-repo", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--challenges", required=True); ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=240.0)
    ap.add_argument("--infer-share", type=float, default=0.35)
    ap.add_argument("--embed-share", type=float, default=0.25)
    ap.add_argument("--num-aug", type=int, default=256)
    ap.add_argument("--max-aug", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=96, help="training batch (halved automatically on OOM)")
    ap.add_argument("--infer-batch", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--halt-max-steps", type=int, default=None, help="cap reasoning steps (testing only)")
    a = ap.parse_args()
    run(a.trm_repo, a.checkpoint, a.challenges, a.workdir, a.out, a.minutes, a.infer_share, a.embed_share,
        a.num_aug, a.max_aug, a.batch_size, a.infer_batch, a.device, a.fp32, log=lambda m: print(m, flush=True),
        halt_max_steps=a.halt_max_steps)


if __name__ == "__main__":
    main()
