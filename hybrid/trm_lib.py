"""Shared TRM helpers for the LUCA hybrid: model building, checkpoint loading, inference and voting.
Device-independent (CUDA, Apple MPS, CPU); no torchrun / distributed setup."""
import json, os, sys, time
from collections import defaultdict


def setup(trm_repo):
    sys.path.insert(0, os.path.abspath(os.path.expanduser(trm_repo)))
    os.environ.setdefault("DISABLE_COMPILE", "1")
    _stub_argdantic()


def _stub_argdantic():
    """TRM's code imports argdantic only for command-line parsing, which this hybrid never uses. Where
    argdantic (or its own dependencies) is unavailable - e.g. an offline Kaggle image - a minimal
    stand-in is registered so the imports succeed."""
    try:
        import argdantic  # noqa: F401
        return
    except Exception:
        pass
    import types

    class ArgParser:
        def __init__(self, *a, **k):
            pass

        def command(self, *a, **k):
            return lambda f: f

        def __call__(self, *a, **k):
            raise SystemExit("argdantic is not installed: call the dataset builder from Python instead")
    mod = types.ModuleType("argdantic")
    mod.ArgParser = ArgParser
    sys.modules["argdantic"] = mod


def pick_device(name="auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_arch(checkpoint, trm_repo, arch=None):
    """The training config shipped next to the checkpoint (all_config.yaml) when present."""
    if arch:
        return arch
    here = os.path.dirname(os.path.abspath(os.path.expanduser(checkpoint)))
    for cand in (os.path.join(here, "all_config.yaml"), os.path.join(os.path.dirname(here), "all_config.yaml")):
        if os.path.exists(cand):
            return cand
    return os.path.join(os.path.expanduser(trm_repo), "config", "arch", "trm.yaml")


def load_arch(path, fp32=False):
    import yaml
    cfg = yaml.safe_load(open(path))
    if isinstance(cfg.get("arch"), dict):
        cfg = cfg["arch"]
    hidden = cfg.get("hidden_size")
    for k, v in list(cfg.items()):
        if isinstance(v, str) and v.strip() == "${.hidden_size}":
            cfg[k] = hidden
    if fp32:
        cfg["forward_dtype"] = "float32"
    return cfg


def load_state(checkpoint):
    import torch
    sd = torch.load(os.path.expanduser(checkpoint), map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}


def emb_key(sd):
    for k in sd:
        if k.endswith("puzzle_emb.weights"):
            return k
    return None


def build_model(arch, meta, batch_size, n_ids, device, halt_max_steps=None):
    """The TRM model with its loss head (the head is used for training only)."""
    import torch
    from utils.functions import load_model_class
    arch = dict(arch)
    loss_cfg = dict(arch.pop("loss"))
    name = arch.pop("name")
    cfg = dict(arch, batch_size=batch_size, vocab_size=meta.vocab_size, seq_len=meta.seq_len,
               num_puzzle_identifiers=n_ids, causal=False)
    if halt_max_steps is not None:
        cfg["halt_max_steps_eval"] = halt_max_steps
    model_cls = load_model_class(name)
    loss_cls = load_model_class(loss_cfg.pop("name"))
    with torch.device(device):
        model = loss_cls(model_cls(cfg), **loss_cfg)
    return model


def load_into(model, sd, new_task_embeddings=False):
    """Load checkpoint weights. new_task_embeddings=True: the model's task-embedding table is for NEW
    tasks (a different dataset): every row starts at the mean of the checkpoint's learned rows
    (the TRM authors' 'mean' re-initialisation for post-training)."""
    import torch
    k = emb_key(sd)
    target = model.state_dict()
    tk = next((t for t in target if t.endswith("puzzle_emb.weights")), None)
    sd = dict(sd)
    if k is not None and tk is not None:
        if new_task_embeddings or sd[k].shape != target[tk].shape:
            if not new_task_embeddings:
                raise ValueError(f"task-embedding table {tuple(sd[k].shape)} != model {tuple(target[tk].shape)}")
            mean = sd[k].to(torch.float32).mean(0, keepdim=True)
            sd[k] = mean.expand(target[tk].shape).to(sd[k].dtype).contiguous()
        if k != tk:
            sd[tk] = sd.pop(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def float32_loss_for(device):
    """Apple GPUs have no float64: make TRM's training loss compute in float32 there."""
    if device.type != "mps":
        return
    import torch
    import models.losses as L

    def stablemax_cross_entropy(logits, labels, ignore_index=-100, valid_mask=None):
        logprobs = L.log_stablemax(logits.to(torch.float32), dim=-1)
        if valid_mask is None:
            valid_mask = (labels != ignore_index)
        tl = torch.where(valid_mask, labels, 0)
        pl = torch.gather(logprobs, index=tl.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
        return -torch.where(valid_mask, pl, 0)
    L.stablemax_cross_entropy = stablemax_cross_entropy


def test_loader(data_path, batch_size, max_aug=None):
    from torch.utils.data import DataLoader
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    ds = PuzzleDataset(PuzzleDatasetConfig(
        seed=0, dataset_paths=[data_path], global_batch_size=batch_size, test_set_mode=True,
        epochs_per_iter=1, rank=0, num_replicas=1, max_eval_augmentations=max_aug), split="test")
    return DataLoader(ds, batch_size=None, num_workers=0), ds.metadata


def vote_pass(model, data_path, device, batch_size, max_aug=None, limit_batches=None, deadline=None, state=None,
              skip_original=False, log=print):
    """One prediction pass: every (augmented) copy of every test input, mapped back and added to the
    votes in `state` (created if None). skip_original: only augmented copies (for a second pass)."""
    import torch
    from dataset.build_arc_dataset import inverse_aug, grid_hash
    from evaluators.arc import _crop
    loader, meta = test_loader(data_path, batch_size, max_aug)
    identifiers = json.load(open(os.path.join(data_path, "identifiers.json")))
    if state is None:
        state = {"hmap": {}, "votes": defaultdict(lambda: defaultdict(lambda: [0, 0.0]))}
    hmap, votes = state["hmap"], state["votes"]
    inner = getattr(model, "model", model)
    model.eval()
    t0 = time.time()
    with torch.inference_mode():
        for n, (set_name, batch, _) in enumerate(loader):
            if (limit_batches and n >= limit_batches) or (deadline and time.time() > deadline):
                log(f"stopping inference after {n} batches (limit / time budget)")
                break
            if skip_original:
                keep = [("|||" in identifiers[int(p)]) for p in batch["puzzle_identifiers"].tolist()]
                if not any(keep):
                    continue
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.device(device):
                carry = inner.initial_carry(batch)
            steps = 0
            while True:
                carry, outputs = inner(carry=carry, batch=batch)
                steps += 1
                if bool(carry.halted.all()):
                    break
            q = outputs["q_halt_logits"].to(torch.float32).sigmoid().cpu().numpy()
            preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
            ids = batch["puzzle_identifiers"].cpu().numpy()
            inps = batch["inputs"].cpu().numpy()
            for pid, inp, pred, qq in zip(ids, inps, preds, q):
                if pid == meta.blank_identifier_id:
                    continue
                name = identifiers[pid]
                if skip_original and "|||" not in name:
                    continue
                orig, inv = inverse_aug(name)
                g = inv(_crop(pred))
                if g.size == 0 or g.max() > 9:
                    continue
                ih = grid_hash(inv(_crop(inp)))
                h = grid_hash(g)
                hmap[h] = g
                v = votes[(orig, ih)][h]
                v[0] += 1; v[1] += float(qq)
            log(f"batch {n + 1} {set_name}: {steps} steps, {time.time() - t0:.0f}s")
    return state


def finalize(state, data_path, top_k=10):
    """Candidates JSON from the votes: {task_id: [[{grid, score}, ...] per test input]}, best first."""
    from dataset.build_arc_dataset import grid_hash, arc_grid_to_np
    test_puzzles = json.load(open(os.path.join(data_path, "test_puzzles.json")))
    hmap, votes = state["hmap"], state["votes"]
    out = {}
    for tid, puzzle in test_puzzles.items():
        per_test = []
        for pair in puzzle["test"]:
            ih = grid_hash(arc_grid_to_np(pair["input"]))
            cand = votes.get((tid, ih), {})
            ranked = sorted(cand.items(), key=lambda kv: (kv[1][0], kv[1][1] / kv[1][0]), reverse=True)[:top_k]
            per_test.append([{"grid": hmap[h].tolist(), "score": c + qs / c / 10.0} for h, (c, qs) in ranked])
        out[tid] = per_test
    return out


def infer(model, data_path, device, batch_size, max_aug=None, top_k=10, limit_batches=None, deadline=None,
          log=print):
    """Predict every augmented copy of every test input, map back through the augmentation, vote.
    Returns {task_id: [[{grid, score}, ...] per test input]} (best first)."""
    state = vote_pass(model, data_path, device, batch_size, max_aug, limit_batches, deadline, log=log)
    return finalize(state, data_path, top_k)
