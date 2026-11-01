"""LUCA hybrid - run a pre-trained Tiny Recursive Model (TRM) and write candidate answers.

Runs on Apple-silicon GPUs (MPS), NVIDIA GPUs (CUDA) or the CPU; no torchrun, no distributed setup.
It evaluates every augmented copy of every test input in a TRM-built dataset, maps each prediction
back through its augmentation, and votes. The output is the candidates file the hybrid reads:

    {"<task_id>": [ [ {"grid": [[...]], "score": votes + mean halting confidence}, ... ], ... ]}
                    one list per test input, in the task's test order, best first

Usage:
    python3 hybrid/trm_infer.py --trm-repo ~/TinyRecursiveModels --checkpoint ~/trm/step_XXXX \\
        --data ~/trm/arc2u-pretrain --out candidates.json [--device mps] [--batch-size 128]
        [--max-aug N] [--top-k 10] [--fp32] [--limit-batches N]

  --trm-repo     a clone of github.com/TrelisResearch/TinyRecursiveModels (model + dataset code)
  --checkpoint   a TRM checkpoint (e.g. from huggingface.co/Trelis/TRM-pretrain_arc2_rearc_200k)
  --data         the TRM-built dataset folder MATCHING the checkpoint (its tasks have embeddings in
                 the checkpoint); for NEW tasks use hybrid/ttt.py (test-time training) instead
  --arch         architecture yaml or full training config (default: all_config.yaml next to the
                 checkpoint if present, else <trm-repo>/config/arch/trm.yaml)
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trm_lib as T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trm-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-aug", type=int, default=None, help="limit augmented copies per test input")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--fp32", action="store_true", help="run in float32 (if bfloat16 is unsupported)")
    ap.add_argument("--halt-max-steps", type=int, default=None)
    ap.add_argument("--limit-batches", type=int, default=None, help="for a quick smoke test")
    a = ap.parse_args()

    T.setup(a.trm_repo)
    device = T.pick_device(a.device)
    arch_path = T.find_arch(a.checkpoint, a.trm_repo, a.arch)
    arch = T.load_arch(arch_path, a.fp32)
    print("settings from", arch_path)
    print("device", device, "| forward dtype", arch.get("forward_dtype"),
          "| hidden", arch.get("hidden_size"), "| H/L cycles", arch.get("H_cycles"), arch.get("L_cycles"))
    data_path = os.path.expanduser(a.data)
    _, meta = T.test_loader(data_path, a.batch_size, a.max_aug)
    sd = T.load_state(a.checkpoint)
    # the model is sized from the checkpoint: it may have been trained on several datasets at once
    # (e.g. rearc-pretrain + arc2u-pretrain), so its task-embedding table covers all of them
    k = T.emb_key(sd)
    n_ids = int(sd[k].shape[0]) if k else meta.num_puzzle_identifiers
    n_data = len(json.load(open(os.path.join(data_path, "identifiers.json"))))
    if n_data > n_ids:
        sys.exit(f"the dataset has {n_data} task identifiers but the checkpoint only {n_ids}: "
                 "--data does not belong to this checkpoint (for new tasks use hybrid/ttt.py)")
    model = T.build_model(arch, meta, a.batch_size, n_ids, device, a.halt_max_steps)
    missing, unexpected = T.load_into(model, sd)
    print("loaded checkpoint | missing", len(missing), "unexpected", len(unexpected))
    model.to(device)
    t0 = time.time()
    out = T.infer(model, data_path, device, a.batch_size, a.max_aug, a.top_k, a.limit_batches,
                  log=lambda m: print(m, flush=True))
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}: {len(out)} tasks, {sum(1 for v in out.values() if v and v[0])} with candidates, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
