# LUCA + TRM hybrid on a Mac (Apple silicon)

The hybrid has three parts:

1. **LUCA**, the symbolic solver. Its exact answers (programs that reproduce every demonstration)
   come first.
2. **TRM**, a pre-trained Tiny Recursive Model (about 7M parameters). It proposes answers for the
   tasks LUCA cannot solve.
3. **The orchestrator** (`hybrid/solve_hybrid.py`). It rejects TRM answers that contradict the
   demonstrations (wrong output size, impossible colours) and fills the two attempts per test input.

Everything below runs on your Mac. The TRM model runs on the Apple GPU through PyTorch's `mps` backend.

## 1. One-time setup (Terminal)

```bash
cd ~ && mkdir -p luca-hybrid && cd luca-hybrid
# unzip LUCA_hybrid_kit.zip here, so that ~/luca-hybrid/LUCA_hybrid exists
python3 -m venv venv && source venv/bin/activate
pip install torch numpy einops numba pydantic argdantic pyyaml omegaconf coolname "huggingface_hub[cli]"
git clone https://github.com/TrelisResearch/TinyRecursiveModels trm-code
```

## 2. Download the pre-trained model

The model is Trelis TRM, 200k epochs, MIT licence. It was trained on ARC-AGI-1 and ARC-AGI-2
training data, ConceptARC and RE-ARC, plus the demonstration pairs (not the answers) of the 120
ARC-AGI-2 evaluation tasks. Its authors report about 15% on the ARC-AGI-2 public evaluation set.

```bash
huggingface-cli download Trelis/TRM-pretrain_arc2_rearc_200k --local-dir trm-model
find trm-model -name identifiers.json          # the dataset folder(s) built for this checkpoint
find trm-model -name "step_*" -o -name "*.pt" -o -name "*.ckpt" | head   # the checkpoint file(s)
```

You need two paths from this:
* **CHECKPOINT**: the checkpoint file. Take the latest step if there are several.
* **DATA**: the folder that contains `identifiers.json` and `test_puzzles.json` for the
  evaluation set. Its name is normally `arc2u-pretrain` or similar. TRM learns one embedding per
  task, so this must be the dataset built *for this checkpoint*. A dataset you build yourself will
  not match it.

If the download does not include the built dataset, build it with the same command the authors
used (from `trm-code`):

```bash
cd trm-code
find kaggle/combined -name '*.json.gz' -print0 | xargs -0 gunzip -f
python3 -m dataset.build_arc_dataset --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2u-pretrain --puzzle-identifiers-start 1597 \
  --subsets concept training2u evaluation2 --test-set-name evaluation2
cd ..
```

## 3. Smoke test (a few minutes)

```bash
python3 LUCA_hybrid/hybrid/trm_infer.py --trm-repo trm-code --checkpoint CHECKPOINT --data DATA \
  --out cand_smoke.json --device mps --batch-size 128 --limit-batches 3
```

* It should print `loaded checkpoint | missing 0 unexpected 0` and three batch lines, each with
  the time taken.
* If you get an error that mentions bfloat16, add `--fp32`.
* The time per batch tells you how long the full run will take (step 4).

## 4. Run TRM on the 120 ARC-AGI-2 evaluation tasks

The dataset holds many augmented copies (rotations and colour permutations) of each test input,
and TRM votes across them. More copies take longer and give better votes. Choose a copy count with
`--max-aug`, based on the time per batch from step 3:

```bash
python3 LUCA_hybrid/hybrid/trm_infer.py --trm-repo trm-code --checkpoint CHECKPOINT --data DATA \
  --out candidates.json --device mps --batch-size 128 --max-aug 64
```

Start with `--max-aug 64`. Use more copies if time allows, fewer if a run takes too long. You can
leave it running overnight.

## 5. Combine with LUCA and score

```bash
python3 LUCA_hybrid/hybrid/split_tasks.py trm-code/kaggle/combined/arc-agi_evaluation2_challenges.json \
  eval_tasks trm-code/kaggle/combined/arc-agi_evaluation2_solutions.json
python3 LUCA_hybrid/hybrid/solve_hybrid.py eval_tasks candidates.json hybrid_out --seconds 30
python3 LUCA_hybrid/hybrid/solve_hybrid.py --score eval_tasks hybrid_out
```

The score line prints how many tasks were solved, and how many of them came from LUCA and how many
from TRM. To see TRM on its own, pass an empty candidates file to step 5: `echo {} > none.json`.

## What else is in the kit

* `hybrid/generate.py`: the synthetic task generator. LUCA's programs are applied to new inputs,
  so every generated task's answer is correct by construction.
* `synthetic_v1/`: 6,334 synthetic tasks from 320 training tasks, for further training of TRM on a
  bigger machine.
* LUCA itself (`run_luca.py`, `luca.py`). See `README.md` and `docs/GENERAL_SOLVER.md`.
