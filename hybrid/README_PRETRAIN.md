# Continued pre-training of the neural model on LUCA-generated tasks (Mac)

The neural model (TRM) learns each hidden task from its 2-4 demonstrations at test time. This
package gives it more practice first: it continues training the checkpoint on tasks LUCA generated,
then a fair before/after test checks whether it actually got better.

## What is in `data_v2/`

| File | Tasks | What |
|---|---|---|
| `composed.json` | 40,000 | new compositions of 1-3 LUCA operations with random settings, on ARC-style grids |
| `variants.json` | 6,334 | variants of 320 ARC training tasks, made with the programs LUCA found for them |
| `arc_training.json` | 1,000 | the ARC-AGI-2 training tasks, mixed in so the model keeps what it knew |
| `conceptarc_challenges.json` / `_solutions.json` | 160 | ConceptARC: the before/after test set, seen by neither checkpoint nor the generator |

No ARC-AGI-2 evaluation task and no ConceptARC task was used to make the training data.

## Steps (Terminal)

```
cd /Users/htonnang/Desktop/AGI2NEURAL && source venv/bin/activate
```

### 1. Smoke test (about 10 minutes)
```
python3 LUCA_hybrid/hybrid/pretrain_more.py --trm-repo trm-code --checkpoint trm-model/step_737320 \
  --tasks LUCA_hybrid/data_v2/composed.json --max-tasks 300 --hours 0.1 --save-every 3 \
  --workdir pretrain_smoke --out-dir trm-model-smoke
```
It should print `device mps`, then `step 1 ...`, `step 100 ...` lines with a speed in steps/s,
then `saved trm-model-smoke/step_...`. Send the last few lines (the speed decides the settings for
the night run).

### 2. Night run (8-9 hours)
```
caffeinate -i python3 LUCA_hybrid/hybrid/pretrain_more.py --trm-repo trm-code --checkpoint trm-model/step_737320 \
  --tasks LUCA_hybrid/data_v2/composed.json LUCA_hybrid/data_v2/variants.json LUCA_hybrid/data_v2/arc_training.json \
  --hours 9 --workdir pretrain_work --out-dir trm-model-v2
```
`caffeinate -i` keeps the Mac awake (keep it plugged in). Building the dataset takes about 10-20
minutes before the first step. A checkpoint is saved every hour in `trm-model-v2/`, so stopping
early (Ctrl+C) still leaves the last hourly checkpoint.

### 3. Before/after test (about 3-4 hours)
```
ls trm-model-v2
caffeinate -i python3 LUCA_hybrid/hybrid/ab_test.py --trm-repo trm-code \
  --checkpoints trm-model/step_737320 trm-model-v2/step_XXXXXX \
  --challenges LUCA_hybrid/data_v2/conceptarc_challenges.json --minutes 90
```
Replace `step_XXXXXX` with the highest step number that `ls` shows. Both checkpoints get the
same 90 minutes of test-time training on the same 160 tasks. The last lines print:
```
RESULT ...
  trm-model/step_737320: tasks A/160 ...
  trm-model-v2/step_...: tasks B/160 ...
  change: +N tasks
```
Send those lines. If the new checkpoint is better, it goes to Kaggle in place of the old one.
