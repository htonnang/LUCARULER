# Submitting the LUCA + TRM hybrid to Kaggle (ARC Prize, ARC-AGI-2 track)

Kaggle runs your notebook **offline** (internet off) on the hidden test tasks. The notebook therefore
needs everything uploaded beforehand as **Kaggle Datasets**. The entry point
`hybrid/kaggle_main.py` finds them automatically under `/kaggle/input`.

On the hidden test set TRM has never seen the tasks, so the script runs **test-time training** on
their demonstrations before predicting. Expect a lower score than the 13/120 measured on the public
evaluation set, whose demonstrations the model had already learned. For reference, the model's authors
reported 6.67% on the semi-private set for an earlier checkpoint with test-time training.

## 1. Prepare four folders on your Mac (Terminal)

```
cd /Users/htonnang/Desktop/AGI2NEURAL && source venv/bin/activate
mkdir -p kaggle_upload && cd kaggle_upload
```

1. **luca-hybrid**, the kit (LUCA and the hybrid scripts):
   ```
   mkdir -p luca-hybrid && cp -R ../LUCA_hybrid luca-hybrid/
   ```
2. **trm-code**, the TRM code (only the parts that are needed):
   ```
   mkdir -p trm-code && cp -R ../trm-code/models ../trm-code/dataset ../trm-code/evaluators ../trm-code/utils ../trm-code/puzzle_dataset.py ../trm-code/config trm-code/
   ```
3. **trm-model**, just the final checkpoint and its settings, not all 40 checkpoints:
   ```
   mkdir -p trm-model && cp ../trm-model/step_737320 ../trm-model/all_config.yaml trm-model/
   ```
4. **wheels**, small Python packages that may be missing on Kaggle (pure Python, so they work on Linux):
   ```
   mkdir -p wheels && pip download argdantic coolname --no-deps --only-binary=:all: -d wheels
   ```

## 2. Upload them as Kaggle Datasets (kaggle.com, in your browser)

For each of the four folders: **Datasets → New Dataset**, drag the folder in, give it the folder's
name (`luca-hybrid`, `trm-code`, `trm-model`, `wheels`) and keep it **Private**. Wait until each
upload has finished processing.

## 3. Create the notebook

1. Open the competition **ARC Prize 2026 – ARC-AGI-2**, then **Code → New Notebook**. The
   competition data is attached automatically.
2. In the notebook's right-hand panel:
   * **Session options → Accelerator:** choose a GPU (whatever the competition allows).
   * **Internet:** **Off**. Code competitions require this.
   * **Add Input → Your Datasets:** add `luca-hybrid`, `trm-code`, `trm-model` and `wheels`.
3. Put this in the first cell:
   ```python
   import glob, os
   # time budget in minutes: keep it about 30 minutes under the competition's runtime limit
   os.environ["LUCA_TOTAL_MINUTES"] = "690"
   main = glob.glob('/kaggle/input/**/hybrid/kaggle_main.py', recursive=True)[0]
   print(main)
   !python {main}
   ```
4. Click **Run All** once to check that it starts. The log shows the paths it found, then
   `fallback submission written`, `LUCA started`, `building dataset` and the training lines.
   When the log shows the training lines, you can stop the run (the full run comes in step 4).

## 4. Submit

1. **Save Version → Save & Run All (Commit)**. Kaggle runs the whole notebook; this takes the full
   time budget.
2. When the commit has finished, open it, then **Submit to Competition** (or go to the competition's
   **Submit Prediction** and choose this notebook and version). Kaggle reruns it on the hidden test
   set and shows the score on the leaderboard.

## Settings (environment variables, set in the first cell before `!python`)

| Variable | Default | Meaning |
|---|---|---|
| `LUCA_TOTAL_MINUTES` | 690 | total time budget (keep it under the competition limit) |
| `LUCA_SECONDS` | 30 | LUCA's time per task |
| `TTT_NUM_AUG` | 256 | augmented copies built per task |
| `TTT_MAX_AUG` | 128 | augmented copies used for voting at prediction time |
| `TTT_BATCH` | 96 | training batch; halved automatically if the GPU runs out of memory |
| `TTT_INFER_BATCH` | 128 | prediction batch |

## Several GPUs (e.g. L4 x4)

The script uses every GPU it finds: the hidden tasks are split into one group per GPU, and each GPU
trains on and predicts its own group, in a separate process, for the whole time budget. With four
GPUs each task gets about four times the training of a single-GPU run. If one GPU's process
fails, only its tasks lose the neural answers (they keep LUCA's). Each process writes its own
log to `/kaggle/working/ttt/shardN/log.txt`, and the main log repeats the latest line of each every
10 minutes.

## What the script guarantees

* A valid `submission.json` exists from the first minute, filled with identity guesses. Every later
  stage only improves it, so a crash or timeout never leaves you without a submission.
* If the TRM stage fails (e.g. a missing dataset), the submission still contains LUCA's answers.
* All of this is open source, as ARC Prize requires for prize eligibility.
