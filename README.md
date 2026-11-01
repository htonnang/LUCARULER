# LUCA–RULER: Exact Laws Meet Test-Time Learning on ARC-AGI-2

LUCA–RULER combines exact symbolic laws with test-time neural learning, scoring **5.83%** on
Kaggle's hidden ARC-AGI-2 tasks (ARC Prize 2026).

- **LUCA** — one language of 102 grid operations and one search. Every parameter is inferred from
  the demonstrations; a law is accepted only if it reproduces every demonstration, otherwise LUCA
  abstains.
- **RULER** — the relational vocabulary: three segmentations, panels, colour and object roles,
  relations (containment, contact, alignment).
- **Neural partner** — a 7M-parameter Tiny Recursive Model (TRM), trained on each hidden task at
  test time, tasks split across four GPUs. LUCA's exact answers always take the first attempt.

| Submission | Kaggle hidden test |
|---|---|
| LUCA alone (earlier release) | 0.83% |
| LUCA + TRM, test-time training on one GPU | 2.5% |
| **LUCA + TRM, test-time training on four GPUs** | **5.83%** |

## Repository layout

| Path | Contents |
|---|---|
| `luca_v31/` | LUCA: the language (`dsl/ops.py`), parameter inference (`dsl/infer.py`), search (`dsl/search.py`, `dsl/solver.py`), object layer (`dsl/objlearn.py`, `dsl/objmap.py`, `dsl/lift.py`) |
| `hybrid/` | Kaggle pipeline: `kaggle_main.py` (entry point of the scored submission), `ttt.py` + `trm_lib.py` (test-time training, 4-GPU sharding, voting), `solve_hybrid.py` + `constraints.py` (combiner and screening) |
| `hybrid/kaggle_symbolic.py` | CPU-only solver: LUCA + the V160 family library |
| `hybrid/synth_compose.py`, `hybrid/generate.py` | Task generators (40,000 composed tasks; variants of solved tasks) |
| `hybrid/pretrain_more.py`, `hybrid/kaggle_pretrain.py`, `hybrid/ab_test.py` | Continued pre-training of TRM and a ConceptARC before/after test |
| `v160/` | V160 equivariant family library (used by `kaggle_symbolic.py`) |
| `tools/variant_check.py` | Label-safe variant checker (rotations, colour permutations, held-out demonstrations) |
| `ledgers/` | Aggregate-only logs of every sealed-set check (public evaluation, Mini-ARC, ConceptARC) |

## Reproducing the scored submission (Kaggle, GPU L4 x4, internet off)

Attach as inputs: this repository (as a Kaggle dataset), the TRM code
(<https://github.com/TrelisResearch/TinyRecursiveModels>) and the checkpoint `step_737320` with its
`all_config.yaml` from <https://huggingface.co/Trelis/TRM-pretrain_arc2_rearc_200k>. Then run:

```
!LUCA_TOTAL_MINUTES=690 TTT_MAX_AUG=256 python $(find /kaggle/input -name kaggle_main.py | head -1)
```

Details: `hybrid/KAGGLE.md`. Local (Mac) instructions: `hybrid/README_MAC.md`.

## Regenerating the composed tasks

`data_v2/composed.json` (248 MB) is not stored here. Recreate it with:

```
python3 hybrid/synth_compose.py ARC_TRAINING_DIR data_v2/composed.json --tasks 40000 --seed 1
```

## Evaluation discipline

No task identifiers in solver code. Sealed sets (ARC-AGI-2 public evaluation, Mini-ARC, ConceptARC,
fresh RE-ARC instances) were checked only at milestones, in aggregate, each check logged in
`ledgers/` with a fingerprint of the solver code.

## Licence

Our code: MIT No Attribution (MIT-0), see `LICENSE`. Third-party code and the TRM checkpoint keep
their own licences.
