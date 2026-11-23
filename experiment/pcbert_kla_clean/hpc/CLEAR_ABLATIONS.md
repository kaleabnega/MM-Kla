# CLEAR-Lactyl component controls

`project_clear_component_ablations.sbatch` is a validation-only Slurm array:
eight variants across seeds 42, 123, and 777 (24 tasks, at most two
concurrent). It does not score the locked internal test. Every task uses the
same 20,599 training and 2,944 validation rows, 51-residue windows, 27
sequence-derived descriptors plus CKSAAP gaps 0-3, four trainable ProtBERT
layers, AdamW, a 30-epoch cap, and validation-AUPRC selection.

The controls change only the named component of the ProtBERT--descriptor
predictor. `none` is the vector-gated, target-attention, residual-head
control. `cls_pool`, `mean_pool`, and `max_pool` replace target-conditioned
pooling. `concat_fusion` and `scalar_gate` replace the vector gate.
`no_residual` removes both residual blocks but keeps the final MLP head.
`frozen_encoder` leaves the same first four ProtBERT layers frozen. The
position-aware predictor is not trained by this array; its already reported
standalone/combined comparison is a separate predictor-level control.

The standalone branch training protocol is **not identical to the five-fold
out-of-fold stacking protocol** used for the final MM-Kla score. Consequently,
these validation results screen component sensitivity; they cannot be inserted
as final two-predictor held-out ablations without a matched stacking rerun.

The pinned lock files must be present on the cluster under
`outputs/clear_lactyl_locked/`, and the HPC virtual environment must have its
`.ready` marker. From the repository's `experiment/pcbert_kla_clean/hpc`
directory on the server:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
sbatch --export=ALL project_clear_component_ablations.sbatch
```

The script rejects a non-compute-node execution, a dirty/mismatched Git
checkout, changed lock hashes, or an existing output directory. Results are
under `outputs/hpc_clear_component_ablations/<variant>/seed_<seed>/`.
After the array completes:

```bash
python scripts/summarize_clear_component_ablations.py \
  --output-root outputs/hpc_clear_component_ablations
```

Run the summary command from `experiment/pcbert_kla_clean/`. It checks that
all runs followed the same declared protocol and reports each metric, paired
differences from the control, and across-seed summaries. Do not select a
threshold or architecture based on the locked internal test.

## Full two-predictor validation controls

`project_clear_stack_ablations.sbatch` repeats the proposed component
replacements inside the five-fold grouped-OOF MM-Kla training protocol, using
seed 42. It includes the ungated fixed-average fusion control. The unchanged
positional BiLSTM OOF and validation scores are reused from the prior
validation-only stack, with lock hash, group IDs, labels, and validation row
identities checked before training. Each task retrains the first predictor
on all five outer folds, refits a logistic combiner to training OOF scores,
and scores validation only. The existing manifest loader parses the internal
test records for integrity checks, but they are never passed to a model,
combiner, or metric in these runs.

Submit from the server's `hpc/` directory after syncing the new Git commit:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
sbatch --export=ALL project_clear_stack_ablations.sbatch
```

The default positional reference points to the existing `MM-Kla-stack`
checkout; set `REFERENCE_DIR` if the saved reference resides elsewhere.
After all nine tasks complete, run from `experiment/pcbert_kla_clean/`:

```bash
python scripts/summarize_clear_stack_ablations.py \
  --output-root outputs/hpc_clear_stack_ablations \
  --reference-dir /absolute/path/to/mmkla_positional_stack/seed_42/results
```

The control task must reproduce the original stacked validation AUPRC and
AUROC within 0.001. This phase remains seed-42 validation evidence, not a
multi-seed or external-test result.

## Two-seed completion and additional controls

The remaining full-stack controls isolate three components of the positional
predictor: its learned position embeddings, second BiLSTM layer, and
center-conditioned attention pooling. In `positional_mean_pool`, the central
state is still passed to the classifier, but the context vector is the mean
of all 51 recurrent states. Two further controls remove either the 27
descriptors or all 1,600 CKSAAP features from the ProtBERT branch; the
remaining family is scaled on the relevant training fold only. Each variant
reuses saved out-of-fold and validation scores from the unchanged opposite
branch, refits the logistic combiner on training OOF scores, and evaluates
only the locked validation partition.

The five new seed-42 variants and the fourteen seed-123 variants form 19
remaining ablation tasks. Seed 123 also needs one unchanged reference-stack
run. The seed-123 array should be submitted with an `afterok` dependency on
the reference job. Both arrays run at most two tasks concurrently. Paths,
dataset hashes, and Git commit are checked before training; existing run
directories are never overwritten.

After completion, from `experiment/pcbert_kla_clean/` run:

```bash
python scripts/summarize_clear_two_seed_ablations.py \
  --output-root outputs/hpc_clear_stack_ablations \
  --seed-42-reference /absolute/path/to/MM-Kla-stack/experiment/pcbert_kla_clean/outputs/hpc_clear_lactyl_development/mmkla_positional_stack/seed_42/results \
  --seed-123-reference outputs/hpc_clear_lactyl_development/mmkla_positional_stack/seed_123/results
```

The summary checks all fourteen variants per seed against their unchanged
reference and reports both seed-specific results and paired differences. A
two-seed result is descriptive, especially when differences are near 0.001
AUPRC; the held-out test remains untouched throughout these ablations.

## Standalone first-predictor controls

`project_clear_standalone_branches.sbatch` runs four validation-only tasks:
ProtBERT sequence-only and 27-descriptor-only, each for seeds 42 and 123.
Sequence-only removes both the 27 descriptors and all CKSAAP features from
the first predictor. Descriptor-only removes its ProtBERT encoder and CKSAAP,
but retains its existing feature projection and classifier. These are not the
same as the earlier independent 27-feature MLP experiment.

Both controls use the existing five grouped outer folds and inner grouped
epoch selection. The unchanged positional predictor's saved scores are reused
from the seed-matched reference stack. Each report records the standalone
first-predictor validation metrics and, separately, its newly refitted
combination with the positional predictor. The locked internal test is not
scored. Tasks run at most two at a time, one GPU per task; outputs are kept
under `outputs/hpc_clear_standalone_branches/<mode>/seed_<seed>/`.

After syncing the code commit, submit from the HPC `hpc/` directory:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
sbatch --export=ALL project_clear_standalone_branches.sbatch
```

Array tasks 0 and 2 are ProtBERT-only (seeds 42 and 123); tasks 1 and 3
are descriptor-only. The wrapper checks the Git commit, dataset lock hashes,
complete reference outputs, and destination before starting training.
