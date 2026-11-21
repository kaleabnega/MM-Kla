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
