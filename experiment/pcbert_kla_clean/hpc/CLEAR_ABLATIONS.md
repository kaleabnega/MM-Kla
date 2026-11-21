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
