# Slurm development runs

These jobs reproduce the validation-only KlaDB human comparison in
`kaggle_human_esm2_lora_three_models.ipynb` without Kaggle's per-run time limit.
One job trains one model for one declared seed. The internal and study-held-out
test sets are not evaluated. `hybridkla_architecture_aligned` is the existing
architecture-aligned retraining pipeline, not an exact reproduction of the
authors' original training procedure.

## Cluster allocation

The job files target the `cvlab-a5000` Slurm partition. The training job
requests one GPU, eight CPUs, 48 GiB of memory, and four days. It does not
select a node. The environment job requests one GPU to verify that PyTorch can
run on the allocated compute node before training is submitted;
neither script can run directly in a login shell. The two RTX 5090 devices
visible on the SSH host are not assumed to be the GPUs allocated by Slurm.

## Inputs and environment

The repository must be present on the server as a Git checkout. The following
files from `outputs/kladb_human_locked/` must be placed together at the same
relative path in the server checkout, or `LOCK_DIR` must point to their directory:

- `split_audit.json`
- `model_manifest.csv`

The training job checks both SHA-256 digests against the locked benchmark.
`full_manifest.csv` is not needed for model training. The two files above are
ignored by Git and must be transferred separately. No dataset is downloaded or
reconstructed by the Slurm jobs.

Submit the environment job once from this `hpc/` directory:

```bash
sbatch setup_environment.sbatch
```

When submitting from another directory, pass `--chdir=/absolute/path/to/hpc`.
Slurm copies batch scripts into its spool area, so the jobs resolve the project
from their working directory rather than from the script file path.

Wait for it to finish successfully. It creates an isolated Python 3 virtual
environment at `../.venv-hpc-cu118` relative to the repository, installs
PyTorch 2.7.1 with CUDA 11.8, the project requirements, and PEFT, and records
a dependency freeze. It marks the environment ready only after a CUDA tensor
operation succeeds on the allocated GPU. This uses no system-level changes and
does not modify the earlier environment. The compute node must have package-index
access for this setup step. The environment is created using the account's
Miniconda Python because the system Python on the compute nodes lacks
`ensurepip`; `PYTHON_FOR_VENV` can override that interpreter. `VENV_DIR` can
override the environment location for both jobs.
Because the environment is isolated, Kaggle's incompatible preinstalled
`torchao` is not inherited.

HybridKla-aligned runs additionally require the official `HybridKla.bin` or
`HybridKla.zip` archive on the server. Set `HYBRID_ARCHIVE` to its absolute
path. The job verifies its checksum and extracts only the two required feature
tables; the model runner verifies those tables again.

## Submit training

From this `hpc/` directory on the server, set a full commit hash and submit one
job for each model and seed. For example:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
export RUN_MODEL=token_gated
sbatch --job-name=project-esm --export=ALL train_development.sbatch
```

Valid `RUN_MODEL` values are `baseline`, `token_gated`, and
`hybridkla_architecture_aligned`. Valid seeds are `42`, `123`, and `777`. For a
HybridKla-aligned job, also export `HYBRID_ARCHIVE` before `sbatch`. The code
commit, locked-data checksums, GPU allocation, and Python environment are
verified before training. The job refuses to overwrite an existing run
directory. The `--export=ALL` option ensures the model, seed, and paths reach
the Slurm job.

Results are written under
`outputs/hpc_human_esm2_lora/<model>/seed_<seed>/` unless `OUTPUT_ROOT` is set.
Slurm writes `slurm-kla-development-<jobid>.out` in the submission directory.
The PCBert-Kla and MM-Kla runs save their validation-selected model weights;
HybridKla-aligned writes its branch checkpoints. These outputs can be large,
so verify available storage before submitting all seeds.

Start with seed 42 for each model. Later seeds must use the same code commit,
dataset lock, and environment. The training code does **not** resume an
interrupted epoch or job; rerunning with a smaller epoch count would be a new
experiment, not continuation. No test partition should be used to select a
model or threshold.

## Thirty-epoch ESM-2 run and paired comparison

Set `RUN_PATIENCE=0` for an MM-Kla run that completes all 30 epochs. The
validation AUPRC is printed after every epoch, and the checkpoint with the
highest validation AUPRC is selected after training. It does not use the test
sets. Use a new `OUTPUT_ROOT`; existing run directories are never overwritten.

`compare_baseline_hybrid.sbatch` runs PCBert-Kla and then the
HybridKla-aligned implementation sequentially on one allocated GPU, for the
same seed and locked development split. PCBert-Kla retains its predeclared
patience of five; the HybridKla-aligned pipeline retains its own stage-level
patience of six. Both runs use the same `OUTPUT_ROOT` as the MM-Kla run, in
separate model directories. The comparison job verifies the official
`HybridKla.bin` checksum before starting PCBert-Kla. By default it expects
the archive beside the repository in the account's work directory; set
`HYBRID_ARCHIVE` to an absolute path to use another location.

The comparison may be queued with `--dependency=afterok:<ESM_JOB_ID>`. This
starts it only after the full-epoch MM-Kla job succeeds. A single comparison
job avoids a second GPU allocation, but if HybridKla-aligned fails, the
completed PCBert-Kla results remain in its output directory.

## CLEAR-Lactyl validation jobs

`project_mmkla_clear.sbatch` trains the current ESM-2-650M LoRA token-gated
MM-Kla model. `project_compare_clear.sbatch` trains the four-layer ProtBert
PCBert-Kla baseline and then the eight-branch HybridKla-aligned model on one
allocated GPU. Both jobs use the same protein/homology-grouped CLEAR-Lactyl
training and validation partitions, seed 42 by default, and **do not score the
held-out test**. The HybridKla runner is architecture-aligned, not an exact
replication of the upstream training code; the provenance of the official
AAindex `top10.txt` selection is not established.

`project_mmkla_protbert_clear.sbatch` runs the token-gated architecture with
four fine-tuned ProtBert layers and the earlier SGD settings (learning rate
0.003, zero weight decay, no scheduler). It writes to
`token_gated_protbert_sgd/seed_42/`, leaving the ESM-2 results and any
incomplete AdamW run intact. It uses the same CLEAR-Lactyl training and
validation partitions, maximum 30 epochs, validation-AUPRC early stopping,
and fixed 0.5 threshold. The ESM-2 run differs in encoder depth and adaptation
method as well as optimizer, so the comparison does not isolate backbone
identity alone.

`project_mmkla_protbert_adamw_clear.sbatch` uses the same four-layer ProtBert
token-gated model, split, seed, batch size, and early-stopping rule as the SGD
job. It applies AdamW (learning rate 2e-5, weight decay 0.01, linear schedule
with 10% warmup) and writes to `token_gated_protbert_adamw/seed_42/`. This is
a comparison of optimizer configurations, not of the optimizer algorithms
alone, because learning rate, weight decay, and scheduling also differ.

The two files in `outputs/clear_lactyl_locked/` required on the server are
`split_audit.json` and `model_manifest.csv`; the submission script checks
their pinned hashes. These are ignored by Git and must be transferred
separately. The existing isolated `.venv-hpc-cu118` environment and official
`HybridKla.bin` are reused. The jobs do not run training on the login node.

Submit from this directory with `EXPECTED_COMMIT` set to the corresponding
40-character commit and `RUN_SEED=42`. The Slurm jobs are named
`project-mmkla-clear` and `project-compare-clear`. Results go to
`outputs/hpc_clear_lactyl_development/<model>/seed_42/` and each job refuses
to overwrite a pre-existing directory. No final test comparison should be
claimed from these validation-only runs.

## Central-lysine BiLSTM experiment

`project_mmkla_bilstm_control.sbatch` reruns the existing four-layer ProtBERT
token-gated MM-Kla model. `project_mmkla_bilstm_clear.sbatch` adds a separate
one-layer, bidirectional 64-unit LSTM over learned 64-dimensional residue
embeddings. Its central-lysine state is concatenated with the existing
attention-pooled ProtBERT sequence vector before the 256-dimensional sequence
projection. The 27-feature branch, gated fusion, and residual classifier are
unchanged. The new branch does not use protein structures or test labels.

Both jobs use the same CLEAR-Lactyl lock, seed, 30-epoch limit, early stopping,
and AdamW schedule. ProtBERT parameters use `2e-5`; non-encoder parameters use
`1e-4` in **both** jobs, so the new branch is the only intentional architectural
difference. Because this learning-rate grouping differs from the earlier MM-Kla
run, its old validation score is not the matched control. The jobs are
validation-only and do not score the held-out test. Compare validation AUPRC
first, then the other saved metrics; do not choose a model using the test set.

Submit from this directory on the server after syncing the new commit and
verifying that the locked data and isolated Python environment are present:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
sbatch --export=ALL project_mmkla_bilstm_control.sbatch
sbatch --export=ALL project_mmkla_bilstm_clear.sbatch
```

Results go to separate `token_gated_protbert_head_lr/seed_42/` and
`token_gated_protbert_bilstm/seed_42/` directories under
`outputs/hpc_clear_lactyl_development/`. Neither script pins a node or runs
training without a Slurm compute allocation. Subsequent seeds should use the
same code commit and lock; neither job resumes an interrupted run.

### Full-depth ProtBERT with last-four-layer adaptation

`project_mmkla_full_protbert_clear.sbatch` evaluates the existing token-gated
MM-Kla head with all 30 pretrained ProtBERT encoder layers retained. The
embeddings and layers 0-25 are frozen; only layers 26-29 and the task-specific
head are trainable. This contrasts with `project_mmkla_bilstm_control.sbatch`,
which retains and trains only layers 0-3. The locked CLEAR-Lactyl training and
validation partitions, seed, batch size, optimizer, learning-rate groups,
30-epoch cap, early stopping, and AUPRC selection are otherwise the same.
This tests the full-depth/last-four configuration, not depth in isolation.
The held-out test is not evaluated.

Submit from the HPC directory after syncing the commit:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
sbatch --export=ALL project_mmkla_full_protbert_clear.sbatch
```

Results are written to
`outputs/hpc_clear_lactyl_development/token_gated_protbert_full30_last4/seed_42/`.

### Four-layer ProtBERT with multi-gap CKSAAP

`project_mmkla_cksaap_clear.sbatch` extends the 27 sequence-derived features
of `project_mmkla_bilstm_control.sbatch` with 1,600 normalized, ordered
amino-acid-pair frequencies: 400 each at gaps `k=0,1,2,3`. The same four
retained and trainable ProtBERT layers, token-gated architecture, AdamW rate
groups (`2e-5` encoder, `1e-4` head), locked CLEAR-Lactyl split, batch size,
seed, 30-epoch limit, and validation-AUPRC selection are used. Feature scaling
is fitted on the training partition only. The held-out test is not scored.

Submit from the HPC directory after syncing the commit:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
sbatch --export=ALL project_mmkla_cksaap_clear.sbatch
```

### All-layer ProtBERT and projected CKSAAP

`project_mmkla_full_protbert_all_clear.sbatch` keeps all 30 ProtBERT layers and
trains all of them. Unlike the earlier `full30_last4` run, no encoder layers
are frozen. It keeps the token-gated head, 27 descriptors, AdamW learning-rate
groups (`2e-5` encoder, `1e-4` head), and the existing validation protocol.
Compare it with the four-layer `token_gated_protbert_head_lr` control and the
`full30_last4` run; the latter is not a test of all-layer fine-tuning.

`project_mmkla_cksaap_projection_clear.sbatch` keeps four trainable ProtBERT
layers. It projects the 1,600 CKSAAP features to 128 dimensions using
LayerNorm, Linear, GELU, and 0.3 dropout. The projected vector is concatenated
with the original 27 descriptors before the existing feature projection and
gated fusion. Compare it with `token_gated_protbert_cksaap` to isolate this
feature-branch change.

Both jobs use the same locked CLEAR-Lactyl train/validation partitions,
seed, batch size 4, maximum 30 epochs, patience 5, and validation AUPRC for
checkpoint selection. Neither scores the held-out test. After syncing the
code commit, submit from the HPC directory:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
sbatch --export=ALL project_mmkla_full_protbert_all_clear.sbatch
sbatch --export=ALL project_mmkla_cksaap_projection_clear.sbatch
```

Results have distinct `token_gated_protbert_full30_all/seed_42/` and
`token_gated_protbert_cksaap_projection/seed_42/` directories under
`outputs/hpc_clear_lactyl_development/`. Each job requires a Slurm compute
allocation and refuses an existing run directory.

Results are written to
`outputs/hpc_clear_lactyl_development/token_gated_protbert_cksaap/seed_42/`.
