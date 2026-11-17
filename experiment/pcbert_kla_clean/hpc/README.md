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

### Locked internal held-out evaluation

`project_clear_saved_test.sbatch` reloads the seed-42 validation-selected
checkpoints for PCBert-Kla, original token-gated MM-Kla, HybridKla-aligned,
and DeepKla-aligned. Here **original MM-Kla** means the four-layer ProtBERT
token-gated model with 27 descriptors, no CKSAAP, and the AdamW encoder/head
learning-rate configuration saved under `token_gated_protbert_head_lr`.
Before scoring held-out rows, each checkpoint must reproduce its saved
validation AUPRC, AUC, and MCC. `project_clear_stack_test.sbatch` reruns the
seed-42 five-fold two-branch experiment because the earlier run did not save
fold checkpoints. It checks its validation AUPRC and AUC against the earlier
report before accepting held-out metrics. Both jobs use threshold 0.5 and
the same locked CLEAR-Lactyl test rows. No test metric is used to choose a
configuration.
The two-branch prediction averages five models trained on different grouped
subsets; the three single-network checkpoints were fitted once on the full
training partition. This training procedure difference should accompany any
comparative table.

The CLEAR-Lactyl `internal_test` partition is protein/homology-group held
out, **not** a separate-study independent dataset. Reports retain that name.
The saved models are architecture-aligned implementations, with DeepKla and
HybridKla explicitly not claimed as exact reproductions of the authors' code.

After syncing the code commit and setting `EXPECTED_COMMIT`, submit from
this directory:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
sbatch --export=ALL project_clear_saved_test.sbatch
sbatch --export=ALL project_clear_stack_test.sbatch
```

Both jobs require an allocated A5000 compute GPU, the locked data files, the
isolated Python environment, and the existing seed-42 result artifacts. The
two-branch reference report is in the sibling `MM-Kla-stack` checkout by
default; `STACK_REFERENCE_REPORT` overrides it. Outputs go to separate
`outputs/hpc_clear_lactyl_final/{saved,stacked}/seed_42/` directories and
will not overwrite previous runs.

`project_deepkla_clear.sbatch` runs the DeepKla-aligned PyTorch port on the
same locked CLEAR-Lactyl train and validation partitions. It uses the released
architecture and residue encoding with RMSprop and MSE, but selects the best
epoch by validation AUPRC under this benchmark's protocol. It is not an exact
Keras 2.0.6 reproduction. The script checks the locked data hashes and code
commit, requires an allocated compute GPU, and writes only validation metrics.
After syncing the code commit and the two locked data files, submit from this
directory:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
sbatch --export=ALL project_deepkla_clear.sbatch
```

The outputs are in
`outputs/hpc_clear_lactyl_development/deepkla_aligned/seed_42/results/`.
The `report.json`, `history.json`, and `validation_predictions.csv` files are
the comparison artifacts; `best_model.pt` contains the validation-selected
weights. No internal-test prediction is generated.

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

### Full-depth ESM-2 150M backbone

`project_mmkla_esm2_150m_full30_clear.sbatch` replaces the ProtBERT backbone
of the full-depth token-gated model with `facebook/esm2_t30_150M_UR50D`.
All 30 ESM-2 layers and the unchanged MM-Kla head are trainable; neither LoRA
nor layer freezing is used. ESM-2 receives unspaced amino-acid windows. The
27 descriptors, site-guided attention, 256-dimensional gated fusion, and
residual classifier are unchanged. CKSAAP is not included.

The job uses the same CLEAR-Lactyl lock and seed-42 protocol as
`token_gated_protbert_full30_all`: batch size 4, AdamW (`2e-5` encoder,
`1e-4` head), weight decay 0.01, linear schedule with 10% warmup, at most 30
epochs, patience 5, and validation AUPRC for checkpoint selection. The
held-out test is not evaluated. Submit after syncing the commit:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
sbatch --export=ALL project_mmkla_esm2_150m_full30_clear.sbatch
```

The separate output directory is
`outputs/hpc_clear_lactyl_development/token_gated_esm2_150m_full30/seed_42/`.

### Two-branch positional sequence experiment

`project_mmkla_positional_stack_clear.sbatch` tests a two-branch development
model on the locked CLEAR-Lactyl split. The first branch is the existing
four-layer ProtBERT token-gated model with 27 descriptors and CKSAAP gaps 0-3.
The second branch independently encodes the same 51-residue windows with
residue and position embeddings, a two-layer bidirectional LSTM, and
central-lysine-guided attention. This is an architecture experiment, not a
reproduction of HybridKla.

Five stratified homology-group folds produce out-of-fold training scores for
both branches. Within each outer fold, checkpoint epochs are chosen on a
separate training-only grouped fold, then each branch is refitted on the
outer training groups. A fixed logistic regression is fitted on the two
out-of-fold scores. Validation scores are averages from the five outer-fold
models; the report gives both branches, their unweighted mean, and the
logistic combination separately. The internal test is not scored.

Submit from the HPC directory after syncing the commit:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
sbatch --export=ALL project_mmkla_positional_stack_clear.sbatch
```

The report, out-of-fold scores, and validation predictions are written to
`outputs/hpc_clear_lactyl_development/mmkla_positional_stack/seed_42/results/`.

### ProtBERT-contextual positional experiment

`project_mmkla_contextual_positional_clear.sbatch` tests a contextual version
of the positional predictor. It computes per-residue embeddings once from
the complete, pretrained 30-layer `Rostlab/prot_bert` encoder in inference
mode. The encoder stays frozen. A trainable 1024-to-64 projection adds these
embeddings to the existing learned residue and position embeddings; the
two-layer BiLSTM, site-guided attention, classifier, optimizer, and grouped
training protocol are otherwise unchanged. This differs from the separate
four-layer **fine-tuned** ProtBERT predictor used in the existing stack.

The job verifies and reuses that stack's saved seed-matched, lock-matched
out-of-fold and validation predictions. It trains only the contextual
positional predictor on the same five outer grouped folds, selects epochs on
inner training-only grouped folds, and refits the logistic combination using
training out-of-fold scores. Its report includes the old positional-only and
stack metrics alongside contextual positional-only and contextual stack
metrics. No held-out test rows are encoded or scored. The new embeddings
are cached in memory for this job and are not saved as an output artifact.

The reference stack must already be complete for the selected seed. Submit
from the HPC directory after syncing the code commit:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
sbatch --export=ALL project_mmkla_contextual_positional_clear.sbatch
```

The default reference is
`outputs/hpc_clear_lactyl_development/mmkla_positional_stack/seed_42/results/`;
`REFERENCE_DIR` can override it. New results go to
`outputs/hpc_clear_lactyl_development/mmkla_contextual_positional_stack/seed_42/results/`.
Only the CLEAR-Lactyl training and validation partitions are used. The
validation results are development evidence, not independent-test performance.

### No-physicochemical-descriptors stack ablation

`project_mmkla_no_27_descriptors_clear.sbatch` removes the 27 handcrafted
physicochemical and composition descriptors from the fine-tuned, four-layer
ProtBERT predictor while retaining its CKSAAP gap-0-to-3 features. It reuses
the frozen contextual positional predictor from the completed seed-matched
contextual stack. Five protein-grouped outer folds and inner grouped epoch
selection retrain only the changed ProtBERT predictor. A new logistic combiner
is fitted on training-only out-of-fold scores from the two predictors. This
ablation does not retrain or score the held-out test partition.

Submit from the HPC directory after syncing the code commit:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
sbatch --export=ALL project_mmkla_no_27_descriptors_clear.sbatch
```

The reference directory defaults to
`outputs/hpc_clear_lactyl_development/mmkla_contextual_positional_stack/seed_42/results/`.
Results are written to
`outputs/hpc_clear_lactyl_development/mmkla_no_27_descriptors_stack/seed_42/results/`.
The validation report includes the full-stack, contextual-only, ablated
ProtBERT-only, and ablated-stack metrics for comparison.

### Contextual predictor plus 27 descriptors

`project_mmkla_contextual_descriptors_clear.sbatch` tests whether the 27
sequence-derived physicochemical and composition descriptors improve the
contextual positional predictor directly. The contextual predictor's
seed-matched out-of-fold and validation scores are reused unchanged. A small
27-to-64-to-1 MLP is fitted on the descriptors alone, with scaling fitted on
each training fold. Five protein-grouped outer folds produce descriptor
out-of-fold scores; inner grouped folds select the epoch by AUPRC. Logistic
regression combines the two predictors using training-only out-of-fold
scores. Validation results include contextual-only, descriptors-only, their
combination, and the previous full two-predictor stack. No held-out test
partition is scored.

Submit from the HPC directory after syncing the code commit:

```bash
export EXPECTED_COMMIT="$(git -C ../../.. rev-parse HEAD)"
export RUN_SEED=42
sbatch --export=ALL project_mmkla_contextual_descriptors_clear.sbatch
```

The default reference is
`outputs/hpc_clear_lactyl_development/mmkla_contextual_positional_stack/seed_42/results/`.
New results are written to
`outputs/hpc_clear_lactyl_development/mmkla_contextual_descriptors/seed_42/results/`.

Results are written to
`outputs/hpc_clear_lactyl_development/token_gated_protbert_cksaap/seed_42/`.
