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
select a node. The environment job also runs under Slurm on a compute node;
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

Wait for it to finish successfully. It creates an isolated Python 3 virtual
environment at `$HOME/.venvs/mm-kla-hpc`, installs the project requirements and
PEFT, and records a dependency freeze. The compute node must have package-index
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
sbatch --export=ALL train_development.sbatch
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
