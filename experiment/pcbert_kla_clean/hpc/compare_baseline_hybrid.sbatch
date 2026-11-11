#!/usr/bin/env bash
#SBATCH --job-name=kla-compare
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" || "${SLURM_JOB_PARTITION:-}" != cvlab-a5000 ]]; then
    echo "Refusing to train outside a Slurm allocation." >&2
    exit 2
fi

HPC_DIR="$(pwd -P)"
if [[ ! -f "${HPC_DIR}/train_development.sbatch" ]]; then
    echo "Submit from the hpc directory or pass --chdir to sbatch." >&2
    exit 2
fi
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/hpc_human_esm2_lora_full30}"
HYBRID_ARCHIVE="${HYBRID_ARCHIVE:-${REPO_DIR}/../HybridKla.bin}"
RUN_SEED="${RUN_SEED:?Set RUN_SEED to 42, 123, or 777}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the repository commit}"

if [[ ! -f "${HYBRID_ARCHIVE}" ]]; then
    echo "Missing HybridKla archive: ${HYBRID_ARCHIVE}" >&2
    exit 2
fi
printf '%s  %s\n' \
    '73ddb128c441d7b213c4636e6bb6ce34501e82143c2b9bc06ec329d68a422993' "${HYBRID_ARCHIVE}" \
    | sha256sum --check --status
for model in baseline hybridkla_architecture_aligned; do
    if [[ -e "${OUTPUT_ROOT}/${model}/seed_${RUN_SEED}" ]]; then
        echo "Run directory already exists: ${OUTPUT_ROOT}/${model}/seed_${RUN_SEED}" >&2
        exit 2
    fi
done

export OUTPUT_ROOT HYBRID_ARCHIVE RUN_SEED EXPECTED_COMMIT
export RUN_PATIENCE=5
RUN_MODEL=baseline bash "${HPC_DIR}/train_development.sbatch"
RUN_MODEL=hybridkla_architecture_aligned bash "${HPC_DIR}/train_development.sbatch"
echo "Baseline and HybridKla-aligned development runs completed for seed ${RUN_SEED}."
