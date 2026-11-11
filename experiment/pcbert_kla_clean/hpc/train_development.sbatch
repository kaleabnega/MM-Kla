#!/usr/bin/env bash
#SBATCH --job-name=kla-development
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
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
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
LOCK_DIR="${LOCK_DIR:-${PROJECT_DIR}/outputs/kladb_human_locked}"
RUN_MODEL="${RUN_MODEL:?Set RUN_MODEL to baseline, token_gated, or hybridkla_architecture_aligned}"
RUN_SEED="${RUN_SEED:?Set RUN_SEED to 42, 123, or 777}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the 40-character Git commit hash}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/hpc_human_esm2_lora}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_MODEL}/seed_${RUN_SEED}"

case "${RUN_MODEL}" in
    baseline|token_gated|hybridkla_architecture_aligned) ;;
    *) echo "Unsupported RUN_MODEL: ${RUN_MODEL}" >&2; exit 2 ;;
esac
case "${RUN_SEED}" in
    42|123|777) ;;
    *) echo "Unsupported RUN_SEED: ${RUN_SEED}" >&2; exit 2 ;;
esac
if [[ ! "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "EXPECTED_COMMIT must be a full lowercase Git commit hash." >&2
    exit 2
fi
if [[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
    echo "Repository commit differs from EXPECTED_COMMIT." >&2
    exit 2
fi
if [[ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked files differ from the committed code." >&2
    exit 2
fi
if [[ ! -f "${VENV_DIR}/.ready" || ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Python environment is missing; submit setup_environment.sbatch first." >&2
    exit 2
fi
if [[ "${RUN_MODEL}" == hybridkla_architecture_aligned ]]; then
    HYBRID_ARCHIVE="${HYBRID_ARCHIVE:?Set HYBRID_ARCHIVE to the official HybridKla.bin or .zip}"
    if [[ ! -f "${HYBRID_ARCHIVE}" ]]; then
        echo "HybridKla archive is missing: ${HYBRID_ARCHIVE}" >&2
        exit 2
    fi
fi
if [[ -e "${RUN_DIR}" ]]; then
    echo "Run directory already exists; refusing to overwrite: ${RUN_DIR}" >&2
    exit 2
fi

LOCK="${LOCK_DIR}/split_audit.json"
MANIFEST="${LOCK_DIR}/model_manifest.csv"
printf '%s  %s\n' \
    '2d1275577a430ff5758aa89c01a62c138c294ddc537bc67f671db3807856a0b8' "${LOCK}" \
    '842ff1b767d60f5645d3ba9c93150ec42ed9966dcc13672a679269327a25b95e' "${MANIFEST}" \
    | sha256sum --check --status

PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one Slurm-allocated GPU"; print("Allocated GPU:", torch.cuda.get_device_name(0), "memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))'
"${PYTHON}" "${PROJECT_DIR}/scripts/run_replication.py" --run human-study-check --human-lock "${LOCK}"

mkdir -p "${RUN_DIR}"
printf 'job_id=%s\nnode=%s\nmodel=%s\nseed=%s\ncommit=%s\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME:-unknown}" "${RUN_MODEL}" "${RUN_SEED}" "${EXPECTED_COMMIT}" \
    > "${RUN_DIR}/job_metadata.txt"
"${PYTHON}" -m pip freeze > "${RUN_DIR}/requirements.freeze.txt"
nvidia-smi

if [[ "${RUN_MODEL}" == hybridkla_architecture_aligned ]]; then
    RESOURCE_DIR="${RUN_DIR}/hybridkla_resources"
    "${PYTHON}" "${HPC_DIR}/extract_hybridkla_resources.py" "${HYBRID_ARCHIVE}" "${RESOURCE_DIR}"
    "${PYTHON}" -u "${PROJECT_DIR}/scripts/run_hybridkla_architecture.py" \
        --benchmark-lock "${LOCK}" \
        --resource-dir "${RESOURCE_DIR}" \
        --output-dir "${RUN_DIR}/results" \
        --device cuda:0 \
        --seed "${RUN_SEED}" \
        --esm-model facebook/esm2_t30_150M_UR50D \
        --esm-epochs 40 --lstm-epochs 30 --feature-epochs 100 --meta-epochs 50 \
        --esm-batch-size 32 --lstm-batch-size 32 --feature-batch-size 2048 \
        --patience 6 --development-only
else
    model_args=(--model-name Rostlab/prot_bert --encoder-layers 4)
    if [[ "${RUN_MODEL}" == token_gated ]]; then
        model_args=(--model-name facebook/esm2_t33_650M_UR50D \
            --encoder-layers 33 --encoder-adaptation lora \
            --lora-rank 8 --lora-alpha 32 --lora-dropout 0.3)
    fi
    "${PYTHON}" -u "${PROJECT_DIR}/scripts/run_replication.py" \
        --run human-study-development --human-lock "${LOCK}" \
        --architecture "${RUN_MODEL}" "${model_args[@]}" \
        --epochs 30 --patience 5 --batch-size 4 --device cuda:0 \
        --optimizer adamw --learning-rate 2e-5 --weight-decay 0.01 \
        --scheduler linear --warmup-ratio 0.1 \
        --positive-weighting none --monitor-metric AUPRC \
        --decision-threshold 0.5 --seed "${RUN_SEED}" \
        --save-models --output-dir "${RUN_DIR}/results"
fi

echo "Development run completed: ${RUN_DIR}"
