#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" || "${SLURM_JOB_PARTITION:-}" != cvlab-a5000 ]]; then
    echo "Refusing to train outside a Slurm compute allocation." >&2
    exit 2
fi

HPC_DIR="$(pwd -P)"
[[ -f "${HPC_DIR}/run_clear_development.sh" ]] || { echo "Run from the hpc directory." >&2; exit 2; }
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
LOCK_DIR="${LOCK_DIR:-${PROJECT_DIR}/outputs/clear_lactyl_locked}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/hpc_clear_lactyl_development}"
HYBRID_ARCHIVE="${HYBRID_ARCHIVE:-${REPO_DIR}/../HybridKla.bin}"
RUN_MODEL="${RUN_MODEL:?Set RUN_MODEL to a supported CLEAR-Lactyl model}"
RUN_SEED="${RUN_SEED:?Set RUN_SEED to a nonnegative integer}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the full repository commit}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_MODEL}/seed_${RUN_SEED}"

case "${RUN_MODEL}" in
    baseline|token_gated|token_gated_protbert|token_gated_protbert_sgd|token_gated_protbert_adamw|token_gated_protbert_head_lr|token_gated_protbert_full30_last4|token_gated_protbert_bilstm|hybridkla_architecture_aligned) ;;
    *) echo "Unsupported model: ${RUN_MODEL}" >&2; exit 2 ;;
esac
if [[ ! "${RUN_SEED}" =~ ^[0-9]+$ || ! "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Invalid seed or expected commit." >&2
    exit 2
fi
if [[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ||
      -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]]; then
    echo "Repository is not the expected clean commit." >&2
    exit 2
fi
if [[ ! -x "${VENV_DIR}/bin/python" || ! -f "${VENV_DIR}/.ready" ]]; then
    echo "Pinned HPC Python environment is unavailable." >&2
    exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
    echo "Run directory already exists: ${RUN_DIR}" >&2
    exit 2
fi
LOCK="${LOCK_DIR}/split_audit.json"
MANIFEST="${LOCK_DIR}/model_manifest.csv"
printf '%s  %s\n' \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' "${LOCK}" \
    'f3fbb6dddf346fd0086d693f783ec476ec802eb56586ed3225a2eda82f72551b' "${MANIFEST}" \
    | sha256sum --check --status

PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print("Allocated GPU:", torch.cuda.get_device_name(0), flush=True)'
"${PYTHON}" "${PROJECT_DIR}/scripts/run_replication.py" --run clear-lactyl-check --clear-lock "${LOCK}"
if [[ "${RUN_MODEL}" == hybridkla_architecture_aligned ]]; then
    printf '%s  %s\n' '73ddb128c441d7b213c4636e6bb6ce34501e82143c2b9bc06ec329d68a422993' "${HYBRID_ARCHIVE}" | sha256sum --check --status
fi

mkdir -p "${RUN_DIR}"
printf 'job_id=%s\nnode=%s\nmodel=%s\nseed=%s\ncommit=%s\nlock_sha256=%s\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${RUN_MODEL}" "${RUN_SEED}" "${EXPECTED_COMMIT}" \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' \
    > "${RUN_DIR}/job_metadata.txt"
"${PYTHON}" -m pip freeze > "${RUN_DIR}/requirements.freeze.txt"
nvidia-smi

if [[ "${RUN_MODEL}" == hybridkla_architecture_aligned ]]; then
    RESOURCE_DIR="${RUN_DIR}/hybridkla_resources"
    "${PYTHON}" "${HPC_DIR}/extract_hybridkla_resources.py" "${HYBRID_ARCHIVE}" "${RESOURCE_DIR}"
    "${PYTHON}" -u "${PROJECT_DIR}/scripts/run_hybridkla_architecture.py" \
        --benchmark-lock "${LOCK}" --resource-dir "${RESOURCE_DIR}" \
        --output-dir "${RUN_DIR}/results" --device cuda:0 --seed "${RUN_SEED}" \
        --esm-model facebook/esm2_t30_150M_UR50D \
        --esm-epochs 40 --lstm-epochs 30 --feature-epochs 100 --meta-epochs 50 \
        --esm-batch-size 32 --lstm-batch-size 32 --feature-batch-size 2048 \
        --patience 6 --development-only
else
    model_args=(--model-name Rostlab/prot_bert --encoder-layers 4)
    optimizer_args=(--optimizer adamw --learning-rate 2e-5 --weight-decay 0.01
        --scheduler linear --warmup-ratio 0.1)
    architecture="${RUN_MODEL}"
    if [[ "${RUN_MODEL}" == token_gated ]]; then
        model_args=(--model-name facebook/esm2_t33_650M_UR50D \
            --encoder-layers 33 --encoder-adaptation lora \
            --lora-rank 8 --lora-alpha 32 --lora-dropout 0.3)
    elif [[ "${RUN_MODEL}" == token_gated_protbert_bilstm ]]; then
        architecture=token_gated_bilstm
        optimizer_args+=(--head-learning-rate 1e-4)
    elif [[ "${RUN_MODEL}" == token_gated_protbert_full30_last4 ]]; then
        architecture=token_gated
        model_args=(--model-name Rostlab/prot_bert --encoder-layers 30 --train-last-encoder-layers 4)
        optimizer_args+=(--head-learning-rate 1e-4)
    elif [[ "${RUN_MODEL}" == token_gated_protbert || "${RUN_MODEL}" == token_gated_protbert_sgd || "${RUN_MODEL}" == token_gated_protbert_adamw || "${RUN_MODEL}" == token_gated_protbert_head_lr ]]; then
        architecture=token_gated
        if [[ "${RUN_MODEL}" == token_gated_protbert_sgd ]]; then
            optimizer_args=(--optimizer sgd --learning-rate 0.003
                --weight-decay 0 --scheduler none)
        elif [[ "${RUN_MODEL}" == token_gated_protbert_head_lr ]]; then
            optimizer_args+=(--head-learning-rate 1e-4)
        fi
    fi
    "${PYTHON}" -u "${PROJECT_DIR}/scripts/run_replication.py" \
        --run clear-lactyl-development --clear-lock "${LOCK}" \
        --architecture "${architecture}" "${model_args[@]}" \
        --epochs 30 --patience 5 --batch-size 4 --device cuda:0 \
        "${optimizer_args[@]}" \
        --positive-weighting none --monitor-metric AUPRC \
        --decision-threshold 0.5 --seed "${RUN_SEED}" \
        --save-models --output-dir "${RUN_DIR}/results"
fi

echo "CLEAR-Lactyl validation-only run completed: ${RUN_DIR}"
