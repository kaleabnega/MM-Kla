#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" || "${SLURM_JOB_PARTITION:-}" != cvlab-a5000 ]]; then
    echo "Refusing to train outside a Slurm compute allocation." >&2
    exit 2
fi

HPC_DIR="$(pwd -P)"
[[ -f "${HPC_DIR}/run_deepkla_clear.sh" ]] || { echo "Submit from the hpc directory." >&2; exit 2; }
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
LOCK="${LOCK_DIR:-${PROJECT_DIR}/outputs/clear_lactyl_locked}/split_audit.json"
RUN_SEED="${RUN_SEED:-42}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the full repository commit}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/hpc_clear_lactyl_development}"
RUN_DIR="${OUTPUT_ROOT}/deepkla_aligned/seed_${RUN_SEED}"

[[ "${RUN_SEED}" =~ ^[0-9]+$ && "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "Invalid seed or expected commit." >&2; exit 2;
}
if [[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ||
      -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]]; then
    echo "Repository is not the expected clean commit." >&2
    exit 2
fi
[[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/.ready" ]] || {
    echo "Pinned HPC Python environment is unavailable." >&2; exit 2;
}
[[ ! -e "${RUN_DIR}" ]] || { echo "Run directory already exists: ${RUN_DIR}" >&2; exit 2; }
printf '%s  %s\n' \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' "${LOCK}" \
    'f3fbb6dddf346fd0086d693f783ec476ec802eb56586ed3225a2eda82f72551b' "$(dirname "${LOCK}")/model_manifest.csv" \
    | sha256sum --check --status

PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print("Allocated GPU:", torch.cuda.get_device_name(0), flush=True)'
mkdir -p "${RUN_DIR}"
printf 'job_id=%s\nnode=%s\nseed=%s\ncommit=%s\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${RUN_SEED}" "${EXPECTED_COMMIT}" > "${RUN_DIR}/job_metadata.txt"
"${PYTHON}" -m pip freeze > "${RUN_DIR}/requirements.freeze.txt"
nvidia-smi
"${PYTHON}" -u "${PROJECT_DIR}/scripts/run_deepkla_clear.py" \
    --clear-lock "${LOCK}" --output-dir "${RUN_DIR}/results" \
    --device cuda:0 --seed "${RUN_SEED}" --epochs 30 --patience 5 --batch-size 64
