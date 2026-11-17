#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" || "${SLURM_JOB_PARTITION:-}" != cvlab-a5000 ]]; then
    echo "Refusing to evaluate outside a Slurm compute allocation." >&2
    exit 2
fi

HPC_DIR="$(pwd -P)"
[[ -f "${HPC_DIR}/run_clear_heldout.sh" ]] || { echo "Submit from the hpc directory." >&2; exit 2; }
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
LOCK="${LOCK_DIR:-${PROJECT_DIR}/outputs/clear_lactyl_locked}/split_audit.json"
SAVED_ROOT="${SAVED_ROOT:-${PROJECT_DIR}/outputs/hpc_clear_lactyl_development}"
STACK_REFERENCE_REPORT="${STACK_REFERENCE_REPORT:-${REPO_DIR}/../MM-Kla-stack/experiment/pcbert_kla_clean/outputs/hpc_clear_lactyl_development/mmkla_positional_stack/seed_42/results/report.json}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the full repository commit}"
RUN_MODE="${RUN_MODE:?Set RUN_MODE to saved or stacked}"
RUN_DIR="${PROJECT_DIR}/outputs/hpc_clear_lactyl_final/${RUN_MODE}/seed_42"

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "Expected a full commit hash." >&2; exit 2; }
[[ "${RUN_MODE}" == saved || "${RUN_MODE}" == stacked ]] || { echo "Invalid run mode." >&2; exit 2; }
if [[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ||
      -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]]; then
    echo "Repository is not the expected clean commit." >&2
    exit 2
fi
[[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/.ready" ]] || {
    echo "Pinned HPC Python environment is unavailable." >&2; exit 2;
}
[[ ! -e "${RUN_DIR}" ]] || { echo "Run directory exists: ${RUN_DIR}" >&2; exit 2; }
printf '%s  %s\n' \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' "${LOCK}" \
    'f3fbb6dddf346fd0086d693f783ec476ec802eb56586ed3225a2eda82f72551b' "$(dirname "${LOCK}")/model_manifest.csv" \
    | sha256sum --check --status
if [[ "${RUN_MODE}" == stacked ]]; then
    [[ -f "${STACK_REFERENCE_REPORT}" ]] || { echo "Prior two-branch report is missing." >&2; exit 2; }
else
    for model in baseline token_gated_protbert_head_lr hybridkla_architecture_aligned deepkla_aligned; do
        [[ -d "${SAVED_ROOT}/${model}/seed_42/results" ]] || {
            echo "Missing selected checkpoint set: ${model}" >&2; exit 2;
        }
    done
fi

PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print("Allocated GPU:", torch.cuda.get_device_name(0), flush=True)'
mkdir -p "${RUN_DIR}"
printf 'job_id=%s\nnode=%s\nmode=%s\nseed=42\ncommit=%s\nlock_sha256=%s\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${RUN_MODE}" "${EXPECTED_COMMIT}" \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' \
    > "${RUN_DIR}/job_metadata.txt"
"${PYTHON}" -m pip freeze > "${RUN_DIR}/requirements.freeze.txt"
nvidia-smi

if [[ "${RUN_MODE}" == stacked ]]; then
    "${PYTHON}" -u "${PROJECT_DIR}/scripts/run_clear_stacked.py" \
        --clear-lock "${LOCK}" --output-dir "${RUN_DIR}/results" \
        --device cuda:0 --seed 42 --epochs 30 --patience 5 \
        --score-internal-test --reference-report "${STACK_REFERENCE_REPORT}"
else
    "${PYTHON}" -u "${PROJECT_DIR}/scripts/evaluate_clear_saved.py" \
        --clear-lock "${LOCK}" --saved-root "${SAVED_ROOT}" \
        --output-dir "${RUN_DIR}/results" --device cuda:0
fi

echo "CLEAR-Lactyl held-out evaluation completed: ${RUN_DIR}"
