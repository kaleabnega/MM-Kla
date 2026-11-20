#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" ||
      "${SLURM_JOB_PARTITION:-}" != cvlab-a5000 ]]; then
    echo "Refusing to run outside a Slurm compute allocation." >&2
    exit 2
fi

HPC_DIR="$(pwd -P)"
[[ -f "${HPC_DIR}/run_transkla_clear.sh" ]] || {
    echo "Submit from the hpc directory." >&2; exit 2;
}
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
TRANSKLA_DIR="${TRANSKLA_DIR:-${REPO_DIR}/../TransKla-source}"
LOCK="${PROJECT_DIR}/outputs/clear_lactyl_locked/split_audit.json"
PROTEOME="${PROJECT_DIR}/downloaded-dataset/UP000005640_9606.fasta.gz"
MODE="${MODE:?Set MODE to develop or heldout}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the full project commit}"
SOURCE_COMMIT=15ab472180cab74682f56f9f018b844cec264cc6
RUN_ROOT="${PROJECT_DIR}/outputs/hpc_clear_lactyl_final/transkla/seed_42"
RUN_DIR="${RUN_ROOT}/${MODE}"

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "Expected a full project commit hash." >&2; exit 2;
}
[[ "${MODE}" == develop || "${MODE}" == heldout ]] || {
    echo "Invalid mode: ${MODE}" >&2; exit 2;
}
[[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" &&
   -z "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]] || {
    echo "Project checkout is not the expected clean commit." >&2; exit 2;
}
[[ "$(git -C "${TRANSKLA_DIR}" rev-parse HEAD)" == "${SOURCE_COMMIT}" &&
   -z "$(git -C "${TRANSKLA_DIR}" status --porcelain --untracked-files=no)" ]] || {
    echo "Author TransKla checkout is not the pinned commit." >&2; exit 2;
}
[[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/.ready" ]] || {
    echo "Pinned HPC Python environment is unavailable." >&2; exit 2;
}
[[ ! -e "${RUN_DIR}" ]] || {
    echo "Run directory already exists: ${RUN_DIR}" >&2; exit 2;
}
printf '%s  %s\n' \
    '3f0ac7070bbc94b82b08c063c9884a7d022974fcff0ca45012bd0b1483d75581' "${TRANSKLA_DIR}/train_TransKla.py" \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' "${LOCK}" \
    'f3fbb6dddf346fd0086d693f783ec476ec802eb56586ed3225a2eda82f72551b' "$(dirname "${LOCK}")/model_manifest.csv" \
    '64299d01e8f41fb6ef03cdf7ec051d3d4e5f041003b6b0fa75b09d439af79dce' "${PROTEOME}" \
    | sha256sum --check --status

"${VENV_DIR}/bin/python" -c \
    'import torch, yaml; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print("GPU:", torch.cuda.get_device_name(0), flush=True)'
mkdir -p "${RUN_ROOT}"
printf 'job_id=%s\nnode=%s\nmode=%s\nseed=42\nproject_commit=%s\nauthor_commit=%s\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${MODE}" \
    "${EXPECTED_COMMIT}" "${SOURCE_COMMIT}" > "${RUN_ROOT}/${MODE}_job_metadata.txt"
"${VENV_DIR}/bin/python" -m pip freeze > "${RUN_ROOT}/${MODE}_requirements.freeze.txt"

ARGS=(--mode "${MODE}" --author-source "${TRANSKLA_DIR}/train_TransKla.py"
      --lock "${LOCK}" --proteome "${PROTEOME}" --output-dir "${RUN_DIR}"
      --device cuda:0 --seed 42)
if [[ "${MODE}" == develop ]]; then
    ARGS+=(--epochs 40 --patience 10 --batch-size 32)
else
    [[ -f "${RUN_ROOT}/develop/report.json" &&
       -f "${RUN_ROOT}/develop/transkla_best.pt" ]] || {
        echo "Validated TransKla checkpoint is missing." >&2; exit 2;
    }
    ARGS+=(--development-dir "${RUN_ROOT}/develop")
fi
"${VENV_DIR}/bin/python" -u "${PROJECT_DIR}/scripts/run_transkla_clear.py" "${ARGS[@]}"
echo "TransKla ${MODE} completed: ${RUN_DIR}"
