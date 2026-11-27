#!/usr/bin/env bash
#SBATCH --job-name=project-attentionkla-clear
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" ||
      "${SLURM_JOB_PARTITION:-}" != cvlab-a5000 ]]; then
    echo "Refusing to train outside a Slurm compute allocation." >&2
    exit 2
fi

HPC_DIR="$(pwd -P)"
[[ -f "${HPC_DIR}/project_attentionkla_clear_development.sbatch" ]] || {
    echo "Submit from the hpc directory." >&2
    exit 2
}
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
LOCK="${PROJECT_DIR}/outputs/clear_lactyl_locked/split_audit.json"
MANIFEST="$(dirname "${LOCK}")/model_manifest.csv"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the full project commit}"
RUN_ROOT="${PROJECT_DIR}/outputs/hpc_clear_lactyl_final/attentionkla/seed_42"
RUN_DIR="${RUN_ROOT}/development"

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "Expected a full project commit hash." >&2
    exit 2
}
[[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" &&
   -z "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]] || {
    echo "Project checkout is not the expected clean commit." >&2
    exit 2
}
[[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/.ready" ]] || {
    echo "Pinned HPC Python environment is unavailable." >&2
    exit 2
}
[[ ! -e "${RUN_DIR}" ]] || {
    echo "Run directory already exists: ${RUN_DIR}" >&2
    exit 2
}
printf '%s  %s\n' \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' "${LOCK}" \
    'f3fbb6dddf346fd0086d693f783ec476ec802eb56586ed3225a2eda82f72551b' "${MANIFEST}" \
    | sha256sum --check --status

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${VENV_DIR}/bin/python" -m unittest discover \
    -s "${PROJECT_DIR}/tests" -p test_attentionkla_development.py -v
"${VENV_DIR}/bin/python" -c \
    'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print("GPU:", torch.cuda.get_device_name(0), flush=True)'
mkdir -p "${RUN_ROOT}"
printf 'job_id=%s\nnode=%s\nseed=42\nproject_commit=%s\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${EXPECTED_COMMIT}" \
    > "${RUN_ROOT}/job_metadata.txt"
"${VENV_DIR}/bin/python" -m pip freeze > "${RUN_ROOT}/requirements.freeze.txt"
"${VENV_DIR}/bin/python" -u "${PROJECT_DIR}/scripts/run_attentionkla_clear_development.py" \
    --lock "${LOCK}" --output-dir "${RUN_DIR}" --device cuda:0 \
    --seed 42 --epochs 15 --patience 2
