#!/usr/bin/env bash
set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURMD_NODENAME:-}" &&
   "${SLURM_JOB_PARTITION:-}" == cvlab-a5000 ]] || {
    echo "Compute-node Slurm allocation required." >&2; exit 2;
}
HPC_DIR="$(pwd -P)"
[[ -f "${HPC_DIR}/run_clear_standalone_branches.sh" ]] || {
    echo "Submit from the hpc directory." >&2; exit 2;
}
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
LOCK_DIR="${LOCK_DIR:-${PROJECT_DIR}/outputs/clear_lactyl_locked}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/hpc_clear_standalone_branches}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the full repository commit}"
task="${SLURM_ARRAY_TASK_ID:?Array task required}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ && "${task}" =~ ^[0-3]$ ]] || exit 2
[[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" &&
   -z "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]] || {
    echo "Repository differs from expected clean commit." >&2; exit 2;
}
[[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/.ready" ]] || exit 2
printf '%s  %s\n' \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' "${LOCK_DIR}/split_audit.json" \
    'f3fbb6dddf346fd0086d693f783ec476ec802eb56586ed3225a2eda82f72551b' "${LOCK_DIR}/model_manifest.csv" \
    | sha256sum --check --status

modes=(protbert_sequence_only descriptors_27_only protbert_sequence_only descriptors_27_only)
seeds=(42 42 123 123)
mode="${modes[task]}"
seed="${seeds[task]}"
if [[ "${seed}" == 42 ]]; then
    REFERENCE_DIR="${REFERENCE_DIR:-${REPO_DIR}/../MM-Kla-stack/experiment/pcbert_kla_clean/outputs/hpc_clear_lactyl_development/mmkla_positional_stack/seed_42/results}"
else
    REFERENCE_DIR="${REFERENCE_DIR:-${PROJECT_DIR}/outputs/hpc_clear_lactyl_development/mmkla_positional_stack/seed_123/results}"
fi
[[ -f "${REFERENCE_DIR}/report.json" && -f "${REFERENCE_DIR}/oof_branch_scores.npz" &&
   -f "${REFERENCE_DIR}/validation_predictions.csv" ]] || {
    echo "Saved positional reference is incomplete: ${REFERENCE_DIR}" >&2; exit 2;
}
PYTHONPATH="${PROJECT_DIR}/src" "${VENV_DIR}/bin/python" -m unittest discover \
    -s "${PROJECT_DIR}/tests" -p 'test_stacked_development.py'
run_dir="${OUTPUT_ROOT}/${mode}/seed_${seed}"
[[ ! -e "${run_dir}" ]] || {
    echo "Existing run: ${run_dir}" >&2; exit 2;
}
mkdir -p "${run_dir}"
printf 'job_id=%s\nnode=%s\ncommit=%s\nmode=%s\nseed=%s\nreference=%s\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${EXPECTED_COMMIT}" \
    "${mode}" "${seed}" "${REFERENCE_DIR}" > "${run_dir}/job_metadata.txt"
"${VENV_DIR}/bin/python" -m pip freeze > "${run_dir}/requirements.freeze.txt"
"${VENV_DIR}/bin/python" -u "${PROJECT_DIR}/scripts/run_clear_stack_extended_ablation.py" \
    --clear-lock "${LOCK_DIR}/split_audit.json" \
    --reference-dir "${REFERENCE_DIR}" --output-dir "${run_dir}/results" \
    --device cuda:0 --seed "${seed}" --epochs 30 --patience 5 \
    --ablation "${mode}"
