#!/usr/bin/env bash
#SBATCH --job-name=project-stack-ablate
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-8%2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURMD_NODENAME:-}" &&
   "${SLURM_JOB_PARTITION:-}" == cvlab-a5000 ]] || {
    echo "Compute-node Slurm allocation required." >&2; exit 2;
}
HPC_DIR="$(pwd -P)"
[[ -f "${HPC_DIR}/project_clear_stack_ablations.sbatch" ]] || {
    echo "Submit from the hpc directory." >&2; exit 2;
}
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
LOCK_DIR="${LOCK_DIR:-${PROJECT_DIR}/outputs/clear_lactyl_locked}"
REFERENCE_DIR="${REFERENCE_DIR:-${REPO_DIR}/../MM-Kla-stack/experiment/pcbert_kla_clean/outputs/hpc_clear_lactyl_development/mmkla_positional_stack/seed_42/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/hpc_clear_stack_ablations}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the full repository commit}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" &&
   -z "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]] || {
    echo "Repository differs from expected clean commit." >&2; exit 2;
}
[[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/.ready" ]] || exit 2
printf '%s  %s\n' \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' "${LOCK_DIR}/split_audit.json" \
    'f3fbb6dddf346fd0086d693f783ec476ec802eb56586ed3225a2eda82f72551b' "${LOCK_DIR}/model_manifest.csv" \
    | sha256sum --check --status
[[ -f "${REFERENCE_DIR}/report.json" && -f "${REFERENCE_DIR}/oof_branch_scores.npz" &&
   -f "${REFERENCE_DIR}/validation_predictions.csv" ]] || {
    echo "Saved positional OOF reference is incomplete." >&2; exit 2;
}

modes=(none cls_pool mean_pool max_pool concat_fusion scalar_gate no_gated_fusion no_residual frozen_encoder)
task="${SLURM_ARRAY_TASK_ID:?Array task required}"
[[ "${task}" =~ ^[0-9]+$ && "${task}" -lt 9 ]] || exit 2
mode="${modes[task]}"
run_dir="${OUTPUT_ROOT}/${mode}/seed_42"
[[ ! -e "${run_dir}" ]] || { echo "Existing run: ${run_dir}" >&2; exit 2; }
mkdir -p "${run_dir}"
printf 'job_id=%s\nnode=%s\ncommit=%s\nmode=%s\nseed=42\nreference=%s\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${EXPECTED_COMMIT}" "${mode}" \
    "${REFERENCE_DIR}" > "${run_dir}/job_metadata.txt"
"${VENV_DIR}/bin/python" -m pip freeze > "${run_dir}/requirements.freeze.txt"
"${VENV_DIR}/bin/python" -u "${PROJECT_DIR}/scripts/run_clear_stack_ablation.py" \
    --clear-lock "${LOCK_DIR}/split_audit.json" \
    --positional-reference-dir "${REFERENCE_DIR}" \
    --output-dir "${run_dir}/results" --ablation "${mode}" \
    --device cuda:0 --seed 42 --epochs 30 --patience 5
