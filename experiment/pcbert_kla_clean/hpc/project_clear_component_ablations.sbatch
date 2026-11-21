#!/usr/bin/env bash
#SBATCH --job-name=project-ablate-clear
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-23%2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURMD_NODENAME:-}" &&
   "${SLURM_JOB_PARTITION:-}" == cvlab-a5000 ]] || {
    echo "Compute-node Slurm allocation required." >&2; exit 2;
}
HPC_DIR="$(pwd -P)"
[[ -f "${HPC_DIR}/project_clear_component_ablations.sbatch" ]] || {
    echo "Submit from the hpc directory." >&2; exit 2;
}
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
LOCK_DIR="${LOCK_DIR:-${PROJECT_DIR}/outputs/clear_lactyl_locked}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/hpc_clear_component_ablations}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the full repository commit}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" &&
   -z "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]] || {
    echo "Repository differs from the expected clean commit." >&2; exit 2;
}
[[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/.ready" ]] || exit 2
printf '%s  %s\n' \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' "${LOCK_DIR}/split_audit.json" \
    'f3fbb6dddf346fd0086d693f783ec476ec802eb56586ed3225a2eda82f72551b' "${LOCK_DIR}/model_manifest.csv" \
    | sha256sum --check --status

modes=(none cls_pool mean_pool max_pool concat_fusion scalar_gate no_residual frozen_encoder)
seeds=(42 123 777)
task="${SLURM_ARRAY_TASK_ID:?Array task required}"
[[ "${task}" =~ ^[0-9]+$ && "${task}" -lt 24 ]] || exit 2
mode="${modes[task % 8]}"
seed="${seeds[task / 8]}"
run_dir="${OUTPUT_ROOT}/${mode}/seed_${seed}"
[[ ! -e "${run_dir}" ]] || { echo "Existing run: ${run_dir}" >&2; exit 2; }
mkdir -p "${run_dir}"
printf 'job_id=%s\nnode=%s\ncommit=%s\nmode=%s\nseed=%s\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${EXPECTED_COMMIT}" "${mode}" "${seed}" \
    > "${run_dir}/job_metadata.txt"
"${VENV_DIR}/bin/python" -m pip freeze > "${run_dir}/requirements.freeze.txt"
extra=(--ablation "${mode}")
if [[ "${mode}" == frozen_encoder ]]; then
    extra=(--ablation none --freeze-encoder)
fi
"${VENV_DIR}/bin/python" -u "${PROJECT_DIR}/scripts/run_replication.py" \
    --run clear-lactyl-development --clear-lock "${LOCK_DIR}/split_audit.json" \
    --architecture token_gated --model-name Rostlab/prot_bert \
    --encoder-layers 4 --add-cksaap --fusion-dim 256 --attention-dim 256 \
    --arch-dropout 0.2 --epochs 30 --patience 5 --batch-size 4 \
    --optimizer adamw --learning-rate 2e-5 --head-learning-rate 1e-4 \
    --weight-decay 0.01 --scheduler linear --warmup-ratio 0.1 \
    --positive-weighting none --monitor-metric AUPRC \
    --decision-threshold 0.5 --device cuda:0 --seed "${seed}" \
    --save-models --output-dir "${run_dir}/results" "${extra[@]}"
