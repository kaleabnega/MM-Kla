#!/usr/bin/env bash
#SBATCH --job-name=project-trans-ens
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURMD_NODENAME:-}" &&
   "${SLURM_JOB_PARTITION:-}" == cvlab-a5000 ]] || {
    echo "Refusing to train outside a Slurm compute allocation." >&2; exit 2;
}
HPC_DIR="$(pwd -P)"
[[ -f "${HPC_DIR}/project_transkla_clear_ensemble.sbatch" ]] || {
    echo "Submit from the hpc directory." >&2; exit 2;
}
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
TRANSKLA_DIR="${TRANSKLA_DIR:-${REPO_DIR}/../TransKla-source}"
LOCK="${PROJECT_DIR}/outputs/clear_lactyl_locked/split_audit.json"
PROTEOME="${PROJECT_DIR}/downloaded-dataset/UP000005640_9606.fasta.gz"
RUN_DIR="${PROJECT_DIR}/outputs/hpc_clear_lactyl_final/transkla/seed_42/ensemble"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the full pushed commit}"

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" &&
   -z "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]] || {
    echo "Project checkout differs from expected clean commit." >&2; exit 2;
}
[[ "$(git -C "${TRANSKLA_DIR}" rev-parse HEAD)" == 15ab472180cab74682f56f9f018b844cec264cc6 &&
   -z "$(git -C "${TRANSKLA_DIR}" status --porcelain --untracked-files=no)" ]] || {
    echo "Author TransKla checkout differs from pinned commit." >&2; exit 2;
}
[[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/.ready" ]] || {
    echo "Pinned HPC Python environment unavailable." >&2; exit 2;
}
[[ ! -e "${RUN_DIR}" ]] || { echo "Run directory exists: ${RUN_DIR}" >&2; exit 2; }
printf '%s  %s\n' \
    '3f0ac7070bbc94b82b08c063c9884a7d022974fcff0ca45012bd0b1483d75581' "${TRANSKLA_DIR}/train_TransKla.py" \
    '7bf570becb2fc2dcf3878d20d7bb8ed276dd7ddc8658c23b664a0a39b8388d00' "${LOCK}" \
    'f3fbb6dddf346fd0086d693f783ec476ec802eb56586ed3225a2eda82f72551b' "$(dirname "${LOCK}")/model_manifest.csv" \
    '64299d01e8f41fb6ef03cdf7ec051d3d4e5f041003b6b0fa75b09d439af79dce' "${PROTEOME}" \
    | sha256sum --check --status

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${VENV_DIR}/bin/python" -m unittest discover \
    -s "${PROJECT_DIR}/tests" -p test_transkla_clear_ensemble.py -v
"${VENV_DIR}/bin/python" -m unittest discover \
    -s "${PROJECT_DIR}/tests" -p test_transkla_clear.py -v
mkdir -p "${RUN_DIR}"
printf 'job_id=%s\nnode=%s\ncommit=%s\nseed=42\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${EXPECTED_COMMIT}" \
    > "${RUN_DIR}/job_metadata.txt"
"${VENV_DIR}/bin/python" -m pip freeze > "${RUN_DIR}/requirements.freeze.txt"
"${VENV_DIR}/bin/python" -u "${PROJECT_DIR}/scripts/run_transkla_clear_ensemble.py" \
    --author-source "${TRANSKLA_DIR}/train_TransKla.py" \
    --lock "${LOCK}" --proteome "${PROTEOME}" \
    --output-dir "${RUN_DIR}/results" --device cuda:0 \
    --seed 42 --epochs 40 --patience 10 --batch-size 32
