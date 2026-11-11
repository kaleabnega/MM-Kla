#!/usr/bin/env bash
#SBATCH --job-name=kla-env
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" || "${SLURM_JOB_PARTITION:-}" != cvlab-a5000 ]]; then
    echo "Refusing to run outside a Slurm allocation." >&2
    exit 2
fi

HPC_DIR="$(pwd -P)"
if [[ ! -f "${HPC_DIR}/setup_environment.sbatch" ]]; then
    echo "Submit from the hpc directory or pass --chdir to sbatch." >&2
    exit 2
fi
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv-hpc-cu118}"
PYTHON_FOR_VENV="${PYTHON_FOR_VENV:-${HOME}/.miniconda3/bin/python}"

if [[ -e "${VENV_DIR}/.ready" ]]; then
    echo "Environment already prepared: ${VENV_DIR}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_FOR_VENV}" ]]; then
    echo "Python for environment creation is unavailable: ${PYTHON_FOR_VENV}" >&2
    exit 2
fi

"${PYTHON_FOR_VENV}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --timeout 120 --retries 5 \
    'torch==2.7.1+cu118' --index-url https://download.pytorch.org/whl/cu118
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt" 'peft>=0.11,<1'
"${VENV_DIR}/bin/python" -m pip check
"${VENV_DIR}/bin/python" -m pip freeze > "${VENV_DIR}/requirements.freeze.txt"
"${VENV_DIR}/bin/python" -c 'import torch, transformers, peft; assert torch.__version__.startswith("2.7.1+cu118"); assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Allocated GPU unavailable"; x = torch.ones(1, device="cuda"); assert x.item() == 1; print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "GPU", torch.cuda.get_device_name(0))'
touch "${VENV_DIR}/.ready"
echo "Environment prepared: ${VENV_DIR}"
