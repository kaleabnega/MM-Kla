#!/usr/bin/env bash
#SBATCH --job-name=kla-env
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" || "${SLURM_JOB_PARTITION:-}" != cvlab-a5000 ]]; then
    echo "Refusing to run outside a Slurm allocation." >&2
    exit 2
fi

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HPC_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/mm-kla-hpc}"

if [[ -e "${VENV_DIR}/.ready" ]]; then
    echo "Environment already prepared: ${VENV_DIR}" >&2
    exit 2
fi

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt" 'peft>=0.11,<1'
"${VENV_DIR}/bin/python" -m pip freeze > "${VENV_DIR}/requirements.freeze.txt"
"${VENV_DIR}/bin/python" -c 'import torch, transformers, peft; print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)'
touch "${VENV_DIR}/.ready"
echo "Environment prepared: ${VENV_DIR}"
