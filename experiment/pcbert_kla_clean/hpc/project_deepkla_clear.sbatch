#!/usr/bin/env bash
#SBATCH --job-name=project-deepkla-clear
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
HPC_DIR="$(pwd -P)"
[[ -f "${HPC_DIR}/run_deepkla_clear.sh" ]] || { echo "Submit from the hpc directory." >&2; exit 2; }
bash "${HPC_DIR}/run_deepkla_clear.sh"
