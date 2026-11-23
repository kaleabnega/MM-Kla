#!/usr/bin/env bash
#SBATCH --job-name=project-clear-standalone
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3%2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
bash "$(pwd -P)/run_clear_standalone_branches.sh"
