#!/usr/bin/env bash
#SBATCH --job-name=project-stack-new42
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4%2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
RUN_SEED=42 bash "$(pwd -P)/run_clear_stack_remaining.sh"
