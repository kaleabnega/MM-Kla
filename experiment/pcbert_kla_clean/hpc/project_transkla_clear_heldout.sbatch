#!/usr/bin/env bash
#SBATCH --job-name=project-transkla-test
#SBATCH --partition=cvlab-a5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
MODE=heldout bash "$(pwd -P)/run_transkla_clear.sh"
