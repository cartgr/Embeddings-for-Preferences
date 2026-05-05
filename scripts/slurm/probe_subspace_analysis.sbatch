#!/bin/bash
#SBATCH -J subspace
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o logs/subspace_%j.out
#SBATCH -e logs/subspace_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

python scripts/experiments/probe_subspace_analysis.py "$@"
