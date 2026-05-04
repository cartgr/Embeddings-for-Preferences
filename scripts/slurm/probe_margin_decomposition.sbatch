#!/bin/bash
#SBATCH -J pmd
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/pmd_%j.out
#SBATCH -e logs/pmd_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

python scripts/experiments/probe_margin_decomposition.py "$@"
