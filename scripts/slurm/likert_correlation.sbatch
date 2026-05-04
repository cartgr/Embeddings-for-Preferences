#!/bin/bash
#SBATCH -J likert
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/likert_%j.out
#SBATCH -e logs/likert_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

python scripts/experiments/likert_correlation.py "$@"
