#!/bin/bash
#SBATCH -J ce
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o logs/ce_%j.out
#SBATCH -e logs/ce_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

python scripts/experiments/classify_errors.py "$@"
