#!/bin/bash
#SBATCH -J rank_sweep
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/rank_%j.out
#SBATCH -e logs/rank_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

python scripts/experiments/rank_sweep.py \
    --base-model sentence-transformers/sentence-t5-xl \
    --output data/results/rank_sweep.json
