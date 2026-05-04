#!/bin/bash
#SBATCH -J err_decomp
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o logs/err_decomp_%j.out
#SBATCH -e logs/err_decomp_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

python scripts/experiments/error_decomposition.py \
    --base-model sentence-transformers/sentence-t5-xl \
    --tuned-model data/models/best/sentence_transformers_sentence_t5_xl/targeted_sweep/lr1p25eem4_n750_r16_a48 \
    --output data/results/error_decomposition.json
