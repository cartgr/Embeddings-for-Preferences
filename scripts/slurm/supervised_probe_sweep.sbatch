#!/bin/bash
#SBATCH -J sup_sweep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/sup_sweep_%j.out
#SBATCH -e logs/sup_sweep_%j.err

# Supervised embedding sweep: 2 methods (PSD r=20, two-tower MLP h=256)
# × 2 encoders (base, tuned) × 8 hyperparam configs × 11 datasets = 352 trials.
# Each trial < 1 min; total ~3-4 h on one GPU.

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

python scripts/experiments/supervised_probe_sweep.py \
    --base-model sentence-transformers/sentence-t5-xl \
    --tuned-model data/models/best/sentence_transformers_sentence_t5_xl/targeted_sweep/lr1p25eem4_n750_r16_a48 \
    --max-train 10000 \
    --max-test 5000 \
    --output data/results/supervised_sweep.json
