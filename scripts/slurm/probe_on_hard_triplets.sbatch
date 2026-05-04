#!/bin/bash
#SBATCH -J probe_hard
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o logs/probe_hard_%j.out
#SBATCH -e logs/probe_hard_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

python scripts/experiments/probe_on_hard_triplets.py \
    --encoder sentence-transformers/sentence-t5-xl \
    --hard-triplets data/processed/triplets/hard_eval_triplets_1k.jsonl \
    --probe-dir data/results/psd_probe_weights \
    --output data/results/probe_on_hard_triplets.json
