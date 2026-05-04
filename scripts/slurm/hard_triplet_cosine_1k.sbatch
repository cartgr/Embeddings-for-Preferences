#!/bin/bash
#SBATCH -J hard_cos_1k
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/hard_cos_1k_%j.out
#SBATCH -e logs/hard_cos_1k_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

python scripts/experiments/hard_triplet_cosine.py \
    --encoders \
        sentence-transformers/sentence-t5-xl \
        intfloat/e5-large-v2 \
        BAAI/bge-large-en-v1.5 \
        sentence-transformers/all-mpnet-base-v2 \
    --hard-triplets data/processed/triplets/hard_eval_triplets_1k.jsonl \
    --output data/results/hard_triplet_cosine_1k.json
