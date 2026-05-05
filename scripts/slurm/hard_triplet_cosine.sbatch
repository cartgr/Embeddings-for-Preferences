#!/bin/bash
#SBATCH -J hard_cos
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/hard_cos_%j.out
#SBATCH -e logs/hard_cos_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Base encoders for the §3 aggregate diagnostic table.
python scripts/experiments/hard_triplet_cosine.py \
    --encoders \
        sentence-transformers/sentence-t5-xl \
        intfloat/e5-large-v2 \
        BAAI/bge-large-en-v1.5 \
        sentence-transformers/all-mpnet-base-v2 \
    --output data/results/hard_triplet_cosine.json
