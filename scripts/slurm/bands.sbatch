#!/bin/bash
#SBATCH -J bands
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/bands_%j.out
#SBATCH -e logs/bands_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs data/results/bands

python scripts/plotting/fig2_bands.py \
    --encoders \
        sentence-transformers/sentence-t5-xl \
        intfloat/e5-large-v2 \
        BAAI/bge-large-en-v1.5 \
        sentence-transformers/all-mpnet-base-v2 \
    --n-bands 5 \
    --output figures/fig2_bands.pdf
