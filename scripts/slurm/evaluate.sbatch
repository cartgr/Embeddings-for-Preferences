#!/bin/bash
#SBATCH -J pref_eval
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=32G
#SBATCH -t 0-00:30
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err

# Usage:
#   sbatch scripts/slurm/evaluate.sbatch sentence-transformers/sentence-t5-large
#   sbatch scripts/slurm/evaluate.sbatch data/models/best/sentence_t5_large

MODEL=${1:-"sentence-transformers/sentence-t5-large"}

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

python scripts/evaluate.py --model "$MODEL" --split test
