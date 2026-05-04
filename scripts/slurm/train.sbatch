#!/bin/bash
#SBATCH -J pref_train
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=32G
#SBATCH -t 0-01:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

# Usage:
#   sbatch scripts/slurm/train.sbatch sentence-transformers/sentence-t5-large

MODEL=${1:-"sentence-transformers/sentence-t5-large"}

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

python scripts/train.py --model "$MODEL" "$@"
