#!/bin/bash
#SBATCH -J norm_eval
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/norm_eval_%j.out
#SBATCH -e logs/norm_eval_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

python scripts/experiments/eval_normal_sweep.py "$@"
