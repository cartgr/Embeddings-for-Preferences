#!/bin/bash
#SBATCH -J paired_test
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/paired_test_%j.out
#SBATCH -e logs/paired_test_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

python scripts/experiments/participant_paired_test.py "$@"
