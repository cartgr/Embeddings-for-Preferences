#!/bin/bash
#SBATCH -J cluster_coh
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o logs/cluster_coh_%j.out
#SBATCH -e logs/cluster_coh_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

python scripts/experiments/cluster_coherence.py
