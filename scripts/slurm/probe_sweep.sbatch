#!/bin/bash
# Per-topic probe sweep (one of: metric, cosine, metric_bias).
# Usage: sbatch --export=ALL,SCORER=metric scripts/slurm/probe_sweep.sbatch
#   or: sbatch scripts/slurm/probe_sweep.sbatch metric
#SBATCH -J probe_sweep
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/probe_sweep_%j.out
#SBATCH -e logs/probe_sweep_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

SCORER="${SCORER:-${1:-metric}}"
python scripts/experiments/probe_sweep.py --scorer "$SCORER"
