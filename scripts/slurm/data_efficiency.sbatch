#!/bin/bash
#SBATCH -J data_eff
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/data_eff_%j.out
#SBATCH -e logs/data_eff_%j.err

# Data efficiency curve: 11 datasets × 9 K values × 4 HP × 3 seeds = ~1188 trials.
# Each trial < 1 min; total ~4-6 h on one GPU.

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

python scripts/experiments/data_efficiency.py \
    --encoder sentence-transformers/sentence-t5-xl \
    --max-train 10000 \
    --max-val 3000 \
    --max-test 5000 \
    --output data/results/data_efficiency.json
