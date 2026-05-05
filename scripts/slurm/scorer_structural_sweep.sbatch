#!/bin/bash
#SBATCH -J scorer_sweep
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/scorer_sweep_%j.out
#SBATCH -e logs/scorer_sweep_%j.err

# ~13 variants * 11 datasets * 5 seeds * 8 HP configs = ~5700 train
# runs. With resume, only the missing (variant, ds, seed) triples run;
# expect ~3-4h to add seeds 45+46 on top of existing 42-44.

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

python scripts/experiments/scorer_structural_sweep.py
