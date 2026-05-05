#!/bin/bash
#SBATCH -J fig5
#SBATCH -p shared
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o logs/fig5_%j.out
#SBATCH -e logs/fig5_%j.err

module load python/3.12.5-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

python scripts/plotting/fig5_data_efficiency.py
