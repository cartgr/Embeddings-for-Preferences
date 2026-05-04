#!/bin/bash
#SBATCH -J download
#SBATCH -p serial_requeue
#SBATCH --mem=16G
#SBATCH -t 0-00:30
#SBATCH -o logs/download_%j.out
#SBATCH -e logs/download_%j.err

module load python/3.12.5-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

python scripts/data/01_download_eval.py
python scripts/data/02_download_issue_sources.py
