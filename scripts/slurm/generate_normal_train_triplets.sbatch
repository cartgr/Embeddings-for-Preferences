#!/bin/bash
# Generate 2k normal (non-engineered) counterfactual training triplets.
# OpenAI API only — no GPU needed; we still submit via slurm to keep the
# .env loading and run path consistent.
#SBATCH -J norm_gen
#SBATCH -p shared
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o logs/norm_gen_%j.out
#SBATCH -e logs/norm_gen_%j.err

module load python/3.12.5-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

# Load OPENAI_API_KEY etc.
if [ -f .env ]; then
    set -a; source .env; set +a
fi

python scripts/data/08_generate_normal_train_triplets.py --n 2000 "$@"
