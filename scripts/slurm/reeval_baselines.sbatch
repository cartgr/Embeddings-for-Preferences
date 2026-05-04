#!/bin/bash
# Re-evaluate every base baseline on all 11 datasets + 875 hard triplets,
# applying each model's intended query/passage encoding format. See
# scripts/experiments/reeval_baselines.py for the model list and dispatch.
#SBATCH -J reeval
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/reeval_%j.out
#SBATCH -e logs/reeval_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

# Load API keys (OPENAI_API_KEY, VOYAGE_API_KEY, ANTHROPIC_API_KEY) from .env
# so OpenAI / Voyage baselines can authenticate on the compute node.
if [ -f .env ]; then
    set -a; source .env; set +a
fi

python scripts/experiments/reeval_baselines.py --cleanup-after-new "$@"
