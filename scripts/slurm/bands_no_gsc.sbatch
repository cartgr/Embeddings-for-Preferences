#!/bin/bash
# fig2_bands.py over the 8 non-GSC datasets (3 Remesh + 5 Polis).
# Useful diagnostic — GSC's pseudo-Likert triplets always have one
# approved and one not, so they dilute the per-band approval rate.
#SBATCH -J bands_ng
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/bands_ng_%j.out
#SBATCH -e logs/bands_ng_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

python scripts/plotting/fig2_bands.py \
    --encoders \
        sentence-transformers/sentence-t5-xl \
        intfloat/e5-large-v2 \
        BAAI/bge-large-en-v1.5 \
        sentence-transformers/all-mpnet-base-v2 \
    --n-bands 5 \
    --exclude-datasets gsc_abortion_gen gsc_abortion_val gsc_chatbot_gen \
    --cache-suffix _no-gsc \
    --output figures/fig2_bands_no_gsc.pdf
