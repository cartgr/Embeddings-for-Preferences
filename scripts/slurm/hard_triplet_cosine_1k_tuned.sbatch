#!/bin/bash
#SBATCH -J hard_cos_tuned
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/hard_cos_tuned_%j.out
#SBATCH -e logs/hard_cos_tuned_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Evaluate the 4 tuned LoRA adapters on the 875-triplet hard eval set.
# Appends to the same JSON alongside base-encoder numbers (different keys).
python scripts/experiments/hard_triplet_cosine.py \
    --encoders \
        data/models/best/sentence_transformers_sentence_t5_xl/targeted_sweep/lr1p25eem4_n750_r16_a48 \
        data/models/best/intfloat_e5_large_v2/hard_recipe/lr5eep5_n1000_r16_a48 \
        data/models/best/BAAI_bge_large_en_v1_5/hard_recipe/lr1p25eep4_n500_r16_a48 \
        data/models/best/sentence_transformers_all_mpnet_base_v2/hard_recipe/lr1p25eep4_n1000_r16_a48 \
    --hard-triplets data/processed/triplets/hard_eval_triplets_1k.jsonl \
    --output data/results/hard_triplet_cosine_1k.json
