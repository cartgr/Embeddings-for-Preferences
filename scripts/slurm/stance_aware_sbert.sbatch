#!/bin/bash
#SBATCH -J stance_aware
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 00:40:00
#SBATCH -o logs/stance_aware_%j.out
#SBATCH -e logs/stance_aware_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# StanceAware-SBERT is a LoRA adapter on all-mpnet-base-v2, so it needs
# custom loading (base model + PeftModel.from_pretrained). The dedicated
# script handles this; results get appended to hard_triplet_cosine_1k.json.
python scripts/experiments/eval_stance_aware_sbert.py \
    --hard-triplets data/processed/triplets/hard_eval_triplets_1k.jsonl \
    --output data/results/hard_triplet_cosine_1k.json
