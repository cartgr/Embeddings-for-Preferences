#!/bin/bash
#SBATCH -J sparsecl
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 00:40:00
#SBATCH -o logs/sparsecl_%j.out
#SBATCH -e logs/sparsecl_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

python scripts/experiments/eval_sparsecl_bge.py \
    --model-id SparseCL/BGE-SparseCL-arguana \
    --hard-triplets data/processed/triplets/hard_eval_triplets_1k.jsonl \
    --output data/results/hard_triplet_cosine_1k.json
