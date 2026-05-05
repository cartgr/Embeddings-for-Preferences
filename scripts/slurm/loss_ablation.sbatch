#!/bin/bash
#SBATCH -J loss_abl
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/loss_abl_%A_%a.out
#SBATCH -e logs/loss_abl_%A_%a.err
#SBATCH -a 0-1

# Loss function ablation on ST5-XL: Bradley-Terry (paper default) vs InfoNCE.
# All other hyperparameters fixed at the best BT config.
# Output: data/models/best/loss_ablation/{bradley_terry,infonce}/

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

LOSSES=(bradley_terry infonce)
TASK=$SLURM_ARRAY_TASK_ID
LOSS="${LOSSES[$TASK]}"

OUT="data/models/best/loss_ablation/${LOSS}"
echo "Task $TASK: loss=$LOSS  ->  $OUT"

python scripts/train.py \
    --model sentence-transformers/sentence-t5-xl \
    --condition political \
    --selection diverse_2000 \
    --gen-model sonnet_standalone_prompt \
    --output-dir "$OUT" \
    --lr 1.25e-4 \
    --max-steps 46 \
    --batch-size 16 \
    --lora-r 16 \
    --lora-alpha 48 \
    --loss $LOSS \
    --hard-triplets data/processed/triplets/hard_train_triplets_10000.jsonl \
    --hard-ratio 1.0 \
    --hard-cap 750 \
    --seed 42
