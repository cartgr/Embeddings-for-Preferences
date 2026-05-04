#!/bin/bash
#SBATCH -J lora_rank
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40
#SBATCH --exclude=holygpu7c0705
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/lora_rank_%A_%a.out
#SBATCH -e logs/lora_rank_%A_%a.err
#SBATCH -a 0-4

# LoRA rank ablation on ST5-XL. Sweep r ∈ {4, 8, 16, 32, 64} with everything
# else held at the paper's best config (lr=1.25e-4, n_hard=750, alpha=48).
# Output: data/models/best/lora_rank_ablation/r{4,8,16,32,64}/

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

RANKS=(4 8 16 32 64)
TASK=$SLURM_ARRAY_TASK_ID
R="${RANKS[$TASK]}"
ALPHA=$((R * 3))   # keep alpha/rank ratio at 3 (matches best config: r=16, alpha=48)

OUT="data/models/best/lora_rank_ablation/r${R}"
echo "Task $TASK:  lora_r=$R  lora_alpha=$ALPHA  ->  $OUT"

python scripts/train.py \
    --model sentence-transformers/sentence-t5-xl \
    --condition political \
    --selection diverse_2000 \
    --gen-model sonnet_standalone_prompt \
    --output-dir "$OUT" \
    --lr 1.25e-4 \
    --max-steps 46 \
    --batch-size 16 \
    --lora-r $R \
    --lora-alpha $ALPHA \
    --hard-triplets data/processed/triplets/hard_train_triplets_10000.jsonl \
    --hard-ratio 1.0 \
    --hard-cap 750 \
    --seed 42
