#!/bin/bash
# Five-seed re-run of the val-best normal-triplet DPT config from the
# normal_sweep grid: lr=1.25e-5, n=2000, r=64. Mirror of the hard-triplet
# val-selected 5-seed run under .../seed{42..46}/, so we can compare
# normal-trained DPT to hard-trained DPT with matched seed variance.
# Two-phase per seed (train --no-eval, then eval) to keep each python
# process under the 20 GiB MIG slice.
#SBATCH -J norm_5seed
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/norm_5seed_%j.out
#SBATCH -e logs/norm_5seed_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

MODEL="sentence-transformers/sentence-t5-xl"
NORMAL_TRIPLETS="data/processed/triplets/normal_train_triplets_2000.jsonl"
PER_DEVICE_BATCH=4
GRAD_ACCUM=4
EFFECTIVE_BATCH=$((PER_DEVICE_BATCH * GRAD_ACCUM))

LR=1.25e-5
N=2000
R=64
ALPHA=$((3 * R))
MAX_STEPS=$((N / EFFECTIVE_BATCH))

SEEDS=(42 43 44 45 46)

# ---- Phase 1: train each seed with --no-eval ----
for SEED in "${SEEDS[@]}"; do
    OUT="data/models/best/sentence_transformers_sentence_t5_xl/normal_valbest/seed${SEED}"
    if [ -f "$OUT/adapter_model.safetensors" ]; then
        echo "==== seed=$SEED already trained, skipping ===="
        continue
    fi
    echo "==== TRAIN seed=$SEED  lr=$LR  n=$N  r=$R  alpha=$ALPHA  max_steps=$MAX_STEPS ===="
    python scripts/train.py \
        --model "$MODEL" \
        --condition political \
        --selection diverse_2000 \
        --gen-model sonnet_standalone_prompt \
        --output-dir "$OUT" \
        --lr "$LR" \
        --max-steps "$MAX_STEPS" \
        --batch-size "$PER_DEVICE_BATCH" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --lora-r "$R" \
        --lora-alpha "$ALPHA" \
        --no-eval \
        --hard-triplets "$NORMAL_TRIPLETS" \
        --hard-ratio 1.0 \
        --hard-cap "$N" \
        --hard-eval-triplets data/processed/triplets/hard_eval_triplets_1k.jsonl \
        --seed "$SEED"
done

# ---- Phase 2: post-hoc eval (val/test/all + hard) for each seed ----
python scripts/experiments/eval_normal_sweep.py \
    --sweep-dir data/models/best/sentence_transformers_sentence_t5_xl/normal_valbest

echo "==== 5-seed normal-DPT run complete ===="
