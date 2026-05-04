#!/bin/bash
# Cross-model fine-tune: 4 encoders x 5 seeds (42..46) = 20 runs,
# all sequentially in a single job. gpu_test 12h wall is sufficient
# (~10-20 min per run, ~3-7 h total).
#
# Effective batch size = 16 via per-device batch 8 + grad accumulation 2.
# Per-seed checkpoint: data/models/best/<model_slug>/seed<N>/
# Skip-existing guard: if a seed's results.json is already there it's
# skipped, so the job is resumable after a preempt.
#SBATCH -J xmodel
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/xmodel_%j.out
#SBATCH -e logs/xmodel_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

# Model catalog: hf_id | lr | hard_cap | query_prefix | passage_prefix.
# Fixed: lora_r=16, lora_alpha=48, eff_batch=16.
# query/passage prefixes match each encoder's documented retrieval format
# so the LoRA learns and is evaluated in the model's native asymmetric
# geometry. ST5-XL / mpnet are symmetric — empty prefixes are a no-op.
MODELS=(
    "sentence-transformers/sentence-t5-xl|1.25e-4|750||"
    "intfloat/e5-large-v2|5e-5|1000|query: |passage: "
    "BAAI/bge-large-en-v1.5|1.25e-4|500|Represent this sentence for searching relevant passages: |"
    "sentence-transformers/all-mpnet-base-v2|1.25e-4|1000||"
)
SEEDS=(42 43 44 45 46)

PER_DEVICE_BATCH=8
GRAD_ACCUM=2
EFFECTIVE_BATCH=$((PER_DEVICE_BATCH * GRAD_ACCUM))   # = 16
LORA_R=16
LORA_ALPHA=48

for ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r MODEL LR CAP QPREFIX PPREFIX <<< "$ENTRY"
    MAX_STEPS=$((CAP / EFFECTIVE_BATCH))
    MODEL_SLUG=$(echo "$MODEL" | sed 's/[\/._-]/_/g')
    echo "==== encoder $MODEL  lr=$LR  hard_cap=$CAP  max_steps=$MAX_STEPS"
    echo "      qprefix='$QPREFIX'  pprefix='$PPREFIX'"

    for SEED in "${SEEDS[@]}"; do
        OUT="data/models/best/${MODEL_SLUG}/seed${SEED}"
        if [ -f "$OUT/results.json" ]; then
            echo "  seed=$SEED: results already exist at $OUT, skipping."
            continue
        fi
        echo "---- seed=$SEED -> $OUT ----"
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
            --lora-r "$LORA_R" \
            --lora-alpha "$LORA_ALPHA" \
            --hard-triplets data/processed/triplets/hard_train_triplets_10000.jsonl \
            --hard-ratio 1.0 \
            --hard-cap "$CAP" \
            --hard-eval-triplets data/processed/triplets/hard_eval_triplets_1k.jsonl \
            --query-prefix "$QPREFIX" \
            --passage-prefix "$PPREFIX" \
            --seed "$SEED"
    done
done

echo "==== sweep complete ===="
