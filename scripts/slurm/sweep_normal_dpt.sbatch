#!/bin/bash
# Hyperparameter val-sweep for DPT trained on the NORMAL counterfactual
# triplets (regime-i ablation, mirror of the hard-triplet val-sweep
# described in App E.1). Single seed across the full grid; the
# val-selected config is then re-run over 5 seeds in a follow-up sbatch.
#
# Grid (matches App E.1): 4 lrs * 5 caps * 5 ranks = 100 configs.
# Each ~3-5 min on ST5-XL → ~5-8 hours total. Skip-existing guard so
# the job is resumable after a preempt.
#SBATCH -J norm_sweep
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/norm_sweep_%j.out
#SBATCH -e logs/norm_sweep_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

# Reduce PyTorch CUDA memory fragmentation. Without this, sequential
# train -> eval cycles in a long bash loop accumulate fragmented
# allocations on the 20 GiB MIG slice and OOM mid-eval even though
# total live tensors are well under capacity.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

MODEL="sentence-transformers/sentence-t5-xl"
NORMAL_TRIPLETS="data/processed/triplets/normal_train_triplets_2000.jsonl"
SEED=42
PER_DEVICE_BATCH=4
GRAD_ACCUM=4
EFFECTIVE_BATCH=$((PER_DEVICE_BATCH * GRAD_ACCUM))

LRS=(1.25e-5 5e-5 1.25e-4 5e-4)
NS=(250 500 750 1000 2000)
RS=(4 8 16 32 64)

n_total=0; n_done=0; n_run=0
for LR in "${LRS[@]}"; do
for N in "${NS[@]}"; do
for R in "${RS[@]}"; do
    n_total=$((n_total + 1))
    ALPHA=$((3 * R))
    MAX_STEPS=$((N / EFFECTIVE_BATCH))
    LR_TAG=$(echo "$LR" | sed 's/[+-]/p/g; s/\./p/g')
    OUT="data/models/best/sentence_transformers_sentence_t5_xl/normal_sweep/lr${LR_TAG}_n${N}_r${R}_seed${SEED}"
    # Skip-existing check: with --no-eval the script writes the LoRA adapter
    # but no results.json; check for the adapter file instead.
    if [ -f "$OUT/adapter_model.safetensors" ]; then
        n_done=$((n_done + 1))
        continue
    fi
    n_run=$((n_run + 1))
    echo "==== [config $n_total/100]  lr=$LR  n=$N  r=$R  alpha=$ALPHA  max_steps=$MAX_STEPS ===="
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
done; done; done

echo "==== sweep complete: $n_total configs, $n_done already-done, $n_run newly trained ===="
