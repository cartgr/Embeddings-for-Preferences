#!/bin/bash
# Train the rank-20 ideal-point metric probe on the TUNED ST5-XL
# embeddings (LoRA adapter from §sec:dpt) instead of the BASE encoder.
# 5 seeds, val-selected hyperparameters per (dataset, seed). Pair with
# the existing data/results/scorer_structural_ablation.json (which has
# the same metric variant on the base encoder) for the §sec:probes
# 'are §6 and §7 orthogonal contributions?' ablation.
#
# Variant filter: only `metric` (the canonical rank-20 ideal-point form)
# — the 13-variant structural ablation on tuned is a separate question.
#
# 11 datasets * 5 seeds * 8 HP configs = 440 probe trainings -> ~30 min.
#SBATCH -J probe_tuned
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/probe_tuned_%j.out
#SBATCH -e logs/probe_tuned_%j.err

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01
export PATH="$HOME/.conda/envs/elicitation/bin:$PATH"

cd /n/home11/cblair/Info-Theoretic-Preference-Elicitation
mkdir -p logs

python scripts/experiments/scorer_structural_sweep.py \
    --encoder data/models/best/sentence_transformers_sentence_t5_xl/targeted_sweep/lr1p25eem4_n750_r16_a48 \
    --variants metric \
    --output data/results/scorer_structural_ablation_tuned.json
