.PHONY: help data pipeline train experiments figures tables clean smoke

MODEL    ?= sentence-transformers/sentence-t5-xl
GEN_MODEL ?= claude-sonnet-4-20250514

help:
	@echo "Targets:"
	@echo "  smoke         end-to-end sanity check (needs base encoder + LFS)"
	@echo "  tables        regenerate paper_tables.tex + dataset_stats.tex"
	@echo "  figures       regenerate Figs 2, 3, 4 (CPU; fig2_bands.py needs GPU)"
	@echo "  experiments   re-run every experiment that feeds a paper artifact"
	@echo "  data          download eval + issue sources (steps 1-2)"
	@echo "  pipeline      synthetic-data pipeline incl. LLM rewrites (steps 3-7)"
	@echo "  train         LoRA fine-tune (MODEL=<hf_id>; default ST5-XL)"
	@echo "  clean         remove temp training files"

smoke:
	python tests/smoke_test.py

# --- Reproduction from shipped artifacts ---------------------------------

# Paper tables: writes paper_tables.tex and dataset_stats.tex at repo root.
tables:
	python scripts/build_paper_tables.py
	python scripts/plotting/table_dataset_stats.py

# Figure PDFs into figures/ at repo root.
figures:
	python scripts/plotting/fig5_data_efficiency.py
	python scripts/plotting/fig_scorer_rank.py
	python scripts/plotting/fig2_bands.py            # needs GPU; ~20 min

# --- Re-run experiments from shipped LoRA checkpoints --------------------

# Sequential CPU/GPU run. On the FASRC cluster prefer the per-experiment
# slurm wrappers in scripts/slurm/.
experiments:
	python scripts/experiments/hard_triplet_cosine.py \
	    --output data/results/hard_triplet_cosine.json
	python scripts/experiments/eval_stance_aware_sbert.py
	python scripts/experiments/eval_sparsecl_bge.py
	python scripts/experiments/probe_sweep.py --scorer metric
	python scripts/experiments/probe_sweep.py --scorer cosine
	python scripts/experiments/probe_sweep.py --scorer metric_bias
	python scripts/experiments/scorer_structural_sweep.py
	python scripts/experiments/rank_sweep.py \
	    --output data/results/rank_sweep.json
	python scripts/experiments/data_efficiency.py \
	    --output data/results/data_efficiency.json
	python scripts/experiments/probe_on_hard_triplets.py \
	    --output data/results/probe_on_hard_triplets.json
	python scripts/experiments/error_decomposition.py \
	    --tuned-model data/models/best/sentence_transformers_sentence_t5_xl/seed42 \
	    --output data/results/error_decomposition.json
	python scripts/experiments/cluster_coherence.py
	python scripts/experiments/likert_correlation.py
	python scripts/experiments/participant_paired_test.py
	python scripts/experiments/probe_margin_decomposition.py
	# Optional: classify_errors.py needs OPENAI_API_KEY.
	# python scripts/experiments/classify_errors.py

# --- From-scratch reproduction (data + training) -------------------------

data:
	python scripts/data/01_download_eval.py
	python scripts/data/02_download_issue_sources.py

pipeline: data
	python scripts/data/03_filter_issues.py --condition political
	python scripts/data/04_generate_opinions.py --model $(GEN_MODEL)
	python scripts/data/05_build_training_triplets.py \
	    --condition political --selection diverse_2000 \
	    --gen-model $(GEN_MODEL)
	python scripts/data/06_generate_hard_triplets.py
	python scripts/data/07_generate_hard_eval.py --generate

train:
	python scripts/train.py --model $(MODEL) \
	    --condition political --selection diverse_2000 \
	    --gen-model sonnet_standalone_prompt \
	    --lr 1.25e-4 --batch-size 16 --lora-r 16 --lora-alpha 48 \
	    --hard-triplets data/processed/triplets/hard_train_triplets_10000.jsonl \
	    --hard-ratio 1.0 --hard-cap 750 --seed 42

# --- Cleanup -------------------------------------------------------------

clean:
	rm -rf data/processed/triplets/_train_*.jsonl
