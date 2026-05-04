# Preference Embeddings

Code and data for *Preference Embeddings* (NeurIPS 2026).
**Paper:** TODO add arxiv / openreview link.

We learn a sentence embedding whose distances reflect **preferential**
similarity — the probability that a user who agrees with one text agrees
with nearby texts — rather than semantic similarity, by fine-tuning on
counterfactual hard triplets.

---

## Setup

Requires Python 3.11+ and CUDA 12.x. Approx 20 GB of LFS artifacts.

```bash
git clone https://github.com/<org>/preference-embeddings.git
cd preference-embeddings
git lfs install && git lfs pull        # ~20 GB

conda env create -f environment.yml
conda activate preference-embeddings
# (or:  python -m venv .venv && source .venv/bin/activate && pip install -e .)

# API keys for the data-generation pipeline + error-classification script
cp .env.template .env
# edit .env to add OPENAI_API_KEY and ANTHROPIC_API_KEY

python tests/smoke_test.py             # verify setup
```

The slurm wrappers in `scripts/slurm/*.sbatch` target the FASRC Cannon
cluster (`module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01`,
partitions `gpu_test` / `gpu_requeue`). On other clusters, edit the
`#SBATCH` headers and `module load` block; the underlying Python
invocations are portable.

---

## Reproducing the paper

Three escalating effort levels.

### 1. Quickstart — regenerate every table from shipped JSONs (~2 min, no GPU)

```bash
make tables
```

Writes `paper_tables.tex` at the repo root, containing the row blocks for
all 13 auto-generated tables, each prefaced by a `% === <label> ===`
separator. Also writes `dataset_stats.tex` at the repo root.

Pre-built figure PDFs (`fig2_bands.pdf`, `fig5_data_efficiency.pdf`,
`fig_rank_saturation.pdf`) live in `figures/`. The numbers in
`paper_tables.tex` are the authoritative regenerable record; if your
copy of the paper has different numbers (e.g. an older draft), trust
the JSON-backed values here.

### 2. Re-run experiments from shipped LoRA checkpoints (~1 GPU-hour)

Recreates every JSON in `data/results/` from the model checkpoints in
`data/models/best/`. On the cluster, prefer the per-experiment slurm
wrappers; without a cluster, `make experiments` runs them sequentially.

```bash
# core diagnostic + projection sweeps
sbatch scripts/slurm/hard_triplet_cosine_1k.sbatch     # tab:cross-model + tab:main-hard inputs
sbatch scripts/slurm/scorer_structural_sweep.sbatch    # tab:probes, tab:scorer-extensions, tab:rank-saturation
sbatch scripts/slurm/probe_sweep.sbatch                # per-topic metric/cosine/metric_bias scorers
sbatch scripts/slurm/rank_sweep.sbatch                 # fig:rank-saturation
sbatch scripts/slurm/data_efficiency.sbatch            # fig:data-eff

# baseline re-evaluations (extends Tab 4 / Tab 12)
sbatch scripts/slurm/reeval_baselines.sbatch
sbatch scripts/slurm/stance_aware_sbert.sbatch
sbatch scripts/slurm/sparsecl_bge.sbatch

# appendix experiments
sbatch scripts/slurm/cluster_coherence.sbatch          # tab:cluster-coherence
sbatch scripts/slurm/likert_correlation.sbatch         # tab:likert-correlation
sbatch scripts/slurm/paired_test.sbatch                # tab:paired-test
sbatch scripts/slurm/probe_on_hard_triplets.sbatch     # tab:probe-on-tuned (base side)
sbatch scripts/slurm/probe_margin_decomposition.sbatch # tab:probe-on-tuned-margin
sbatch scripts/slurm/error_decomposition.sbatch        # narrative numbers in App C
sbatch scripts/slurm/triplet_decomposition.sbatch      # Δ_S / Δ_T diagnostic per dataset
sbatch scripts/slurm/probe_subspace_analysis.sbatch    # principal-angle numbers in App E.6
```

Then re-run `make tables` (and `make figures` if you want to refresh
plots) to propagate the updated JSONs to `paper_tables.tex`.

### 3. From scratch — re-train + re-generate data (~$30 API + ~2 GPU-hours)

```bash
# Step 1-2: download eval datasets and issue sources (no API key needed)
make data

# Step 3-7: synthetic-data pipeline (~$25 in API costs total)
make pipeline                                  # uses GEN_MODEL=claude-sonnet-4-20250514

# Step 8: 4 encoders × 5 seeds = ~80 A100-min
sbatch scripts/slurm/cross_model_sweep.sbatch
sbatch scripts/slurm/eval_normal_sweep.sbatch  # post-hoc evaluation

# Step 9: re-run all paper experiments
make experiments                               # or sbatch wrappers above
make figures
make tables
```

---

## Reference: paper artifact → producing script

| Paper                | Generating script(s)                                    | Slurm wrapper                                | Output                                                      |
|----------------------|---------------------------------------------------------|----------------------------------------------|-------------------------------------------------------------|
| Tab 1 — example      | `table1_example.py`                                     | —                                            | `data/results/table1_example.json`                           |
| Tab 2 — main-hard    | `experiments/hard_triplet_cosine.py`                    | `hard_triplet_cosine_1k.sbatch`              | `data/results/hard_triplet_cosine_1k.json` → `paper_tables.tex` |
| Tab 3 — cross-model  | `train.py` (×5 seeds) + `eval_normal_sweep.py`          | `cross_model_sweep.sbatch`, `eval_normal_sweep.sbatch` | `data/models/best/<slug>/seed*/results.json` → `paper_tables.tex` |
| Tab 4 — main         | `experiments/reeval_baselines.py` + cross-model seeds   | `reeval_baselines.sbatch`                    | `data/results/per_model/base_*.json` → `paper_tables.tex`   |
| Tab 5 — probes       | `experiments/scorer_structural_sweep.py`                | `scorer_structural_sweep.sbatch`             | `data/results/scorer_structural_ablation.json` → `paper_tables.tex` |
| Tab 6 — errors       | `experiments/classify_errors.py` (LLM categorisation)   | `classify_errors.sbatch`                     | `data/results/error_classification.json` (not auto-emitted) |
| Tab 7 — rank-ablation| existing `data/models/best/lora_rank_ablation/r*/results.json` | `lora_rank_ablation.sbatch`           | `paper_tables.tex`                                          |
| Tab 8 — paired-test  | `experiments/participant_paired_test.py`                | `paired_test.sbatch`                         | `data/results/paired_test_dpt_vs_base.json` → `paper_tables.tex` |
| Tab 9 — rank-saturation | `experiments/scorer_structural_sweep.py` (rank variants) | `scorer_structural_sweep.sbatch`        | same JSON → `paper_tables.tex`                              |
| Tab 10 — cluster-coherence | `experiments/cluster_coherence.py`                | `cluster_coherence.sbatch`                   | `data/results/cluster_coherence.json` → `paper_tables.tex`  |
| Tab 11 — likert-correlation | `experiments/likert_correlation.py`              | `likert_correlation.sbatch`                  | `data/results/likert_correlation.json` → `paper_tables.tex` |
| Tab 12 — full-main   | same as Tab 4                                            | same                                         | `paper_tables.tex`                                          |
| Tab 13 — probe-on-tuned | `experiments/scorer_structural_sweep.py` (base + tuned) | `scorer_structural_sweep.sbatch`, `scorer_metric_tuned.sbatch` | `scorer_structural_ablation[_tuned].json` → `paper_tables.tex` |
| Tab 14 — probe-on-tuned-margin | `experiments/probe_margin_decomposition.py`    | `probe_margin_decomposition.sbatch`          | `data/results/probe_margin_decomposition.json` → `paper_tables.tex` |
| Tab 15 — datasets    | `plotting/table_dataset_stats.py`                       | —                                            | `dataset_stats.tex` (at repo root)                          |
| Fig 2 — bands        | `plotting/fig2_bands.py` (GPU)                          | `bands.sbatch`                               | `figures/fig2_bands.pdf`                                    |
| Fig 3 — rank-saturation | `plotting/fig_scorer_rank.py`                       | —                                            | `figures/fig_rank_saturation.pdf`                           |
| Fig 4 — data-efficiency | `plotting/fig5_data_efficiency.py` + `experiments/data_efficiency.py` | `data_efficiency.sbatch`, `fig5_data_efficiency.sbatch` | `figures/fig5_data_efficiency.pdf`               |

---

## Repo layout

```
.
├── configs/                  # paths, models, eval settings
├── src/                      # library (encoders, trainers, probes, evaluator)
├── scripts/
│   ├── build_paper_tables.py # writes paper_tables.tex at repo root
│   ├── table1_example.py     # single-triplet similarity demo (Tab 1)
│   ├── train.py              # fine-tune entry point
│   ├── evaluate.py           # evaluation entry point
│   ├── data/                 # ordered pipeline 01..08
│   ├── experiments/          # paper experiments
│   ├── plotting/             # figure + dataset-stats generators
│   └── slurm/                # SLURM sbatch wrappers (FASRC Cannon defaults)
├── data/                     # LFS-tracked artifacts
│   ├── processed/eval/       # 11 human-labeled triplet datasets
│   ├── processed/issues/     # filtered political issues
│   ├── processed/opinions/   # LLM-generated opinions
│   ├── processed/triplets/   # synthetic training + hard eval triplets
│   ├── models/best/          # LoRA adapters (4 encoder families × 5 seeds)
│   └── results/              # experiment JSONs + PSD probe weights
├── figures/                  # generated PDFs/PNGs (Fig 2, 3, 4 + bands ablation)
├── paper_tables.tex          # auto-generated row blocks for every paper table
├── dataset_stats.tex         # auto-generated dataset-stats table
├── tests/smoke_test.py       # quick sanity check
├── Makefile                  # convenience targets
└── pyproject.toml
```

---

## Datasets

11 human-preference triplet datasets in `data/processed/eval/`:

| Source | Dataset                                            | Participants | Triplets    |
|--------|----------------------------------------------------|-------------:|------------:|
| GSC    | abortion_gen, abortion_val, chatbot_gen            | 100 / 100 / 100 | 0.8K / 3.5K / 1.1K |
| Polis  | seattle, bowling_green, brexit, canadian, ubi      | 13–222       | 0.5K – 1.1M |
| Remesh | campus_protests, foreign_intervention, right_to_assemble | 289–298 | 2K – 25K    |

Schema (JSONL, one triplet per line):

```json
{"anchor_texts":   ["user's own text 1", "user's own text 2"],
 "preferred":      "higher-rated statement",
 "dispreferred":   "lower-rated statement",
 "dataset":        "gsc_abortion_gen",
 "participant_id": "gen1"}
```

---

## Cite

```bibtex
@inproceedings{preference_embeddings_2026,
  title     = {Preference Embeddings},
  author    = {Anonymous},
  booktitle = {Proceedings of NeurIPS 2026},
  year      = {2026},
}
```

See `CITATION.cff` for machine-readable metadata.

---

## License

Code: MIT. Paper artifacts: see individual sources. The GSC, Polis,
Remesh, Habermas, and Kialo datasets are distributed under their own
upstream licenses; consult those before redistributing.
