# Reproducing the paper

`make experiments` runs every experiment sequentially. On a SLURM
cluster, prefer the per-experiment wrappers below — they parallelize
across the queue. The `#SBATCH` headers and partition names assume one
specific cluster; edit them for your environment.

## Per-experiment slurm wrappers

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

After the jobs complete, re-run `make tables` (and `make figures` for
plots) to propagate updated JSONs into `paper_tables.tex`.

## Training from scratch

```bash
make data                                      # download eval datasets and issue sources
make pipeline                                  # synthetic-data pipeline (~$25 API), GEN_MODEL=claude-sonnet-4-20250514
sbatch scripts/slurm/cross_model_sweep.sbatch  # 4 encoders × 5 seeds, ~80 A100-min
sbatch scripts/slurm/eval_normal_sweep.sbatch  # post-hoc evaluation
make experiments && make figures && make tables
```

## Paper artifact → producing script

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
