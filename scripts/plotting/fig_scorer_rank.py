#!/usr/bin/env python3
"""Per-topic scorer rank saturation plot.

Reads data/results/scorer_structural_ablation.json (rank_1..rank_100 variants
plus the baseline metric at r=20) and plots natural and hard-triplet accuracy
against rank, macro-averaged across 11 datasets with std over 3 seeds.

Usage:
    python scripts/plotting/fig_scorer_rank.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from style import setup_style, COLORS, get_figure_size

import matplotlib.pyplot as plt
import numpy as np


def macro_means(variant_block):
    seeds = sorted({s for ds_m in variant_block.values() for s in ds_m.keys()})
    nats, hards = [], []
    for seed in seeds:
        nat_per_ds = [m[seed]["test"] for m in variant_block.values()
                      if seed in m and m[seed].get("test") is not None]
        hard_per_ds = [m[seed]["hard_acc"] for m in variant_block.values()
                       if seed in m and m[seed].get("hard_acc") is not None]
        nats.append(np.mean(nat_per_ds))
        hards.append(np.mean(hard_per_ds))
    return np.mean(nats), np.std(nats), np.mean(hards), np.std(hards)


def main():
    setup_style(use_latex=True)
    data = json.load(open("data/results/scorer_structural_ablation.json"))

    ranks = [1, 2, 5, 10, 20, 50, 100]
    rows = []
    for r in ranks:
        key = "metric" if r == 20 else f"rank_{r}"
        nat, nat_s, hard, hard_s = macro_means(data[key])
        rows.append((r, nat, nat_s, hard, hard_s))

    rs = [row[0] for row in rows]
    nat_m = np.array([row[1] for row in rows]) * 100
    nat_s = np.array([row[2] for row in rows]) * 100
    hard_m = np.array([row[3] for row in rows]) * 100
    hard_s = np.array([row[4] for row in rows]) * 100

    # Physical size must match the rendered width so LaTeX does not rescale
    # the figure; 0.42 \textwidth = 0.42 * 5.5 in = 2.31 in. Same convention as
    # fig2_bands.py so body figures are dimensionally consistent.
    fig, ax = plt.subplots(figsize=get_figure_size(2.31, aspect=0.85))
    ax.errorbar(rs, nat_m, yerr=nat_s, marker="o", color=COLORS["blue"],
                label="Natural", capsize=1.5, linewidth=1.1, markersize=3)
    ax.errorbar(rs, hard_m, yerr=hard_s, marker="s", color=COLORS["red"],
                label="Hard", capsize=1.5, linewidth=1.1, markersize=3)
    ax.axvline(20, color=COLORS["gray"], linestyle=":", linewidth=0.7,
               alpha=0.7, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(rs)
    ax.set_xticklabels([str(r) for r in rs])
    ax.set_xlabel("Projection rank $r$")
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_ylim(67, 84)
    ax.legend(loc="lower right", handlelength=1.0, borderaxespad=0.3)

    out = Path("figures/fig_rank_saturation.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        p = out.with_suffix(ext); plt.savefig(p); print(f"Saved {p}")

    print("\nNumbers for appendix table:")
    for r, n, ns, h, hs in rows:
        print(f"  r={r:>3}  nat={n*100:.1f}±{ns*100:.2f}  hard={h*100:.1f}±{hs*100:.2f}")


if __name__ == "__main__":
    main()
