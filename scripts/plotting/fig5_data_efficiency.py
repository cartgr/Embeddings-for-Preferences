#!/usr/bin/env python3
"""Figure 5 — Data-efficiency, per-dataset panels.

Eleven small panels, six on the top row and five centered beneath. Each
panel plots test accuracy vs. labels-per-topic K for one dataset, with
two horizontal reference lines per panel:
  * base ST5-XL cosine, hard-coded from Tab 4 / Tab 12 (paper_tables.tex
    tab:full-main, ST5-XL row).
  * DPT-tuned ST5-XL cosine, hard-coded from Tab 4 / Tab 12 (ST5-XL+DPT
    row, mean over 5 seeds).

Physical width matches NeurIPS \\textwidth (5.5 in) so that text rendered
through LaTeX with mathptmx scales 1:1 with the body font (Times Roman,
10pt body / 9pt small).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from style import setup_style, COLORS

import matplotlib.pyplot as plt
import numpy as np

EVAL_DATASETS = [
    "gsc_abortion_gen", "gsc_abortion_val", "gsc_chatbot_gen",
    "remesh_campus_protests", "remesh_foreign_intervention", "remesh_right_to_assemble",
    "polis_15_per_hour_seattle", "polis_american_assembly_bowling_green",
    "polis_brexit_consensus", "polis_canadian_electoral_reform", "polis_scoop_hivemind_ubi",
]
SHORT = {
    "gsc_abortion_gen":                       "GSC AbG",
    "gsc_abortion_val":                       "GSC AbV",
    "gsc_chatbot_gen":                        "GSC Chat",
    "remesh_campus_protests":                 "Camp",
    "remesh_foreign_intervention":            "Frgn",
    "remesh_right_to_assemble":               "R2A",
    "polis_15_per_hour_seattle":              "Seattle",
    "polis_american_assembly_bowling_green":  "BG",
    "polis_brexit_consensus":                 "Brexit",
    "polis_canadian_electoral_reform":        "Canadian",
    "polis_scoop_hivemind_ubi":               "UBI",
}
PANEL_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

# Per-dataset base and DPT-tuned ST5-XL cosine accuracy, hard-coded from
# Tab 4 / Tab 12 (paper_tables.tex tab:full-main, ST5-XL row and ST5-XL+DPT
# row). Pinned here so the figure's reference lines match the paper's
# headline numbers regardless of any local participant-split or seed
# differences in data_efficiency.json.
BASE_ACC = {
    "gsc_abortion_gen":                       81.1,
    "gsc_abortion_val":                       70.5,
    "gsc_chatbot_gen":                        69.7,
    "remesh_campus_protests":                 65.8,
    "remesh_foreign_intervention":            61.1,
    "remesh_right_to_assemble":               69.9,
    "polis_15_per_hour_seattle":              60.7,
    "polis_american_assembly_bowling_green":  57.6,
    "polis_brexit_consensus":                 56.4,
    "polis_canadian_electoral_reform":        55.5,
    "polis_scoop_hivemind_ubi":               69.2,
}
TUNED_ACC = {
    "gsc_abortion_gen":                       81.8,
    "gsc_abortion_val":                       76.4,
    "gsc_chatbot_gen":                        65.9,
    "remesh_campus_protests":                 67.5,
    "remesh_foreign_intervention":            65.6,
    "remesh_right_to_assemble":               69.5,
    "polis_15_per_hour_seattle":              69.8,
    "polis_american_assembly_bowling_green":  65.1,
    "polis_brexit_consensus":                 68.4,
    "polis_canadian_electoral_reform":        62.5,
    "polis_scoop_hivemind_ubi":               61.9,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/results/data_efficiency.json")
    ap.add_argument("--output",
                    default="figures/fig5_data_efficiency.pdf")
    args = ap.parse_args()
    setup_style(use_latex=True)

    # Tighten font sizes for the small panels — keep LaTeX/Times so the
    # font *family* matches the paper body exactly even though point size
    # has to drop to fit a 0.9in-wide panel.
    plt.rcParams.update({
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    d = json.load(open(args.input))
    k_grid = d["k_grid"]
    curves = {}
    for ds in EVAL_DATASETS:
        if ds not in d["results"]: continue
        by_k = {c["K"]: c["mean_test"] for c in d["results"][ds]["curve"]}
        xs = [k for k in k_grid if k in by_k]
        ys = [by_k[k] for k in xs]
        curves[ds] = (xs, ys)

    base_cos  = BASE_ACC
    tuned_cos = TUNED_ACC

    # Layout: 6 panels on top, 5 centered beneath. Each panel = 2 mosaic cells.
    mosaic = """
AABBCCDDEEFF
.GGHHIIJJKK.
"""
    # Disable constrained_layout (set globally by setup_style); we hand-pick
    # subplot positions below since the 6+5 mosaic confuses the auto layout.
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axd = plt.subplot_mosaic(
        mosaic.strip(),
        figsize=(5.5, 2.6),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.55, "wspace": 0.25},
    )

    line_color   = COLORS["blue"]
    base_color   = COLORS["red"]
    tuned_color  = COLORS["green"]

    top_keys = set("ABCDEF")
    bot_keys = set("GHIJK")

    for key, ds in zip(PANEL_KEYS, EVAL_DATASETS):
        if key not in axd: continue
        ax = axd[key]
        if ds not in curves: continue
        xs, ys = curves[ds]
        h_curve, = ax.plot(xs, [y * 100 for y in ys],
                           color=line_color, linewidth=1.0, marker="o", markersize=2.0)
        if ds in base_cos:
            h_base = ax.axhline(base_cos[ds],
                                color=base_color, linestyle="--", linewidth=0.7)
        if ds in tuned_cos:
            h_tuned = ax.axhline(tuned_cos[ds],
                                 color=tuned_color, linestyle=":", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_title(SHORT[ds], pad=2)
        ax.tick_params(axis="both", which="both", length=2.0, pad=1.5)
        # x-ticks at K = 100, 1000, 10000 only
        ax.set_xticks([100, 1000, 10000])
        ax.get_xaxis().set_major_formatter(
            plt.matplotlib.ticker.FuncFormatter(lambda x, _: {100:"100", 1000:"1k", 10000:"10k"}.get(int(x), "")))
        ax.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        ax.set_yticks([50, 60, 70, 80, 90])
        ax.set_ylim(45, 100)
        # x labels visible on every TOP-row panel (default sharex would
        # hide them since each has a panel below it in part of its column);
        # y labels visible only on the leftmost BOTTOM-row panel (G), since
        # the bottom row is offset so default sharey hides G's y labels.
        if key in top_keys:
            ax.tick_params(axis="x", labelbottom=True)
        if key == "G":
            ax.tick_params(axis="y", labelleft=True)

    # Common axis labels on the figure (anchored to the leftmost / bottom panels).
    fig.supxlabel("Labels per topic $K$", fontsize=8, y=0.07)
    fig.supylabel("Test accuracy (\\%)",   fontsize=8, x=0.012)

    # Single shared legend below the panels.
    legend_handles = [
        plt.Line2D([], [], color=line_color,  marker="o", markersize=2.5, linewidth=1.0,
                   label="Per-topic projected embedding"),
        plt.Line2D([], [], color=base_color,  linestyle="--", linewidth=0.9,
                   label="Base ST5-XL cosine"),
        plt.Line2D([], [], color=tuned_color, linestyle=":", linewidth=1.1,
                   label="DPT-tuned ST5-XL cosine"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncols=3,
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=8,
               handlelength=1.6, columnspacing=1.5, handletextpad=0.4)

    plt.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.18)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        p = out.with_suffix(ext); plt.savefig(p); print(f"Saved {p}")


if __name__ == "__main__":
    main()
