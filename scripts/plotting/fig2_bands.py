#!/usr/bin/env python3
"""Figure 2 — Proximity bands.

For each encoder, bin all eval-triplet (anchor, statement) pairs by
cosine-similarity decile and compute approval rate per bin.

Reads:
  - data/processed/eval/*.jsonl
  - scores with HuggingFace sentence transformers (GPU required)

Caches per-encoder band stats to data/results/bands_<model_slug>.json so
re-runs only embed missing models.

Usage:
    python scripts/plotting/fig2_bands.py \\
        --encoders sentence-transformers/sentence-t5-xl intfloat/e5-large-v2 \\
                   BAAI/bge-large-en-v1.5 sentence-transformers/all-mpnet-base-v2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from style import setup_style, COLORS, get_figure_size

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.evaluator import EVAL_DATASETS
from src.embedding.model import get_device, load_model


PALETTE = [COLORS["blue"], COLORS["orange"], COLORS["green"],
           COLORS["red"], COLORS["purple"]]

DISPLAY_NAMES = {
    "sentence-transformers/sentence-t5-xl":     "ST5-XL",
    "intfloat/e5-large-v2":                     "e5-large-v2",
    "BAAI/bge-large-en-v1.5":                   "BGE-large-en",
    "sentence-transformers/all-mpnet-base-v2":  "all-mpnet-base",
}


def model_slug(m: str) -> str:
    return m.replace("/", "_").replace("-", "_").replace(".", "_")


def bands_for_encoder(model_id: str, eval_dir: Path, n_bands: int = 10,
                       max_triplets_per_ds: int = 5000, seed: int = 0,
                       datasets: list = None):
    import random
    rng = random.Random(seed)
    device = get_device()
    model = load_model(model_id, device=device)

    sims, approvals = [], []
    for ds in (datasets or EVAL_DATASETS):
        path = eval_dir / f"{ds}.jsonl"
        if not path.exists(): continue
        triplets = [json.loads(l) for l in open(path)]
        if len(triplets) > max_triplets_per_ds:
            triplets = rng.sample(triplets, max_triplets_per_ds)

        texts = set()
        for t in triplets:
            texts.update(t["anchor_texts"])
            texts.add(t["preferred"]); texts.add(t["dispreferred"])
        texts = list(texts)
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        t2e = dict(zip(texts, embs))

        for t in triplets:
            emb_a = np.mean([t2e[x] for x in t["anchor_texts"] if x in t2e], axis=0)
            na = np.linalg.norm(emb_a)
            if na == 0: continue
            for emb_s, approved in [(t2e[t["preferred"]], 1), (t2e[t["dispreferred"]], 0)]:
                ns = np.linalg.norm(emb_s)
                if ns == 0: continue
                sims.append(float(np.dot(emb_a, emb_s) / (na * ns)))
                approvals.append(approved)

    sims = np.array(sims); approvals = np.array(approvals)
    edges = np.quantile(sims, np.linspace(0, 1, n_bands + 1))
    bands = []
    for i in range(n_bands):
        mask = (sims >= edges[i]) & (sims < edges[i+1]) if i < n_bands - 1 else (sims >= edges[i])
        if mask.sum() == 0: continue
        bands.append({
            "band": i + 1, "sim_lo": float(edges[i]), "sim_hi": float(edges[i+1]),
            "sim_mean": float(sims[mask].mean()),
            "approval_rate": float(approvals[mask].mean()),
            "n": int(mask.sum()),
        })
    del model
    return bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=[
        "sentence-transformers/sentence-t5-xl",
        "intfloat/e5-large-v2",
        "BAAI/bge-large-en-v1.5",
        "sentence-transformers/all-mpnet-base-v2",
    ])
    ap.add_argument("--eval-dir",  type=Path, default=Path("data/processed/eval"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/results/bands"))
    ap.add_argument("--output", default="figures/fig2_bands.pdf")
    ap.add_argument("--table-out", default=None,
                    help="Optional: path to emit a LaTeX bands table (one row per encoder).")
    ap.add_argument("--n-bands", type=int, default=5,
                    help="Number of equal-mass bands (5 = quintiles, 10 = deciles).")
    ap.add_argument("--exclude-datasets", nargs="*", default=[],
                    help="Datasets to drop from the bands. E.g. all GSC: "
                         "--exclude-datasets gsc_abortion_gen gsc_abortion_val gsc_chatbot_gen")
    ap.add_argument("--cache-suffix", default="",
                    help="Appended to bands cache filenames. Use a distinct suffix "
                         "(e.g. '_no-gsc') when --exclude-datasets is non-empty so "
                         "the full-dataset bands cache isn't clobbered.")
    args = ap.parse_args()
    setup_style(use_latex=True)
    datasets = [d for d in EVAL_DATASETS if d not in set(args.exclude_datasets)]
    if args.exclude_datasets:
        print(f"Excluding {args.exclude_datasets}; using {len(datasets)} datasets")

    # Physical size must match the rendered width so that LaTeX does not
    # rescale the figure (which would shrink the text away from body size).
    # Rendered width = 0.42 \textwidth = 0.42 * 5.5 in = 2.31 in.
    fig, ax = plt.subplots(figsize=get_figure_size(2.31, aspect=0.85))
    all_bands = {}

    # Plot in reverse order so the first encoder (ST5-XL) is drawn last
    # — sits on top in overlaps — and shows up at the BOTTOM of the
    # legend. (matplotlib's default legend ordering is the plotting order,
    # top-to-bottom.)
    for i, enc in enumerate(args.encoders):
        cache_path = args.cache_dir / f"bands_{model_slug(enc)}_n{args.n_bands}{args.cache_suffix}.json"
        if cache_path.exists():
            bands = json.load(open(cache_path))["bands"]
            print(f"  cached: {enc}")
        else:
            print(f"  computing: {enc}")
            bands = bands_for_encoder(enc, args.eval_dir, args.n_bands, datasets=datasets)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump({"model": enc, "n_bands": args.n_bands, "bands": bands},
                      open(cache_path, "w"), indent=2)
        all_bands[enc] = bands

    for i, enc in reversed(list(enumerate(args.encoders))):
        bands = all_bands[enc]
        xs = [b["band"] for b in bands]
        ys = [b["approval_rate"] for b in bands]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.1,
                color=PALETTE[i % len(PALETTE)],
                label=DISPLAY_NAMES.get(enc, enc.split("/")[-1]))

    band_label = {5: "quintile", 10: "decile"}.get(args.n_bands, f"{args.n_bands}-tile")
    ax.set_xlabel(f"Cosine-similarity {band_label}")
    ax.set_ylabel("Approval rate")
    ax.set_xticks(range(1, args.n_bands + 1))
    ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.3)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        p = out.with_suffix(ext); plt.savefig(p); print(f"Saved {p}")

    # Emit a LaTeX table with one row per encoder.
    if args.table_out:
        tbl = Path(args.table_out); tbl.parent.mkdir(parents=True, exist_ok=True)
        cols = "l" + "c" * args.n_bands
        header = " & ".join([""] + [str(i) for i in range(1, args.n_bands + 1)]) + r" \\"
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            rf"\caption{{Approval rate (\%) by cosine-similarity {band_label}. "
            rf"{band_label.capitalize()}~1 contains the most distant anchor--statement "
            rf"pairs; {band_label}~{args.n_bands} the most similar. Approval rate rises "
            rf"monotonically with similarity across every encoder, confirming that "
            rf"semantic proximity and preference are observationally correlated "
            rf"(Proposition~\ref{{prop:obs}}, regime~(i)).}}",
            r"\label{tab:bands}",
            rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{cols}@{{}}}}",
            r"\toprule",
            rf"& \multicolumn{{{args.n_bands}}}{{c}}{{\textbf{{Cosine-similarity {band_label}}}}} \\",
            rf"\cmidrule(lr){{2-{args.n_bands + 1}}}",
            r"\textbf{Model} & " + header.split("& ", 1)[1],
            r"\midrule",
        ]
        for enc in args.encoders:
            bands = all_bands.get(enc, [])
            cells = [f"{b['approval_rate']*100:.1f}" for b in bands]
            name = DISPLAY_NAMES.get(enc, enc.split("/")[-1])
            lines.append(" & ".join([name] + cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
        tbl.write_text("\n".join(lines) + "\n")
        print(f"Saved {tbl}")


if __name__ == "__main__":
    main()
