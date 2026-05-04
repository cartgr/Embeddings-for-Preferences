#!/usr/bin/env python3
"""Evaluate the per-topic rank-20 projected embedding on hard triplets.

For each eval dataset, load its probe weight matrix L (d x 20) fit on the
dataset's own participant vote data, embed the dataset's hard triplets with
frozen base sentence-T5-XL, and score each triplet with
    cos( L^T psi(a), L^T psi(p) )  vs  cos( L^T psi(a), L^T psi(n) ).
Triplet is correct if the match scores higher than the distractor.

This tests whether the per-topic probe, which reaches 79% on natural-data
triplets, has actually learned stance geometry or is riding topic-stance
correlation the way base cosine does.

Output: data/results/probe_on_hard_triplets.json
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embedding.model import get_device, load_model


def cosine(x, y):
    nx = np.linalg.norm(x); ny = np.linalg.norm(y)
    return float(np.dot(x, y) / (nx * ny)) if nx and ny else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="sentence-transformers/sentence-t5-xl")
    ap.add_argument("--hard-triplets",
                    default="data/processed/triplets/hard_eval_triplets_1k.jsonl")
    ap.add_argument("--probe-dir", default="data/results/psd_probe_weights")
    ap.add_argument("--output", default="data/results/probe_on_hard_triplets.json")
    args = ap.parse_args()

    triplets = [json.loads(l) for l in open(args.hard_triplets)]
    print(f"Loaded {len(triplets)} hard triplets")

    # Group by dataset; skip datasets without a per-topic probe weight file
    by_ds = {}
    for t in triplets:
        by_ds.setdefault(t["dataset"], []).append(t)

    probes = {}
    for ds in list(by_ds):
        p = Path(args.probe_dir) / f"L_perds_{ds}.npy"
        if not p.exists():
            print(f"  no probe for {ds}; skipping {len(by_ds[ds])} triplets")
            del by_ds[ds]
            continue
        probes[ds] = np.load(p)  # (d, r)

    print(f"Evaluating on {sum(len(v) for v in by_ds.values())} triplets across {len(by_ds)} datasets")

    # Load encoder once
    device = get_device()
    print(f"Loading {args.encoder} on {device}")
    model = load_model(args.encoder, device=device)

    # Collect all unique texts per dataset, embed, score
    results = {}
    agg_correct = 0
    agg_total = 0
    base_cos_correct = 0
    for ds, trs in by_ds.items():
        L = probes[ds]
        texts = list({t["anchor"] for t in trs}
                     | {t["preference_match"] for t in trs}
                     | {t["semantic_distractor"] for t in trs})
        embs = model.encode(texts, convert_to_numpy=True,
                             show_progress_bar=False, batch_size=64)
        t2e = dict(zip(texts, embs))

        n_correct = 0
        n_cos_correct = 0
        for t in trs:
            a, p, n = t["anchor"], t["preference_match"], t["semantic_distractor"]
            pa = L.T @ t2e[a]  # (r,)
            pp = L.T @ t2e[p]
            pn = L.T @ t2e[n]
            sp = cosine(pa, pp)
            sn = cosine(pa, pn)
            if sp > sn:
                n_correct += 1
            # Also track base-cosine accuracy on the same triplets for delta
            if cosine(t2e[a], t2e[p]) > cosine(t2e[a], t2e[n]):
                n_cos_correct += 1
        acc = n_correct / len(trs)
        cos_acc = n_cos_correct / len(trs)
        print(f"  {ds:<45} n={len(trs):>3}  probe={acc*100:5.1f}%  cos={cos_acc*100:5.1f}%  delta={ (acc-cos_acc)*100:+5.1f}")
        results[ds] = {
            "n": len(trs),
            "probe_acc": acc,
            "base_cos_acc": cos_acc,
            "delta_pp": (acc - cos_acc) * 100,
        }
        agg_correct += n_correct
        agg_total += len(trs)
        base_cos_correct += n_cos_correct

    mean_probe = np.mean([r["probe_acc"] for r in results.values()])
    mean_cos = np.mean([r["base_cos_acc"] for r in results.values()])
    print()
    print(f"Pooled accuracy (micro-avg):  probe={agg_correct/agg_total*100:.1f}%  cos={base_cos_correct/agg_total*100:.1f}%")
    print(f"Mean across datasets (macro): probe={mean_probe*100:.1f}%  cos={mean_cos*100:.1f}%")

    out = {
        "encoder": args.encoder,
        "hard_triplets_path": args.hard_triplets,
        "probe_dir": args.probe_dir,
        "per_dataset": results,
        "pooled": {
            "n": agg_total,
            "probe_acc": agg_correct / agg_total,
            "base_cos_acc": base_cos_correct / agg_total,
        },
        "macro_mean": {
            "probe_acc": float(mean_probe),
            "base_cos_acc": float(mean_cos),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
