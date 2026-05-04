#!/usr/bin/env python3
"""Supervised embedding method comparison.

Compares four methods for learning a topic-specific preference embedding from
voting data, each on both the base and fine-tuned encoder:

  1. PSD head (rank=20)              — Mahalanobis metric, 15K params
  2. Two-tower MLP (hidden=256, out=20) — nonlinear per-text embedding

Crossed with:
  A. frozen BASE encoder (sentence-T5-XL)
  B. frozen FINE-TUNED encoder (our Section 4-5 method)

For each of 4 (method × encoder) pairs, sweeps a modest hyperparameter grid
(LR, weight decay, epochs) per dataset, trains probe on val split, evaluates
on test split. Records all results for aggregation/analysis.

Output: JSON with per-dataset per-method best config + full grid.

Usage:
    python scripts/supervised_embedding_sweep.py \\
        --tuned-model data/models/best/.../targeted_sweep/lr1p25eem4_n750_r16_a48 \\
        --output data/results/supervised_sweep.json
"""
import argparse
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ProjectPaths
from src.embedding.model import get_device, load_model
from src.evaluation.evaluator import EVAL_DATASETS, cosine_similarity
from src.evaluation.probes import ProbeTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# 3-way participant split (disjoint participants across train/val/test).
# Statements are shared (all participants vote on a common per-dataset pool);
# deployment story: onboard new voters given some labeled votes.
SPLIT_SEED = 0
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2


def three_way_split(participant_ids, seed: int = SPLIT_SEED):
    ids = sorted(set(participant_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    return {
        "train": set(ids[:n_train]),
        "val": set(ids[n_train:n_train + n_val]),
        "test": set(ids[n_train + n_val:]),
    }


def load_eval_split(eval_dir: Path, dataset: str):
    data_path = eval_dir / f"{dataset}.jsonl"
    if not data_path.exists(): return None
    triplets = [json.loads(l) for l in open(data_path)]
    all_pids = [t["participant_id"] for t in triplets]
    splits = three_way_split(all_pids)
    out = {}
    for name, pid_set in splits.items():
        subset = [t for t in triplets if t["participant_id"] in pid_set]
        if not subset: return None
        out[name] = subset
    return out


def embed_triplets(triplets, model, max_triplets: int, seed: int = 42):
    if not triplets: return None
    if len(triplets) > max_triplets:
        rng = random.Random(seed)
        triplets = rng.sample(triplets, max_triplets)

    all_texts = set()
    for t in triplets:
        all_texts.update(t["anchor_texts"])
        all_texts.add(t["preferred"]); all_texts.add(t["dispreferred"])
    all_texts = list(all_texts)
    embs = model.encode(all_texts, convert_to_numpy=True,
                         show_progress_bar=False, batch_size=64)
    t2e = dict(zip(all_texts, embs))

    anchor_cache, a_list, p_list, d_list = {}, [], [], []
    for t in triplets:
        pid = t["participant_id"]
        if pid not in anchor_cache:
            embs_a = [t2e[x] for x in t["anchor_texts"] if x in t2e]
            if not embs_a: continue
            a = np.mean(embs_a, axis=0)
            if np.linalg.norm(a) == 0: continue
            anchor_cache[pid] = a
        a = anchor_cache.get(pid)
        if a is None: continue
        p = t2e.get(t["preferred"]); d = t2e.get(t["dispreferred"])
        if p is None or d is None: continue
        a_list.append(a); p_list.append(p); d_list.append(d)
    if not a_list: return None
    return np.array(a_list), np.array(p_list), np.array(d_list)


def cosine_accuracy(a, p, d):
    correct = 0
    for ai, pi, di in zip(a, p, d):
        ps = cosine_similarity(ai, pi); ds = cosine_similarity(ai, di)
        if ps > ds: correct += 1
        elif ps == ds: correct += 0.5
    return correct / len(a)


def train_probe(train_data, val_data, test_data, probe_type, lr, epochs, wd,
                 device, batch_size=512):
    trainer = ProbeTrainer(
        probe_type=probe_type, embedding_dim=train_data[0].shape[1],
        lr=lr, epochs=epochs, batch_size=batch_size,
        device=device, weight_decay=wd,
    )
    trainer.train(*train_data)
    return {
        "train": trainer.triplet_accuracy(*train_data),
        "val": trainer.triplet_accuracy(*val_data),
        "test": trainer.triplet_accuracy(*test_data),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="sentence-transformers/sentence-t5-xl")
    ap.add_argument("--tuned-model", required=True)
    ap.add_argument("--max-train", type=int, default=10000)
    ap.add_argument("--max-val", type=int, default=3000)
    ap.add_argument("--max-test", type=int, default=5000)
    ap.add_argument("--output", default="data/results/supervised_sweep.json")
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    device = get_device()
    probe_device = "cuda" if str(device) != "cpu" else "cpu"

    # Methods: probe_type labels matching PROBE_REGISTRY resolution
    methods = {
        "psd_r20":            "psd_r20",
        "two_tower_h256_o20": "two_tower_h256_o20",
    }

    # Modest hyperparameter grid
    grid = [
        dict(lr=3e-4, epochs=30,  wd=0.0),
        dict(lr=3e-4, epochs=30,  wd=1e-3),
        dict(lr=3e-4, epochs=100, wd=0.0),
        dict(lr=1e-3, epochs=30,  wd=0.0),
        dict(lr=1e-3, epochs=30,  wd=1e-3),
        dict(lr=1e-3, epochs=100, wd=0.0),
        dict(lr=1e-3, epochs=100, wd=1e-3),
        dict(lr=3e-3, epochs=30,  wd=1e-3),
    ]

    all_results = {}

    for encoder_label, encoder_path in [
        ("base", args.base_model),
        ("tuned", args.tuned_model),
    ]:
        logger.info(f"\n{'='*60}\nEncoder: {encoder_label} = {encoder_path}")
        model = load_model(encoder_path, device=device)

        # Embed every dataset's train/val/test once; reuse across methods/configs
        embedded = {}
        for ds in EVAL_DATASETS:
            logger.info(f"  embedding {ds}")
            splits = load_eval_split(paths.eval_dir, ds)
            if splits is None: continue
            tr = embed_triplets(splits["train"], model, max_triplets=args.max_train)
            vd = embed_triplets(splits["val"],   model, max_triplets=args.max_val)
            td = embed_triplets(splits["test"],  model, max_triplets=args.max_test)
            if tr is None or vd is None or td is None: continue
            embedded[ds] = {"train": tr, "val": vd, "test": td,
                             "cosine_val":  cosine_accuracy(*vd),
                             "cosine_test": cosine_accuracy(*td)}

        # Run each method × hyperparam × dataset. Train on train, select on val,
        # report test. No test-set leakage into hyperparam selection.
        for method_label, probe_type in methods.items():
            logger.info(f"\n--- method: {method_label} on {encoder_label} ---")
            for ds, data in embedded.items():
                grid_results = []
                for hp in grid:
                    res = train_probe(
                        data["train"], data["val"], data["test"], probe_type,
                        lr=hp["lr"], epochs=hp["epochs"], wd=hp["wd"],
                        device=probe_device,
                    )
                    grid_results.append({**hp, **res})
                # Select on val, report corresponding test.
                best = max(grid_results, key=lambda r: r["val"])
                logger.info(f"  {ds:<42}  cos_test={data['cosine_test']:.3f}  "
                            f"val={best['val']:.3f}  test={best['test']:.3f}  "
                            f"(lr={best['lr']}, epochs={best['epochs']}, wd={best['wd']})")
                all_results[f"{encoder_label}__{method_label}__{ds}"] = {
                    "encoder": encoder_label,
                    "method": method_label,
                    "dataset": ds,
                    "cosine_val": data["cosine_val"],
                    "cosine_test": data["cosine_test"],
                    "grid": grid_results,
                    "best": best,
                }

        del model
        torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "tuned_model": args.tuned_model,
            "methods": list(methods.keys()),
            "grid": grid,
            "results": all_results,
        }, f, indent=2)
    logger.info(f"\nSaved to {args.output}")

    # Print summary table
    print("\n" + "="*100)
    print(f"{'dataset':<42} {'cos':>6} {'base/psd':>10} {'base/2twr':>10} "
          f"{'tuned/psd':>10} {'tuned/2twr':>10}")
    print("-"*100)
    for ds in EVAL_DATASETS:
        cells = [ds]
        cos_val = None
        for enc in ["base", "tuned"]:
            for m in ["psd_r20", "two_tower_h256_o20"]:
                k = f"{enc}__{m}__{ds}"
                r = all_results.get(k)
                if r:
                    cos_val = r["cosine_test"]
                    cells.append(f"{r['best']['test']:.3f}")
                else:
                    cells.append("--")
        print(f"{cells[0]:<42} {cos_val or 0:>6.3f} {cells[1]:>10} {cells[2]:>10} "
              f"{cells[3]:>10} {cells[4]:>10}")


if __name__ == "__main__":
    main()
