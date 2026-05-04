#!/usr/bin/env python3
"""Data efficiency curve for the supervised PSD probe.

For each of 11 eval datasets, sweeps K (number of labeled training triplets)
from K=20 up to the full per-dataset train pool, fits a rank-20 PSD head on
K subsampled triplets (from the train split of a 3-way participant split),
selects hyperparams on val, reports test accuracy.

Output: JSON with per-dataset per-K curves for plotting.

Usage:
    python scripts/data_efficiency_sweep.py \\
        --output data/results/data_efficiency.json
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

SPLIT_SEED = 0
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2

# K grid — labeled training triplets. Clamped per dataset to available train size.
K_GRID = [20, 50, 100, 200, 500, 1000, 2500, 5000, 10000]

# Hyperparameter grid (reduced from main sweep — just the common winners)
HP_GRID = [
    dict(lr=3e-4, epochs=30,  wd=0.0),
    dict(lr=1e-3, epochs=30,  wd=1e-3),
    dict(lr=1e-3, epochs=100, wd=0.0),
    dict(lr=1e-3, epochs=100, wd=1e-3),
]

# Seeds for K-subsample (avg over these to reduce variance at small K)
SUBSAMPLE_SEEDS = [0, 1, 2]


def three_way_split(participant_ids, seed: int = SPLIT_SEED):
    ids = sorted(set(participant_ids))
    rng = random.Random(seed); rng.shuffle(ids)
    n = len(ids); nt = int(n * TRAIN_FRAC); nv = int(n * VAL_FRAC)
    return {
        "train": set(ids[:nt]),
        "val":   set(ids[nt:nt + nv]),
        "test":  set(ids[nt + nv:]),
    }


def load_splits(eval_dir: Path, dataset: str):
    data_path = eval_dir / f"{dataset}.jsonl"
    if not data_path.exists(): return None
    triplets = [json.loads(l) for l in open(data_path)]
    pid_split = three_way_split([t["participant_id"] for t in triplets])
    out = {}
    for name, pids in pid_split.items():
        subset = [t for t in triplets if t["participant_id"] in pids]
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
    c = 0
    for ai, pi, di in zip(a, p, d):
        ps = cosine_similarity(ai, pi); ds = cosine_similarity(ai, di)
        if ps > ds: c += 1
        elif ps == ds: c += 0.5
    return c / len(a)


def subsample(data, K, seed):
    a, p, d = data
    n = len(a)
    if K >= n: return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=K, replace=False)
    return a[idx], p[idx], d[idx]


def train_and_score(train_data, val_data, test_data, hp, device):
    trainer = ProbeTrainer(
        probe_type="psd_r20",
        embedding_dim=train_data[0].shape[1],
        lr=hp["lr"], epochs=hp["epochs"], batch_size=512,
        device=device, weight_decay=hp["wd"],
    )
    trainer.train(*train_data)
    return {
        "val":  trainer.triplet_accuracy(*val_data),
        "test": trainer.triplet_accuracy(*test_data),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="sentence-transformers/sentence-t5-xl")
    ap.add_argument("--max-train", type=int, default=10000)
    ap.add_argument("--max-val",   type=int, default=3000)
    ap.add_argument("--max-test",  type=int, default=5000)
    ap.add_argument("--output", default="data/results/data_efficiency.json")
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    device = get_device()
    probe_device = "cuda" if str(device) != "cpu" else "cpu"

    model = load_model(args.encoder, device=device)

    results = {}
    for ds in EVAL_DATASETS:
        logger.info(f"\n=== {ds} ===")
        splits = load_splits(paths.eval_dir, ds)
        if splits is None:
            logger.info("  (missing data)"); continue

        train_e = embed_triplets(splits["train"], model, args.max_train)
        val_e   = embed_triplets(splits["val"],   model, args.max_val)
        test_e  = embed_triplets(splits["test"],  model, args.max_test)
        if train_e is None or val_e is None or test_e is None:
            logger.info("  (embed failed)"); continue

        n_train = len(train_e[0])
        cos_test = cosine_accuracy(*test_e)
        logger.info(f"  n_train={n_train}  cos_test={cos_test:.3f}")

        ds_curve = []
        for K in K_GRID:
            K_use = min(K, n_train)
            # Average over subsample seeds (at small K, one sample is noisy)
            seed_results = []
            for s in SUBSAMPLE_SEEDS:
                sub = subsample(train_e, K_use, seed=s)
                grid = []
                for hp in HP_GRID:
                    r = train_and_score(sub, val_e, test_e, hp, probe_device)
                    grid.append({**hp, **r})
                best = max(grid, key=lambda x: x["val"])
                seed_results.append(best)
                if K_use == n_train: break  # no subsampling variance at full K
            mean_val = np.mean([r["val"] for r in seed_results])
            mean_test = np.mean([r["test"] for r in seed_results])
            std_test = np.std([r["test"] for r in seed_results]) if len(seed_results) > 1 else 0.0
            logger.info(f"  K={K_use:>5}  val={mean_val:.3f}  test={mean_test:.3f} ± {std_test:.3f}")
            ds_curve.append({
                "K": K_use,
                "mean_val": float(mean_val),
                "mean_test": float(mean_test),
                "std_test": float(std_test),
                "seed_results": seed_results,
            })
            if K_use == n_train: break

        results[ds] = {
            "n_train": n_train,
            "cos_test": cos_test,
            "curve": ds_curve,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "encoder": args.encoder,
            "k_grid": K_GRID,
            "hp_grid": HP_GRID,
            "subsample_seeds": SUBSAMPLE_SEEDS,
            "results": results,
        }, f, indent=2)
    logger.info(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
