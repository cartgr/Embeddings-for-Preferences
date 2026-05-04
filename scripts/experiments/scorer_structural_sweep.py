#!/usr/bin/env python3
"""Structural-commitment sweep for the ideal-point metric scorer.

Tests what pieces of score(a,j) = -||L^T psi(a) - L^T psi(j)||^2 are
load-bearing, by holding everything else fixed and relaxing one
structural commitment at a time. For every variant we:

  - sweep LR/wd/epochs over the same grid as supervised_probe_sweep.py
  - run across 11 eval datasets with the same 3-way participant split
  - repeat over N_SEEDS random seeds
  - evaluate on the natural-data test split AND on the dataset's hard
    triplets (when available)
  - val-select hyperparameters per (dataset, seed) and report
    test/hard accuracy at the selected config

Variants (numbers match Claims 1-8 from the experimental design):

  metric       (baseline): -||L^T psi(a) - L^T psi(j)||^2
  nonlinear    (C1):       -||phi(psi(a)) - phi(psi(j))||^2
  rank_<r>     (C2):       metric at rank r in {1, 2, 5, 10, 50, 100, d}
  asymmetric   (C3):       -||L_a^T psi(a) - L_j^T psi(j)||^2
  inner_prod   (C4):       <L^T psi(a), L^T psi(j)>
  anisotropic  (C5):       -sum_k m_k (pa_k - pj_k)^2 with learnable m
  item_bias    (C6):       metric + w^T psi(j)
  salience     (C8):       -sum_k w_k(psi(a)) (pa_k - pj_k)^2 with w = f(a)

Output: data/results/scorer_structural_ablation.json (resumable: skips
(variant, seed) pairs already present).
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import ProjectPaths
from src.embedding.model import get_device, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Shared with supervised_probe_sweep.py
SPLIT_SEED = 0
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2

GRID = [
    dict(lr=3e-4, epochs=30,  wd=0.0),
    dict(lr=3e-4, epochs=30,  wd=1e-3),
    dict(lr=3e-4, epochs=100, wd=0.0),
    dict(lr=1e-3, epochs=30,  wd=0.0),
    dict(lr=1e-3, epochs=30,  wd=1e-3),
    dict(lr=1e-3, epochs=100, wd=0.0),
    dict(lr=1e-3, epochs=100, wd=1e-3),
    dict(lr=3e-3, epochs=30,  wd=1e-3),
]
RANK = 20
BATCH_SIZE = 1024
N_SEEDS = 5
SEED_LIST = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------
# Probe classes. All forward(a, j) -> scalar score per row.
# ---------------------------------------------------------------------

class Metric(nn.Module):
    def __init__(self, dim, rank=RANK):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)
    def forward(self, a, j):
        pa, pj = a @ self.L, j @ self.L
        return -((pa - pj) ** 2).sum(dim=-1)


class Nonlinear(nn.Module):
    def __init__(self, dim, rank=RANK, hidden=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, rank),
        )
    def forward(self, a, j):
        pa, pj = self.phi(a), self.phi(j)
        return -((pa - pj) ** 2).sum(dim=-1)


class Asymmetric(nn.Module):
    def __init__(self, dim, rank=RANK):
        super().__init__()
        self.La = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.Lj = nn.Parameter(torch.randn(dim, rank) * 0.01)
    def forward(self, a, j):
        pa, pj = a @ self.La, j @ self.Lj
        return -((pa - pj) ** 2).sum(dim=-1)


class InnerProduct(nn.Module):
    def __init__(self, dim, rank=RANK):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)
    def forward(self, a, j):
        pa, pj = a @ self.L, j @ self.L
        return (pa * pj).sum(dim=-1)


class Anisotropic(nn.Module):
    def __init__(self, dim, rank=RANK):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.log_m = nn.Parameter(torch.zeros(rank))  # softplus → weights
    def forward(self, a, j):
        pa, pj = a @ self.L, j @ self.L
        diff = pa - pj
        m = F.softplus(self.log_m) + 1e-6
        return -(diff * diff * m).sum(dim=-1)


class ItemBias(nn.Module):
    def __init__(self, dim, rank=RANK):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.w = nn.Parameter(torch.zeros(dim))
    def forward(self, a, j):
        pa, pj = a @ self.L, j @ self.L
        metric = -((pa - pj) ** 2).sum(dim=-1)
        bias = j @ self.w
        return metric + bias


class Salience(nn.Module):
    """Per-user salience weights w(psi(a)) on each projected dimension."""
    def __init__(self, dim, rank=RANK):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.W = nn.Linear(dim, rank)
    def forward(self, a, j):
        pa, pj = a @ self.L, j @ self.L
        w = F.softplus(self.W(a)) + 1e-6  # (batch, rank)
        diff = pa - pj
        return -(diff * diff * w).sum(dim=-1)


def build_probe(name, dim, rank=RANK):
    if name == "metric":       return Metric(dim, rank)
    if name == "nonlinear":    return Nonlinear(dim, rank)
    if name == "asymmetric":   return Asymmetric(dim, rank)
    if name == "inner_prod":   return InnerProduct(dim, rank)
    if name == "anisotropic":  return Anisotropic(dim, rank)
    if name == "item_bias":    return ItemBias(dim, rank)
    if name == "salience":     return Salience(dim, rank)
    if name.startswith("rank_"):
        r = int(name.split("_")[1])
        return Metric(dim, rank=r)
    raise ValueError(f"unknown variant {name!r}")


VARIANTS = [
    "metric", "nonlinear", "asymmetric", "inner_prod",
    "anisotropic", "item_bias", "salience",
    # Rank sweep (claim 2)
    "rank_1", "rank_2", "rank_5", "rank_10", "rank_50", "rank_100",
]


# ---------------------------------------------------------------------
# Data + training
# ---------------------------------------------------------------------

def three_way_split(pids, seed=SPLIT_SEED):
    ids = sorted(set(pids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    return {
        "train": set(ids[:n_train]),
        "val":   set(ids[n_train:n_train + n_val]),
        "test":  set(ids[n_train + n_val:]),
    }


def embed_triplets(triplets, model, cap=None, seed=42):
    if not triplets: return None
    if cap and len(triplets) > cap:
        rng = random.Random(seed)
        triplets = rng.sample(triplets, cap)
    texts = set()
    for t in triplets:
        texts.update(t["anchor_texts"])
        texts.add(t["preferred"]); texts.add(t["dispreferred"])
    texts = list(texts)
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    t2e = dict(zip(texts, embs))
    A, P, N = [], [], []
    for t in triplets:
        a = np.mean([t2e[x] for x in t["anchor_texts"] if x in t2e], axis=0)
        if np.linalg.norm(a) == 0: continue
        A.append(a); P.append(t2e[t["preferred"]]); N.append(t2e[t["dispreferred"]])
    return np.stack(A), np.stack(P), np.stack(N)


def triplet_acc(probe, a, p, n, device):
    probe.eval()
    with torch.no_grad():
        at = torch.tensor(a, dtype=torch.float32, device=device)
        pt = torch.tensor(p, dtype=torch.float32, device=device)
        nt = torch.tensor(n, dtype=torch.float32, device=device)
        sp = probe(at, pt); sn = probe(at, nt)
        return (sp > sn).float().mean().item()


def train_one_config(variant, train_data, val_data, test_data, hp, dim, seed, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    probe = build_probe(variant, dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=hp["lr"], weight_decay=hp["wd"])

    A, P, N = train_data
    ds = TensorDataset(torch.tensor(A, dtype=torch.float32, device=device),
                       torch.tensor(P, dtype=torch.float32, device=device),
                       torch.tensor(N, dtype=torch.float32, device=device))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(hp["epochs"]):
        probe.train()
        for a, p, n in loader:
            sp = probe(a, p); sn = probe(a, n)
            loss = -F.logsigmoid(sp - sn).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    val_acc = triplet_acc(probe, *val_data, device)
    test_acc = triplet_acc(probe, *test_data, device)
    return probe, val_acc, test_acc


def precompute_hard_cache(hard_by_ds, model, device):
    """Embed every unique hard-triplet text once per dataset, return tensors
    (anchor, match, distractor) stacked ready for probe forward calls."""
    cache = {}
    for ds, hard in hard_by_ds.items():
        if not hard: continue
        texts = list({t["anchor"] for t in hard}
                     | {t["preference_match"] for t in hard}
                     | {t["semantic_distractor"] for t in hard})
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        t2e = dict(zip(texts, embs))
        A = torch.tensor(np.stack([t2e[t["anchor"]]             for t in hard]),
                         dtype=torch.float32, device=device)
        P = torch.tensor(np.stack([t2e[t["preference_match"]]   for t in hard]),
                         dtype=torch.float32, device=device)
        N = torch.tensor(np.stack([t2e[t["semantic_distractor"]]for t in hard]),
                         dtype=torch.float32, device=device)
        cache[ds] = (A, P, N)
    return cache


def hard_acc_for_probe(probe, dataset, hard_cache):
    tensors = hard_cache.get(dataset)
    if tensors is None: return None, 0
    A, P, N = tensors
    probe.eval()
    with torch.no_grad():
        sp = probe(A, P); sn = probe(A, N)
        return float((sp > sn).float().mean().item()), int(A.shape[0])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="sentence-transformers/sentence-t5-xl")
    ap.add_argument("--hard-triplets",
                    default="data/processed/triplets/hard_eval_triplets_1k.jsonl")
    ap.add_argument("--output", default="data/results/scorer_structural_ablation.json")
    ap.add_argument("--max-train-triplets", type=int, default=30000)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Subset of variants to run. Default = all.")
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    device = get_device()
    logger.info(f"Loading {args.encoder} on {device}")
    model = load_model(args.encoder, device=device)
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"embedding dim={dim}")

    hard_by_ds = {}
    for l in open(args.hard_triplets):
        t = json.loads(l)
        hard_by_ds.setdefault(t["dataset"], []).append(t)

    datasets = sorted([p.stem.replace("L_perds_", "")
                       for p in Path("data/results/psd_probe_weights").glob("L_perds_*.npy")])
    logger.info(f"datasets: {datasets}")

    variants_to_run = args.variants or VARIANTS
    logger.info(f"variants: {variants_to_run}")

    # Pre-embed all datasets once (base encoder outputs are cached in RAM)
    cache = {}
    for ds in datasets:
        path = Path(paths.eval_dir) / f"{ds}.jsonl"
        if not path.exists(): continue
        triplets = [json.loads(l) for l in open(path)]
        splits = three_way_split([t["participant_id"] for t in triplets])
        train_trs = [t for t in triplets if t["participant_id"] in splits["train"]]
        val_trs   = [t for t in triplets if t["participant_id"] in splits["val"]]
        test_trs  = [t for t in triplets if t["participant_id"] in splits["test"]]
        if not (train_trs and val_trs and test_trs): continue
        cache[ds] = {
            "train": embed_triplets(train_trs, model, cap=args.max_train_triplets),
            "val":   embed_triplets(val_trs, model, cap=10000),
            "test":  embed_triplets(test_trs, model, cap=10000),
        }
        logger.info(f"  cached {ds}: n(train/val/test)="
                    f"{len(cache[ds]['train'][0])}/{len(cache[ds]['val'][0])}/{len(cache[ds]['test'][0])}")

    logger.info("Pre-embedding hard triplets...")
    hard_cache = precompute_hard_cache(hard_by_ds, model, device)
    logger.info(f"  cached hard triplets for {len(hard_cache)} datasets "
                f"({sum(A.shape[0] for A, _, _ in hard_cache.values())} total)")

    # Resume
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        results = json.load(open(out_path))
    else:
        results = {}

    for variant in variants_to_run:
        if variant not in results: results[variant] = {}
        for ds in datasets:
            if ds not in cache: continue
            if ds not in results[variant]: results[variant][ds] = {}
            for seed in SEED_LIST:
                seed_key = str(seed)
                if seed_key in results[variant][ds]:
                    continue
                logger.info(f"=== variant={variant}  ds={ds}  seed={seed} ===")
                best = None
                for hp in GRID:
                    probe, val_acc, test_acc = train_one_config(
                        variant, cache[ds]["train"], cache[ds]["val"], cache[ds]["test"],
                        hp, dim, seed, device)
                    if best is None or val_acc > best["val"]:
                        best = {"hp": hp, "val": val_acc, "test": test_acc, "probe": probe}
                hard_acc, hard_n = hard_acc_for_probe(best["probe"], ds, hard_cache)
                results[variant][ds][seed_key] = {
                    "best_hp": best["hp"],
                    "val": best["val"],
                    "test": best["test"],
                    "hard_acc": hard_acc,
                    "hard_n": hard_n,
                }
                logger.info(f"  best val={best['val']:.3f}  test={best['test']:.3f}  hard={hard_acc if hard_acc else 0:.3f}")
                # Save after every (variant, ds, seed) triple
                with open(out_path, "w") as f:
                    # Make serializable (drop the probe object)
                    sanitized = {v: {d: {s: {kk: vv for kk, vv in r.items()}
                                           for s, r in seeds.items()}
                                      for d, seeds in ds_dict.items()}
                                 for v, ds_dict in results.items()}
                    json.dump(sanitized, f, indent=2)

    # Summary
    logger.info("\n=== Summary (macro-mean over datasets, mean±std over seeds) ===")
    for variant in variants_to_run:
        if variant not in results: continue
        # Per-seed macro means
        test_per_seed = {s: [] for s in SEED_LIST}
        hard_per_seed = {s: [] for s in SEED_LIST}
        for ds, seeds in results[variant].items():
            for seed in SEED_LIST:
                sk = str(seed)
                if sk not in seeds: continue
                test_per_seed[seed].append(seeds[sk]["test"])
                if seeds[sk]["hard_acc"] is not None:
                    hard_per_seed[seed].append(seeds[sk]["hard_acc"])
        test_means = [np.mean(v) for v in test_per_seed.values() if v]
        hard_means = [np.mean(v) for v in hard_per_seed.values() if v]
        if not test_means: continue
        logger.info(f"  {variant:<14}  nat={np.mean(test_means)*100:.1f}±{np.std(test_means)*100:.2f}"
                    f"  hard={np.mean(hard_means)*100:.1f}±{np.std(hard_means)*100:.2f}")


if __name__ == "__main__":
    main()
