#!/usr/bin/env python3
"""Per-topic rank-20 probe sweep on a frozen encoder.

One script, three scorers, all over Bradley-Terry on the same per-participant
3-way split with the same LR/wd/epoch grid:

  --scorer metric        -||L^T psi(a) - L^T psi(j)||^2
                         (ideal-point utility; main scorer in the paper)
  --scorer cosine        T * cos(L^T psi(a), L^T psi(j))
                         (drops the item-norm term; sweeps T as well)
  --scorer metric_bias   -||L^T psi(a) - L^T psi(j)||^2 + w^T psi(j)
                         (metric + free per-item linear bias direction)

Output JSON is identical in shape across scorers (best_hp, val, test,
hard_acc, hard_n, grid) so downstream plotting / table code is agnostic.
The saved L is the projection only; for metric_bias, the hard-triplet
eval reconstructs the probe from L alone (i.e., the bias direction is
not applied at hard-eval time, preserving the original behaviour).
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

SPLIT_SEED = 0
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
RANK = 20
BATCH_SIZE = 1024

_BASE_GRID = [
    dict(lr=3e-4, epochs=30,  wd=0.0),
    dict(lr=3e-4, epochs=30,  wd=1e-3),
    dict(lr=3e-4, epochs=100, wd=0.0),
    dict(lr=1e-3, epochs=30,  wd=0.0),
    dict(lr=1e-3, epochs=30,  wd=1e-3),
    dict(lr=1e-3, epochs=100, wd=0.0),
    dict(lr=1e-3, epochs=100, wd=1e-3),
    dict(lr=3e-3, epochs=30,  wd=1e-3),
]

# Per-scorer hyperparameter grid. metric / metric_bias have unbounded
# squared-distance scores, so T=1 is fine; cosine is bounded in [-1, 1]
# and gradients starve at T=1, so we sweep T.
GRIDS = {
    "metric":      [{**hp, "T": 1.0} for hp in _BASE_GRID],
    "metric_bias": [{**hp, "T": 1.0} for hp in _BASE_GRID],
    "cosine":      [{**hp, "T": T} for T in (1.0, 10.0, 20.0) for hp in _BASE_GRID],
}

DEFAULT_OUTPUTS = {
    "metric":      "data/results/metric_probe_sweep.json",
    "cosine":      "data/results/cosine_probe_sweep.json",
    "metric_bias": "data/results/metric_bias_probe_sweep.json",
}


class MetricProbe(nn.Module):
    """score(a,j) = -T * ||L^T a - L^T j||^2."""
    def __init__(self, dim, rank=RANK, temperature=1.0):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.T = temperature

    def forward(self, a, b):
        za, zb = a @ self.L, b @ self.L
        return -self.T * ((za - zb) ** 2).sum(dim=-1)


class CosineProbe(nn.Module):
    """score(a,j) = T * cos(L^T a, L^T j)."""
    def __init__(self, dim, rank=RANK, temperature=1.0):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.T = temperature

    def forward(self, a, b):
        za, zb = a @ self.L, b @ self.L
        return self.T * F.cosine_similarity(za, zb, dim=-1)


class MetricBiasProbe(nn.Module):
    """score(a,j) = T * (-||L^T a - L^T j||^2 + w^T j).

    Free per-item linear bias direction in addition to the tied metric.
    """
    def __init__(self, dim, rank=RANK, temperature=1.0):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.w = nn.Parameter(torch.zeros(dim))
        self.T = temperature

    def forward(self, a, b):
        za, zb = a @ self.L, b @ self.L
        return self.T * (-((za - zb) ** 2).sum(dim=-1) + b @ self.w)


PROBE_CLASSES = {"metric": MetricProbe, "cosine": CosineProbe, "metric_bias": MetricBiasProbe}


def three_way_split(pids, seed=SPLIT_SEED):
    ids = sorted(set(pids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * TRAIN_FRAC); n_val = int(n * VAL_FRAC)
    return {"train": set(ids[:n_train]),
            "val":   set(ids[n_train:n_train + n_val]),
            "test":  set(ids[n_train + n_val:])}


def embed_triplets(triplets, model, cap=None, seed=42):
    if not triplets:
        return None
    if cap and len(triplets) > cap:
        triplets = random.Random(seed).sample(triplets, cap)
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
        return (probe(at, pt) > probe(at, nt)).float().mean().item()


def train_one_config(probe_cls, train_data, val_data, test_data, hp, dim, device):
    probe = probe_cls(dim=dim, rank=RANK, temperature=hp["T"]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    A, P, N = train_data
    ds = TensorDataset(torch.tensor(A, dtype=torch.float32, device=device),
                       torch.tensor(P, dtype=torch.float32, device=device),
                       torch.tensor(N, dtype=torch.float32, device=device))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(hp["epochs"]):
        probe.train()
        for a, p, n in loader:
            loss = -F.logsigmoid(probe(a, p) - probe(a, n)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return {
        "val":  triplet_acc(probe, *val_data, device),
        "test": triplet_acc(probe, *test_data, device) if test_data else None,
        "L":    probe.L.detach().cpu().numpy().copy(),
    }


def hard_triplet_acc(dataset, L, probe_cls, model, device, hard_by_ds):
    """Reconstruct probe from saved L (only) and score the dataset's hard triplets."""
    hard = hard_by_ds.get(dataset, [])
    if not hard:
        return None, 0
    texts = list({t["anchor"] for t in hard}
                 | {t["preference_match"] for t in hard}
                 | {t["semantic_distractor"] for t in hard})
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    t2e = dict(zip(texts, embs))
    probe = probe_cls(dim=L.shape[0], rank=RANK).to(device)
    with torch.no_grad():
        probe.L.copy_(torch.tensor(L, dtype=torch.float32, device=device))
    n_correct = 0
    for t in hard:
        a = torch.tensor(t2e[t["anchor"]], dtype=torch.float32, device=device).unsqueeze(0)
        p = torch.tensor(t2e[t["preference_match"]], dtype=torch.float32, device=device).unsqueeze(0)
        n = torch.tensor(t2e[t["semantic_distractor"]], dtype=torch.float32, device=device).unsqueeze(0)
        if probe(a, p).item() > probe(a, n).item():
            n_correct += 1
    return n_correct / len(hard), len(hard)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scorer", required=True, choices=sorted(PROBE_CLASSES))
    ap.add_argument("--encoder", default="sentence-transformers/sentence-t5-xl")
    ap.add_argument("--hard-triplets",
                    default="data/processed/triplets/hard_eval_triplets_1k.jsonl")
    ap.add_argument("--output", default=None,
                    help="Output JSON path (default: data/results/{scorer}_probe_sweep.json)")
    ap.add_argument("--max-train-triplets", type=int, default=50000)
    args = ap.parse_args()

    output = args.output or DEFAULT_OUTPUTS[args.scorer]
    grid = GRIDS[args.scorer]
    probe_cls = PROBE_CLASSES[args.scorer]
    logger.info(f"scorer={args.scorer}  grid_size={len(grid)}  output={output}")

    paths = ProjectPaths.auto()
    device = get_device()
    logger.info(f"Loading {args.encoder} on {device}")
    model = load_model(args.encoder, device=device)
    dim = model.get_sentence_embedding_dimension()

    hard_by_ds = {}
    for line in open(args.hard_triplets):
        t = json.loads(line)
        hard_by_ds.setdefault(t["dataset"], []).append(t)

    results = {}
    L_dir = Path("data/results/psd_probe_weights")
    datasets = sorted(p.stem.replace("L_perds_", "") for p in L_dir.glob("L_perds_*.npy"))
    for ds in datasets:
        path = Path(paths.eval_dir) / f"{ds}.jsonl"
        if not path.exists():
            logger.info(f"  skip {ds}: no eval file"); continue
        triplets = [json.loads(l) for l in open(path)]
        splits = three_way_split([t["participant_id"] for t in triplets])
        train_trs = [t for t in triplets if t["participant_id"] in splits["train"]]
        val_trs   = [t for t in triplets if t["participant_id"] in splits["val"]]
        test_trs  = [t for t in triplets if t["participant_id"] in splits["test"]]
        if not (train_trs and val_trs and test_trs):
            logger.info(f"  skip {ds}: empty split"); continue

        train_data = embed_triplets(train_trs, model, cap=args.max_train_triplets)
        val_data   = embed_triplets(val_trs,   model, cap=10000)
        test_data  = embed_triplets(test_trs,  model, cap=10000)
        if not (train_data and val_data and test_data):
            continue

        logger.info(f"\n=== {ds} ===  n(train/val/test)="
                    f"{len(train_data[0])}/{len(val_data[0])}/{len(test_data[0])}")
        grid_results = []
        for hp in grid:
            r = train_one_config(probe_cls, train_data, val_data, test_data, hp, dim, device)
            grid_results.append({**hp, "val": r["val"], "test": r["test"], "L": r["L"]})
            logger.info(f"  hp={hp}  val={r['val']:.3f}  test={r['test']:.3f}")

        best = max(grid_results, key=lambda r: r["val"])
        hard_acc, hard_n = hard_triplet_acc(ds, best["L"], probe_cls, model, device, hard_by_ds)
        results[ds] = {
            "best_hp": {k: best[k] for k in ("lr", "epochs", "wd", "T")},
            "val":  best["val"],
            "test": best["test"],
            "hard_acc": hard_acc, "hard_n": hard_n,
            "grid": [{k: r[k] for k in ("lr", "epochs", "wd", "T", "val", "test")} for r in grid_results],
        }
        logger.info(f"  BEST  test={best['test']:.3f}  hard={hard_acc if hard_acc else 0:.3f}")

    test_vals = [r["test"]     for r in results.values() if r["test"]     is not None]
    hard_vals = [r["hard_acc"] for r in results.values() if r["hard_acc"] is not None]
    logger.info(f"\nNatural test macro: {np.mean(test_vals)*100:.1f}  (n={len(test_vals)})")
    logger.info(f"Hard    macro:      {np.mean(hard_vals)*100:.1f}  (n={len(hard_vals)})")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({
            "scorer": args.scorer,
            "per_dataset": results,
            "natural_macro": float(np.mean(test_vals)) if test_vals else None,
            "hard_macro":    float(np.mean(hard_vals)) if hard_vals else None,
        }, f, indent=2)
    logger.info(f"Saved {output}")


if __name__ == "__main__":
    main()
