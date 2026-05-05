#!/usr/bin/env python3
"""Decompose the rank-20 metric-probe score margin under base and DPT-tuned
encoders, on every hard-triplet evaluation dataset.

The metric scorer is ``score(a, x) = -||L^T psi(a) - L^T psi(x)||^2``. Its
margin between match ``p`` and distractor ``n`` decomposes as

    score(a, p) - score(a, n)
        = 2 <L^T psi(a), L^T(psi(p) - psi(n))>
            + ||L^T psi(n)||^2 - ||L^T psi(p)||^2
        = inner_product_margin + item_norm_diff.

For every eval dataset we (a) encode all unique texts under each of the two
encoders, (b) val-select a rank-20 metric probe on the natural triplets
following the same protocol as ``scorer_structural_sweep.py``, and (c)
average the inner-product-margin and item-norm-difference terms over the
dataset's share of the 875 hard triplets.

Output: ``data/results/probe_margin_decomposition.json`` of shape

    { <dataset>: {
        "base":  {"ip_margin": float, "norm_diff": float, "n": int},
        "tuned": {"ip_margin": float, "norm_diff": float, "n": int},
        "best_hp": {"base": {...}, "tuned": {...}}
      } }

This file is consumed by ``scripts/build_paper_tables.py`` to emit
``tab:probe-on-tuned-margin``.
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
log = logging.getLogger(__name__)

SPLIT_SEED = 0
TRAIN_FRAC = 0.6
VAL_FRAC   = 0.2
RANK = 20
BATCH_SIZE = 1024
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


class MetricProbe(nn.Module):
    def __init__(self, dim, rank=RANK):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)

    def forward(self, a, b):
        za, zb = a @ self.L, b @ self.L
        return -((za - zb) ** 2).sum(dim=-1)


def three_way_split(pids, seed=SPLIT_SEED):
    ids = sorted(set(pids))
    rng = random.Random(seed); rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * TRAIN_FRAC); n_val = int(n * VAL_FRAC)
    return {"train": set(ids[:n_train]),
            "val":   set(ids[n_train:n_train + n_val]),
            "test":  set(ids[n_train + n_val:])}


def build_apn(triplets, t2e):
    A, P, N = [], [], []
    for t in triplets:
        a = np.mean([t2e[x] for x in t["anchor_texts"] if x in t2e], axis=0)
        if np.linalg.norm(a) == 0:
            continue
        A.append(a); P.append(t2e[t["preferred"]]); N.append(t2e[t["dispreferred"]])
    return np.stack(A), np.stack(P), np.stack(N)


def train_probe(dim, train_data, hp, seed, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    probe = MetricProbe(dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    A, P, N = train_data
    ds = TensorDataset(
        torch.tensor(A, dtype=torch.float32, device=device),
        torch.tensor(P, dtype=torch.float32, device=device),
        torch.tensor(N, dtype=torch.float32, device=device),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(hp["epochs"]):
        probe.train()
        for a, p, n in loader:
            sp, sn = probe(a, p), probe(a, n)
            loss = -F.logsigmoid(sp - sn).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def eval_acc(probe, data, device):
    probe.eval()
    with torch.no_grad():
        a, p, n = data
        at = torch.tensor(a, dtype=torch.float32, device=device)
        pt = torch.tensor(p, dtype=torch.float32, device=device)
        nt = torch.tensor(n, dtype=torch.float32, device=device)
        return (probe(at, pt) > probe(at, nt)).float().mean().item()


def fit_probe_for_dataset(triplets, t2e, dim, seed, device):
    """Val-select an HP from GRID, return (best_L, best_hp, val_acc)."""
    splits = three_way_split([t["participant_id"] for t in triplets])
    train_t = [t for t in triplets if t["participant_id"] in splits["train"]]
    val_t   = [t for t in triplets if t["participant_id"] in splits["val"]]
    if not train_t or not val_t:
        return None, None, None
    train_data = build_apn(train_t, t2e)
    val_data   = build_apn(val_t,   t2e)
    best = (None, None, -1.0)
    for hp in GRID:
        probe = train_probe(dim, train_data, hp, seed, device)
        v = eval_acc(probe, val_data, device)
        if v > best[2]:
            L = probe.L.detach().cpu().numpy().copy()
            best = (L, hp, v)
    return best


def margin_decomp(L, hard_triplets, t2e):
    """Compute mean inner-product margin and item-norm difference under L."""
    A = np.stack([t2e[t["anchor"]]              for t in hard_triplets])
    P = np.stack([t2e[t["preference_match"]]    for t in hard_triplets])
    N = np.stack([t2e[t["semantic_distractor"]] for t in hard_triplets])
    LA = A @ L; LP = P @ L; LN = N @ L
    ip_margin = (2.0 * (LA * (LP - LN)).sum(axis=1))
    norm_diff = (LN ** 2).sum(axis=1) - (LP ** 2).sum(axis=1)
    return float(ip_margin.mean()), float(norm_diff.mean()), len(hard_triplets)


def encode_dataset(model, texts, batch_size=64):
    embs = model.encode(texts, convert_to_numpy=True,
                        show_progress_bar=False, batch_size=batch_size)
    return dict(zip(texts, embs))


def run_one_encoder(encoder_path, datasets, hard_by_ds, paths, seed, device):
    """For each dataset, fit val-selected probe and decompose hard triplets."""
    log.info(f"loading {encoder_path}")
    model = load_model(encoder_path, device=device)
    out = {}
    for ds in datasets:
        nat_path = paths.eval_dir / f"{ds}.jsonl"
        if not nat_path.exists():
            log.warning(f"  skip {ds} (no natural triplets)")
            continue
        if ds not in hard_by_ds or not hard_by_ds[ds]:
            log.warning(f"  skip {ds} (no hard triplets)")
            continue
        natural = [json.loads(l) for l in open(nat_path)]
        hard    = hard_by_ds[ds]
        texts = set()
        for t in natural:
            texts.update(t["anchor_texts"]); texts.add(t["preferred"]); texts.add(t["dispreferred"])
        for t in hard:
            texts.update([t["anchor"], t["preference_match"], t["semantic_distractor"]])
        t2e = encode_dataset(model, sorted(texts))
        dim = next(iter(t2e.values())).shape[0]
        L, hp, val = fit_probe_for_dataset(natural, t2e, dim, seed, device)
        if L is None:
            log.warning(f"  skip {ds} (probe fit failed)")
            continue
        ip, nd, n = margin_decomp(L, hard, t2e)
        out[ds] = {"ip_margin": ip, "norm_diff": nd, "n": n,
                   "val_acc": val, "hp": hp}
        log.info(f"  {ds}: ip={ip:+.3f} norm_diff={nd:+.3f} n={n}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-encoder", default="sentence-transformers/sentence-t5-xl")
    ap.add_argument("--tuned-encoder",
                    default="data/models/best/sentence_transformers_sentence_t5_xl/seed42")
    ap.add_argument("--hard-triplets",
                    default="data/processed/triplets/hard_eval_triplets_1k.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/results/probe_margin_decomposition.json")
    args = ap.parse_args()

    paths  = ProjectPaths.auto()
    device = get_device()

    hard_by_ds = {}
    with open(args.hard_triplets) as f:
        for line in f:
            t = json.loads(line)
            hard_by_ds.setdefault(t["dataset"], []).append(t)
    datasets = sorted(hard_by_ds)

    base_results  = run_one_encoder(args.base_encoder,  datasets, hard_by_ds, paths, args.seed, device)
    tuned_results = run_one_encoder(args.tuned_encoder, datasets, hard_by_ds, paths, args.seed, device)

    out = {}
    for ds in datasets:
        if ds not in base_results or ds not in tuned_results:
            continue
        out[ds] = {"base": base_results[ds], "tuned": tuned_results[ds]}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"wrote {args.output} ({len(out)} datasets)")


if __name__ == "__main__":
    main()
