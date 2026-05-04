#!/usr/bin/env python3
"""Decompose the cosine margin of each hard triplet into in-subspace
and out-of-subspace components, using the dataset's val-selected
ideal-point projection L (rank AND hyperparameters val-selected).

For unit-normalised embeddings,
    cos(a, p) - cos(a, n)
        = <psi(a), psi(p) - psi(n)>
        = <psi_S(a), psi_S(p) - psi_S(n)>           # Delta_s
        + <psi_perp(a), psi_perp(p) - psi_perp(n)>  # Delta_t

where psi_S = P_S psi is the orthogonal projection onto col(L) and
psi_perp = (I - P_S) psi is the residual. The framework predicts
that on natural data E[Delta_t] > 0 (regime i) while on hard triplets
E[Delta_t] < 0 (regime ii) — measure it.

Pipeline:
  1. Load natural-eval triplets for --dataset, 3-way participant split.
  2. Encode all unique texts (natural + hard) once with the base encoder.
  3. Sweep rank in {1,2,5,10,20,50,100} x BT hyperparameters; train each
     metric probe on the train split and val-score. Pick best (rank*, hp*).
  4. Use that best L to decompose every hard triplet in --dataset.

Output: data/results/triplet_decomposition_{dataset}.json
"""
import argparse
import json
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

SPLIT_SEED = 0
TRAIN_FRAC = 0.6
VAL_FRAC   = 0.2
BATCH_SIZE = 1024
RANKS = [1, 2, 5, 10, 20, 50, 100]
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
    """score(a, j) = -||L^T a - L^T j||^2."""
    def __init__(self, dim, rank):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)

    def forward(self, a, b):
        za, zb = a @ self.L, b @ self.L
        return -((za - zb) ** 2).sum(dim=-1)


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


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
        if np.linalg.norm(a) == 0: continue
        A.append(a); P.append(t2e[t["preferred"]]); N.append(t2e[t["dispreferred"]])
    return np.stack(A), np.stack(P), np.stack(N)


def train_probe(probe, train_data, hp, device):
    opt = torch.optim.Adam(probe.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    A, P, N = train_data
    ds = TensorDataset(torch.tensor(A, dtype=torch.float32, device=device),
                       torch.tensor(P, dtype=torch.float32, device=device),
                       torch.tensor(N, dtype=torch.float32, device=device))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(hp["epochs"]):
        probe.train()
        for a, p, n in loader:
            sp, sn = probe(a, p), probe(a, n)
            loss = -F.logsigmoid(sp - sn).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def acc(probe, data, device):
    probe.eval()
    with torch.no_grad():
        a, p, n = data
        at = torch.tensor(a, dtype=torch.float32, device=device)
        pt = torch.tensor(p, dtype=torch.float32, device=device)
        nt = torch.tensor(n, dtype=torch.float32, device=device)
        return (probe(at, pt) > probe(at, nt)).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gsc_abortion_gen")
    ap.add_argument("--encoder", default="sentence-transformers/sentence-t5-xl")
    ap.add_argument("--hard-triplets",
                    default="data/processed/triplets/hard_eval_triplets_1k.jsonl")
    ap.add_argument("--seed", type=int, default=42, help="probe init seed")
    ap.add_argument("--max-train-triplets", type=int, default=50000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    out_path = Path(args.output or f"data/results/triplet_decomposition_{args.dataset}.json")

    # --- 1. natural triplets, 3-way split ---
    nat_path = paths.eval_dir / f"{args.dataset}.jsonl"
    if not nat_path.exists():
        sys.exit(f"missing eval file: {nat_path}")
    natural = [json.loads(l) for l in open(nat_path)]
    splits = three_way_split([t["participant_id"] for t in natural])
    train_t = [t for t in natural if t["participant_id"] in splits["train"]]
    val_t   = [t for t in natural if t["participant_id"] in splits["val"]]
    if args.max_train_triplets and len(train_t) > args.max_train_triplets:
        train_t = random.Random(args.seed).sample(train_t, args.max_train_triplets)
    print(f"natural: train={len(train_t)} val={len(val_t)}", flush=True)

    # --- 2. hard triplets for this dataset ---
    hard_t = [json.loads(l) for l in open(args.hard_triplets)
               if json.loads(l)["dataset"] == args.dataset]
    if not hard_t:
        sys.exit(f"no hard triplets for {args.dataset}")
    print(f"hard: n={len(hard_t)}", flush=True)

    # --- 3. encode every unique text once ---
    texts = set()
    for t in train_t + val_t:
        texts.update(t["anchor_texts"]); texts.add(t["preferred"]); texts.add(t["dispreferred"])
    for t in hard_t:
        texts.update([t["anchor"], t["preference_match"], t["semantic_distractor"]])
    texts = sorted(texts)
    device = get_device()
    print(f"loading {args.encoder} on {device}", flush=True)
    model = load_model(args.encoder, device=device)
    print(f"encoding {len(texts)} unique texts", flush=True)
    embs_raw = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    t2e_raw  = dict(zip(texts, embs_raw))
    embs_unit = unit(embs_raw)
    t2e_unit  = dict(zip(texts, embs_unit))

    # Probe trains on RAW (un-normalised) embeddings — same convention as
    # retrain_metric_probe.py / supervised_probe_sweep.py.
    train_data = build_apn(train_t, t2e_raw)
    val_data   = build_apn(val_t,   t2e_raw)
    dim = embs_raw.shape[1]
    print(f"embedding dim={dim}; train/val tensors built", flush=True)

    # --- 4. val-select (rank, hp) ---
    grid_records = []
    best_val = -1.0
    best_record = None
    for rank in RANKS:
        for hp in GRID:
            torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
            probe = MetricProbe(dim, rank).to(device)
            train_probe(probe, train_data, hp, device)
            v = acc(probe, val_data, device)
            grid_records.append({"rank": rank, **hp, "val": v})
            print(f"  rank={rank:>3}  lr={hp['lr']:>6}  ep={hp['epochs']:>3}  wd={hp['wd']:>5}"
                  f"  val={v:.4f}", flush=True)
            if v > best_val:
                best_val = v
                best_record = {"rank": rank, **hp, "val": v}
                best_L = probe.L.detach().cpu().numpy().copy()
    print(f"\nbest: rank={best_record['rank']}  lr={best_record['lr']}  ep={best_record['epochs']}"
          f"  wd={best_record['wd']}  val={best_record['val']:.4f}", flush=True)

    # --- 5. orthonormal basis of col(L) for the projection ---
    Q, _ = np.linalg.qr(best_L)        # Q: (d, rank), Q^T Q = I
    proj_err = float(np.linalg.norm(Q.T @ Q - np.eye(best_record["rank"])))
    print(f"Q shape={Q.shape}  ||Q^T Q - I||={proj_err:.3e}", flush=True)

    # --- 6. decompose every hard triplet ---
    rows = []
    for t in hard_t:
        a, p, n = t["anchor"], t["preference_match"], t["semantic_distractor"]
        ea, ep, en = t2e_unit[a], t2e_unit[p], t2e_unit[n]
        qa, qp, qn = Q.T @ ea, Q.T @ ep, Q.T @ en
        cos_margin = float(np.dot(ea, ep - en))
        delta_s    = float(np.dot(qa, qp - qn))
        delta_t    = cos_margin - delta_s
        norm_diff  = float(np.dot(qn, qn) - np.dot(qp, qp))   # ||psi_S(n)||^2 - ||psi_S(p)||^2
        utility_margin = 2 * delta_s + norm_diff
        rows.append({
            "anchor": a, "match": p, "distractor": n,
            "cos_margin":     cos_margin,
            "Delta_s":        delta_s,
            "Delta_t":        delta_t,
            "psi_S_norm_diff": norm_diff,
            "utility_margin": utility_margin,
            "cos_correct":     int(cos_margin     > 0),
            "utility_correct": int(utility_margin > 0),
        })

    arr = lambda k: np.array([r[k] for r in rows])
    summary = {
        "n":              len(rows),
        "cos_margin":     {"mean": float(arr("cos_margin").mean()),
                           "std":  float(arr("cos_margin").std())},
        "Delta_s":        {"mean": float(arr("Delta_s").mean()),
                           "std":  float(arr("Delta_s").std())},
        "Delta_t":        {"mean": float(arr("Delta_t").mean()),
                           "std":  float(arr("Delta_t").std())},
        "utility_margin": {"mean": float(arr("utility_margin").mean()),
                           "std":  float(arr("utility_margin").std())},
        "frac_Delta_s_pos":     float((arr("Delta_s") > 0).mean()),
        "frac_Delta_t_neg":     float((arr("Delta_t") < 0).mean()),
        "frac_cos_correct":     float(arr("cos_correct").mean()),
        "frac_utility_correct": float(arr("utility_correct").mean()),
    }

    print()
    print(f"=== {args.dataset}  (n={summary['n']}, val-selected rank={best_record['rank']}) ===")
    print(f"  cos_margin     = {summary['cos_margin']['mean']:+.4f} +/- {summary['cos_margin']['std']:.4f}")
    print(f"  Delta_s        = {summary['Delta_s']['mean']:+.4f} +/- {summary['Delta_s']['std']:.4f}   "
          f"(frac > 0: {summary['frac_Delta_s_pos']:.2%})")
    print(f"  Delta_t        = {summary['Delta_t']['mean']:+.4f} +/- {summary['Delta_t']['std']:.4f}   "
          f"(frac < 0: {summary['frac_Delta_t_neg']:.2%})")
    print(f"  utility margin = {summary['utility_margin']['mean']:+.4f} +/- {summary['utility_margin']['std']:.4f}")
    print(f"  cos triplet acc:     {summary['frac_cos_correct']:.2%}")
    print(f"  utility triplet acc: {summary['frac_utility_correct']:.2%}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "dataset": args.dataset,
        "encoder": args.encoder,
        "best_hp": best_record,
        "grid":    grid_records,
        "summary": summary,
        "triplets": rows,
    }, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
