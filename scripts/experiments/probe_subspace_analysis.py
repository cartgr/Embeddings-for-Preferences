#!/usr/bin/env python3
"""Compare the rank-20 metric probe's learned subspace when fit on the
base ST5-XL encoder vs on the DPT-tuned encoder.

For each dataset (default: all 11):
  1. Train the metric probe on BASE psi at the (ds, seed=42) val-selected
     HP from data/results/scorer_structural_ablation.json -> L_base
  2. Train on TUNED psi' at the corresponding HP from the
     ..._ablation_tuned.json -> L_tuned
  3. Orthonormalize: Q_base, Q_tuned (d x 20)
  4. Principal angles between col(L_base) and col(L_tuned):
       SVD(Q_base.T @ Q_tuned) -> 20 cosines in [0, 1]
       (1 = perfectly aligned direction; 0 = orthogonal)
  5. Per hard triplet of this dataset, decompose the probe-score margin
       margin = 2 * <L^T a, L^T (p - n)> + (||L^T n||^2 - ||L^T p||^2)
              = ip_term                  +  norm_term
     and report each term's mean / sign frac under (base, L_base) vs
     (tuned, L_tuned).

Output: data/results/probe_subspace_analysis.json
        per-dataset principal-angle cosines + per-triplet decomposition
        + aggregate summaries.
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
from src.evaluation.evaluator import EVAL_DATASETS

SPLIT_SEED = 0
TRAIN_FRAC = 0.6
VAL_FRAC   = 0.2
RANK = 20
BATCH_SIZE = 1024
TUNED_PATH = ("data/models/best/sentence_transformers_sentence_t5_xl/"
              "targeted_sweep/lr1p25eem4_n750_r16_a48")


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
    n = len(ids); n_tr = int(n * TRAIN_FRAC); n_va = int(n * VAL_FRAC)
    return {"train": set(ids[:n_tr]), "val": set(ids[n_tr:n_tr+n_va]), "test": set(ids[n_tr+n_va:])}


def build_apn(triplets, t2e):
    A, P, N = [], [], []
    for t in triplets:
        a = np.mean([t2e[x] for x in t["anchor_texts"] if x in t2e], axis=0)
        if np.linalg.norm(a) == 0: continue
        A.append(a); P.append(t2e[t["preferred"]]); N.append(t2e[t["dispreferred"]])
    return np.stack(A), np.stack(P), np.stack(N)


def train_at_hp(dim, train_data, hp, device, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    probe = MetricProbe(dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    A, P, N = train_data
    ds = TensorDataset(torch.tensor(A, dtype=torch.float32, device=device),
                       torch.tensor(P, dtype=torch.float32, device=device),
                       torch.tensor(N, dtype=torch.float32, device=device))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(int(hp["epochs"])):
        probe.train()
        for a, p, n in loader:
            sp, sn = probe(a, p), probe(a, n)
            loss = -F.logsigmoid(sp - sn).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return probe.L.detach().cpu().numpy()


def per_triplet_decomp(L, t2e, hard_triplets):
    """Score margin = 2 * <z_a, z_p - z_n> + ||z_n||^2 - ||z_p||^2.
    Returns lists of (margin, ip_term, norm_term) per hard triplet."""
    rows = []
    for t in hard_triplets:
        a, p, n = t["anchor"], t["preference_match"], t["semantic_distractor"]
        za = L.T @ t2e[a]
        zp = L.T @ t2e[p]
        zn = L.T @ t2e[n]
        ip   = 2 * float(np.dot(za, zp - zn))
        norm = float(np.dot(zn, zn) - np.dot(zp, zp))
        rows.append({"margin": ip + norm, "ip_term": ip, "norm_term": norm,
                     "correct": int(ip + norm > 0)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--hard-triplets",
                    default="data/processed/triplets/hard_eval_triplets_1k.jsonl")
    ap.add_argument("--ablation-base",  default="data/results/scorer_structural_ablation.json")
    ap.add_argument("--ablation-tuned", default="data/results/scorer_structural_ablation_tuned.json")
    ap.add_argument("--output",         default="data/results/probe_subspace_analysis.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    datasets = args.datasets or [d for d in EVAL_DATASETS]
    base_hp_d  = json.load(open(args.ablation_base))["metric"]
    tuned_hp_d = json.load(open(args.ablation_tuned))["metric"]
    hard_all = [json.loads(l) for l in open(args.hard_triplets)]

    device = get_device()
    print(f"Loading base ST5-XL on {device}", flush=True)
    base = load_model("sentence-transformers/sentence-t5-xl", device=device)
    print(f"Loading tuned LoRA from {TUNED_PATH}", flush=True)
    tuned = load_model(TUNED_PATH, device=device)

    out = {}
    for ds in datasets:
        print(f"\n=== {ds} ===", flush=True)
        # Sources
        eval_path = paths.eval_dir / f"{ds}.jsonl"
        if not eval_path.exists():
            print(f"  skip (no eval file)"); continue
        triplets = [json.loads(l) for l in open(eval_path)]
        splits = three_way_split([t["participant_id"] for t in triplets])
        train_t = [t for t in triplets if t["participant_id"] in splits["train"]]
        if not train_t: continue
        if str(args.seed) not in base_hp_d.get(ds, {}) or str(args.seed) not in tuned_hp_d.get(ds, {}):
            print(f"  skip: missing seed {args.seed} in ablation jsons"); continue
        hp_base  = base_hp_d[ds][str(args.seed)]["best_hp"]
        hp_tuned = tuned_hp_d[ds][str(args.seed)]["best_hp"]
        hard = [t for t in hard_all if t.get("dataset") == ds]
        print(f"  train_triplets={len(train_t)}  hard={len(hard)}")
        print(f"  hp_base={hp_base}  hp_tuned={hp_tuned}")

        # Encode every unique text under each encoder once
        natural_texts = set()
        for t in train_t:
            natural_texts.update(t["anchor_texts"])
            natural_texts.update([t["preferred"], t["dispreferred"]])
        hard_texts = set()
        for t in hard:
            hard_texts.update([t["anchor"], t["preference_match"], t["semantic_distractor"]])
        all_texts = sorted(natural_texts | hard_texts)
        eb = base.encode(all_texts,  convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        et = tuned.encode(all_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        t2e_base  = dict(zip(all_texts, eb))
        t2e_tuned = dict(zip(all_texts, et))

        # Train probes at val-selected HP on each encoder
        train_data_base  = build_apn(train_t, t2e_base)
        train_data_tuned = build_apn(train_t, t2e_tuned)
        dim = eb.shape[1]
        L_base  = train_at_hp(dim, train_data_base,  hp_base,  device, seed=args.seed)
        L_tuned = train_at_hp(dim, train_data_tuned, hp_tuned, device, seed=args.seed)
        print(f"  L_base shape={L_base.shape}  L_tuned shape={L_tuned.shape}")

        # Principal angles between subspaces
        Qb, _ = np.linalg.qr(L_base)
        Qt, _ = np.linalg.qr(L_tuned)
        s = np.linalg.svd(Qb.T @ Qt, compute_uv=False)
        s = np.clip(s, 0.0, 1.0)
        print(f"  principal-angle cosines: max={s.max():.3f}  mean={s.mean():.3f}  "
              f"min={s.min():.3f}  median={np.median(s):.3f}")
        print(f"    (1.0 = aligned dim, 0.0 = orthogonal dim; rank-20)")

        # Per-triplet score-margin decomposition on hard triplets
        rows_base  = per_triplet_decomp(L_base,  t2e_base,  hard) if hard else []
        rows_tuned = per_triplet_decomp(L_tuned, t2e_tuned, hard) if hard else []

        def summary(rows):
            if not rows: return None
            arr = lambda k: np.array([r[k] for r in rows])
            return {
                "n":              len(rows),
                "margin":         {"mean": float(arr("margin").mean()),
                                   "std":  float(arr("margin").std())},
                "ip_term":        {"mean": float(arr("ip_term").mean()),
                                   "std":  float(arr("ip_term").std())},
                "norm_term":      {"mean": float(arr("norm_term").mean()),
                                   "std":  float(arr("norm_term").std())},
                "frac_ip_pos":    float((arr("ip_term")   > 0).mean()),
                "frac_norm_pos":  float((arr("norm_term") > 0).mean()),
                "frac_correct":   float((arr("margin")    > 0).mean()),
            }
        sb = summary(rows_base);  st = summary(rows_tuned)
        if sb and st:
            print(f"  hard-triplet score margin (probe on encoder):")
            print(f"    base : margin={sb['margin']['mean']:+.4f}  ip={sb['ip_term']['mean']:+.4f}  "
                  f"norm={sb['norm_term']['mean']:+.4f}  acc={sb['frac_correct']:.2%}")
            print(f"    tuned: margin={st['margin']['mean']:+.4f}  ip={st['ip_term']['mean']:+.4f}  "
                  f"norm={st['norm_term']['mean']:+.4f}  acc={st['frac_correct']:.2%}")

        out[ds] = {
            "hp_base":  hp_base,
            "hp_tuned": hp_tuned,
            "principal_cosines": [float(x) for x in s],
            "summary_base":  sb,
            "summary_tuned": st,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
