#!/usr/bin/env python3
"""Rank sweep for the per-topic PSD probe.

Measures test accuracy as a function of PSD rank r ∈ {1, 2, 5, 10, 20, 50, 100, 200}
on top of frozen base ST5-XL. Uses the same 3-way participant split (60/20/20)
and val-selected hyperparameters as the main supervised sweep. Goal: show
accuracy plateau around r=20, supporting the ideal-point model's low-rank
prediction.

Output: JSON with per-dataset per-rank curves and the val-selected test number.
"""
import argparse, json, logging, os, random, sys, warnings
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
RANK_GRID = [1, 2, 5, 10, 20, 50, 100, 200]
HP_GRID = [
    dict(lr=3e-4, epochs=30, wd=0.0),
    dict(lr=3e-4, epochs=30, wd=1e-3),
    dict(lr=1e-3, epochs=30, wd=0.0),
    dict(lr=1e-3, epochs=100, wd=0.0),
    dict(lr=1e-3, epochs=100, wd=1e-3),
]


def three_way_split(pids, seed=SPLIT_SEED):
    ids = sorted(set(pids)); rng = random.Random(seed); rng.shuffle(ids)
    n = len(ids); nt = int(n * TRAIN_FRAC); nv = int(n * VAL_FRAC)
    return {"train": set(ids[:nt]), "val": set(ids[nt:nt+nv]), "test": set(ids[nt+nv:])}


def load_splits(eval_dir, dataset):
    path = eval_dir / f"{dataset}.jsonl"
    if not path.exists(): return None
    triplets = [json.loads(l) for l in open(path)]
    split = three_way_split([t["participant_id"] for t in triplets])
    out = {}
    for name, pids in split.items():
        subset = [t for t in triplets if t["participant_id"] in pids]
        if not subset: return None
        out[name] = subset
    return out


def embed_triplets(triplets, model, max_triplets, seed=42):
    if not triplets: return None
    if len(triplets) > max_triplets:
        rng = random.Random(seed); triplets = rng.sample(triplets, max_triplets)
    texts = set()
    for t in triplets:
        texts.update(t["anchor_texts"])
        texts.add(t["preferred"]); texts.add(t["dispreferred"])
    texts = list(texts)
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    t2e = dict(zip(texts, embs))
    cache, A, P, N = {}, [], [], []
    for t in triplets:
        pid = t["participant_id"]
        if pid not in cache:
            es = [t2e[x] for x in t["anchor_texts"] if x in t2e]
            if not es: continue
            a = np.mean(es, axis=0)
            if np.linalg.norm(a) == 0: continue
            cache[pid] = a
        a = cache.get(pid)
        if a is None: continue
        p = t2e.get(t["preferred"]); n = t2e.get(t["dispreferred"])
        if p is None or n is None: continue
        A.append(a); P.append(p); N.append(n)
    if not A: return None
    return np.array(A), np.array(P), np.array(N)


def cosine_accuracy(a, p, d):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    pn = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-10)
    dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-10)
    cp = (an * pn).sum(1); cd = (an * dn).sum(1)
    return float((cp > cd).mean() + 0.5 * (cp == cd).mean())


def train_and_score(train, val, test, rank, hp, device):
    trainer = ProbeTrainer(probe_type=f"psd_r{rank}", embedding_dim=train[0].shape[1],
                            lr=hp["lr"], epochs=hp["epochs"], batch_size=512,
                            device=device, weight_decay=hp["wd"])
    trainer.train(*train)
    return {"val": trainer.triplet_accuracy(*val),
            "test": trainer.triplet_accuracy(*test),
            "train": trainer.triplet_accuracy(*train)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="sentence-transformers/sentence-t5-xl")
    ap.add_argument("--max-train", type=int, default=10000)
    ap.add_argument("--max-val", type=int, default=3000)
    ap.add_argument("--max-test", type=int, default=5000)
    ap.add_argument("--output", default="data/results/rank_sweep.json")
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    device = get_device()
    probe_device = "cuda" if str(device) != "cpu" else "cpu"
    model = load_model(args.base_model, device=device)

    all_results = {}
    for ds in EVAL_DATASETS:
        splits = load_splits(paths.eval_dir, ds)
        if splits is None: continue
        tr = embed_triplets(splits["train"], model, args.max_train)
        vd = embed_triplets(splits["val"], model, args.max_val)
        td = embed_triplets(splits["test"], model, args.max_test)
        if tr is None or vd is None or td is None: continue
        cos_t = cosine_accuracy(*td)
        logger.info(f"\n=== {ds} (cos_test={cos_t:.3f}) ===")

        per_rank = {}
        for r in RANK_GRID:
            grid = []
            for hp in HP_GRID:
                res = train_and_score(tr, vd, td, r, hp, probe_device)
                grid.append({**hp, **res})
            best = max(grid, key=lambda x: x["val"])
            logger.info(f"  r={r:>3}  val={best['val']:.3f} test={best['test']:.3f} "
                        f"(lr={best['lr']}, ep={best['epochs']}, wd={best['wd']})")
            per_rank[str(r)] = {"grid": grid, "best": best}

        all_results[ds] = {"cosine_test": cos_t, "per_rank": per_rank}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"base_model": args.base_model, "rank_grid": RANK_GRID,
                    "hp_grid": HP_GRID, "results": all_results}, f, indent=2)
    logger.info(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
