#!/usr/bin/env python3
"""Evaluate cosine accuracy on hard vs regular eval triplets.

Used in:
  §3 — base encoders fail on hard triplets (aggregate).
  §5 — tuned models restore hard-triplet accuracy (delta over base).

Inputs:
  data/processed/triplets/hard_eval_triplets.jsonl
    schema: {"dataset", "anchor", "paraphrase", "flip"}
    where paraphrase = preference match, flip = semantic distractor

  data/processed/eval/*.jsonl (regular eval triplets)

Output:
  data/results/hard_triplet_cosine.json
    {model_id: {"normal_mean": float, "hard_acc": float, "hard_n": int}}

Usage:
    python scripts/experiments/hard_triplet_cosine.py \\
        --encoders sentence-transformers/sentence-t5-xl \\
                   intfloat/e5-large-v2 \\
                   BAAI/bge-large-en-v1.5 \\
                   sentence-transformers/all-mpnet-base-v2
"""
import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import ProjectPaths
from src.embedding.model import get_device, load_model
from src.evaluation.evaluator import EVAL_DATASETS, cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _pos(t): return t.get("preference_match", t.get("paraphrase"))
def _neg(t): return t.get("semantic_distractor", t.get("flip"))


def hard_accuracy(model, hard_triplets):
    texts = set()
    for t in hard_triplets:
        texts.update([t["anchor"], _pos(t), _neg(t)])
    texts = list(texts)
    embs = model.encode(texts, convert_to_numpy=True,
                         show_progress_bar=False, batch_size=64)
    t2e = dict(zip(texts, embs))
    correct = 0
    for t in hard_triplets:
        a = t2e[t["anchor"]]; p = t2e[_pos(t)]; d = t2e[_neg(t)]
        sp = cosine_similarity(a, p); sd = cosine_similarity(a, d)
        if sp > sd: correct += 1
        elif sp == sd: correct += 0.5
    return correct / len(hard_triplets)


def normal_accuracy(model, eval_dir: Path, max_per_ds: int = 2000, seed: int = 0):
    """Mean cosine accuracy across 11 eval datasets."""
    import random
    rng = random.Random(seed)
    per_ds = {}
    for ds in EVAL_DATASETS:
        path = eval_dir / f"{ds}.jsonl"
        if not path.exists(): continue
        triplets = [json.loads(l) for l in open(path)]
        if len(triplets) > max_per_ds:
            triplets = rng.sample(triplets, max_per_ds)

        texts = set()
        for t in triplets:
            texts.update(t["anchor_texts"])
            texts.add(t["preferred"]); texts.add(t["dispreferred"])
        texts = list(texts)
        embs = model.encode(texts, convert_to_numpy=True,
                             show_progress_bar=False, batch_size=64)
        t2e = dict(zip(texts, embs))

        correct = total = 0
        for t in triplets:
            anchor = np.mean([t2e[x] for x in t["anchor_texts"] if x in t2e], axis=0)
            if np.linalg.norm(anchor) == 0: continue
            sp = cosine_similarity(anchor, t2e[t["preferred"]])
            sd = cosine_similarity(anchor, t2e[t["dispreferred"]])
            if sp > sd: correct += 1
            elif sp == sd: correct += 0.5
            total += 1
        if total:
            per_ds[ds] = correct / total
    mean = sum(per_ds.values()) / len(per_ds) if per_ds else 0.0
    return mean, per_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True,
                    help="HuggingFace IDs or local paths.")
    ap.add_argument("--hard-triplets",
                    default="data/processed/triplets/hard_eval_triplets.jsonl")
    ap.add_argument("--output", default="data/results/hard_triplet_cosine.json")
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    device = get_device()

    hard = [json.loads(l) for l in open(args.hard_triplets)]
    logger.info(f"Loaded {len(hard)} hard triplets.")

    existing = {}
    out_path = Path(args.output)
    if out_path.exists():
        existing = json.load(open(out_path))

    results = {**existing}
    for enc in args.encoders:
        if enc in results:
            logger.info(f"  skip (cached): {enc}")
            continue
        logger.info(f"Loading {enc}")
        model = load_model(enc, device=device)

        hard_acc = hard_accuracy(model, hard)
        logger.info(f"  hard acc: {hard_acc:.3f}")

        normal_mean, per_ds = normal_accuracy(model, paths.eval_dir)
        logger.info(f"  normal mean across {len(per_ds)} datasets: {normal_mean:.3f}")

        results[enc] = {
            "hard_acc": hard_acc,
            "hard_n": len(hard),
            "normal_mean": normal_mean,
            "normal_per_dataset": per_ds,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        del model
        import torch; torch.cuda.empty_cache()

    print("\nFinal:")
    print(f"{'model':<55} {'normal':>8} {'hard':>7} {'Δ':>7}")
    for enc, r in results.items():
        d = r["normal_mean"] - r["hard_acc"]
        print(f"{enc:<55} {r['normal_mean']:>8.3f} {r['hard_acc']:>7.3f} {d:>+7.3f}")


if __name__ == "__main__":
    main()
