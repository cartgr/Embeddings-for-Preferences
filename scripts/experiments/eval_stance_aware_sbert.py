#!/usr/bin/env python3
"""Evaluate StanceAware-SBERT on the 10 natural-data datasets + 875 hard triplets.

StanceAware-SBERT is a LoRA adapter on top of sentence-transformers/all-mpnet-base-v2
(https://huggingface.co/vahidthegreat/StanceAware-SBERT), so we have to apply the
adapter explicitly via PeftModel.from_pretrained rather than the usual
SentenceTransformer(model_id) loader.

Appends results to data/results/hard_triplet_cosine_1k.json under the key
'vahidthegreat/StanceAware-SBERT' so it lives alongside the other base-encoder
numbers reported in the paper.
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
from src.embedding.model import get_device
from src.evaluation.evaluator import EVAL_DATASETS, cosine_similarity
from scripts.experiments.hard_triplet_cosine import hard_accuracy, normal_accuracy, _pos, _neg  # reuse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_stance_aware_sbert(device: str):
    """Load StanceAware-SBERT by applying its LoRA adapter to all-mpnet-base-v2."""
    from sentence_transformers import SentenceTransformer
    from peft import PeftModel
    base_id = "sentence-transformers/all-mpnet-base-v2"
    adapter_id = "vahidthegreat/StanceAware-SBERT"
    logger.info(f"Loading base {base_id}")
    model = SentenceTransformer(base_id, device=device)
    logger.info(f"Applying LoRA adapter {adapter_id}")
    model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, adapter_id)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-triplets",
                    default="data/processed/triplets/hard_eval_triplets_1k.jsonl")
    ap.add_argument("--output", default="data/results/hard_triplet_cosine_1k.json")
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    device = get_device()

    hard = [json.loads(l) for l in open(args.hard_triplets)]
    logger.info(f"Loaded {len(hard)} hard triplets")

    existing = {}
    out_path = Path(args.output)
    if out_path.exists():
        existing = json.load(open(out_path))

    key = "vahidthegreat/StanceAware-SBERT"
    model = load_stance_aware_sbert(device)
    logger.info("Evaluating on hard triplets...")
    hard_acc = hard_accuracy(model, hard)
    logger.info(f"  hard acc: {hard_acc:.3f}")
    logger.info("Evaluating on 10 natural datasets...")
    normal_mean, per_ds = normal_accuracy(model, paths.eval_dir)
    logger.info(f"  normal mean across {len(per_ds)} datasets: {normal_mean:.3f}")

    existing[key] = {
        "hard_acc": hard_acc,
        "hard_n": len(hard),
        "normal_mean": normal_mean,
        "normal_per_dataset": per_ds,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"Saved {out_path}")

    print("\nFinal:")
    print(f"{'model':<55} {'normal':>8} {'hard':>7} {'Δ':>7}")
    for enc, r in existing.items():
        d = r["normal_mean"] - r["hard_acc"]
        print(f"{enc:<55} {r['normal_mean']:>8.3f} {r['hard_acc']:>7.3f} {d:>+7.3f}")


if __name__ == "__main__":
    main()
