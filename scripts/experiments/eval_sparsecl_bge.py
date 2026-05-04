#!/usr/bin/env python3
"""Evaluate SparseCL/BGE-SparseCL-arguana on 10 natural datasets + 875 hard triplets.

This model ships as a raw HF transformer (no sentence-transformers config,
no modules.json), so we wrap AutoModel in a minimal encoder that applies
BGE-style [CLS] pooling + L2 normalization. The resulting class is
duck-typed to be compatible with the hard_triplet_cosine.py helpers, which
only call .encode(texts, convert_to_numpy=True, show_progress_bar=..., batch_size=...).

Appends to data/results/hard_triplet_cosine_1k.json.
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
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import ProjectPaths
from src.embedding.model import get_device
from scripts.experiments.hard_triplet_cosine import hard_accuracy, normal_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class BGECLSEncoder:
    """BGE-style encoder: [CLS] pooling + L2 normalize."""

    def __init__(self, model_id: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.device = device
        self.model.to(device)

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**enc)
                # BGE models use CLS pooling.
                cls = out.last_hidden_state[:, 0, :]
                cls = F.normalize(cls, p=2, dim=-1)
                all_embs.append(cls.cpu().numpy())
        embs = np.concatenate(all_embs, axis=0)
        return embs if convert_to_numpy else torch.from_numpy(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="SparseCL/BGE-SparseCL-arguana")
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

    logger.info(f"Loading {args.model_id} on {device}")
    model = BGECLSEncoder(args.model_id, device=device)

    hard_acc = hard_accuracy(model, hard)
    logger.info(f"  hard acc: {hard_acc:.3f}")
    normal_mean, per_ds = normal_accuracy(model, paths.eval_dir)
    logger.info(f"  normal mean across {len(per_ds)} datasets: {normal_mean:.3f}")

    existing[args.model_id] = {
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
