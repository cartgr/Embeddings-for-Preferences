#!/usr/bin/env python3
"""Smoke test: load base ST5-XL and evaluate 10 triplets from gsc_abortion_gen.

Run from the repo root:
    python tests/smoke_test.py

Expected output: cosine accuracy in [0, 1] (typically ~0.8 for this subset).

Fails loudly if:
  - sentence-transformers / torch aren't installed
  - data/processed/eval/gsc_abortion_gen.jsonl is missing (LFS not pulled?)
  - base ST5-XL can't be downloaded
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main():
    eval_path = REPO_ROOT / "data/processed/eval/gsc_abortion_gen.jsonl"
    if not eval_path.exists():
        sys.exit(
            f"ERROR: {eval_path} not found. If you cloned with Git LFS support,\n"
            f"run `git lfs pull` to download the paper artifacts."
        )
    # Check for LFS pointer (small file ≈ 130 bytes, full content starts with 'version https://git-lfs')
    if eval_path.stat().st_size < 200:
        with open(eval_path) as f:
            head = f.read(100)
        if "git-lfs" in head:
            sys.exit(
                f"ERROR: {eval_path} is still an LFS pointer. Run `git lfs pull`."
            )

    print("Loading sentence-T5-XL from HuggingFace (first run downloads ~5 GB)...")
    from src.embedding.model import load_model, get_device
    from src.evaluation.evaluator import cosine_similarity

    model = load_model("sentence-transformers/sentence-t5-xl", device=get_device())

    # Load 10 triplets
    triplets = []
    with open(eval_path) as f:
        for i, line in enumerate(f):
            triplets.append(json.loads(line))
            if i >= 9: break
    print(f"Loaded {len(triplets)} triplets.")

    # Embed all texts
    texts = set()
    for t in triplets:
        texts.update(t["anchor_texts"])
        texts.add(t["preferred"]); texts.add(t["dispreferred"])
    texts = list(texts)
    print(f"Embedding {len(texts)} unique texts...")
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    t2e = dict(zip(texts, embs))

    # Score
    correct = 0
    for t in triplets:
        anchor = np.mean([t2e[x] for x in t["anchor_texts"] if x in t2e], axis=0)
        sim_p = cosine_similarity(anchor, t2e[t["preferred"]])
        sim_d = cosine_similarity(anchor, t2e[t["dispreferred"]])
        if sim_p > sim_d:
            correct += 1
    acc = correct / len(triplets)
    print(f"\nCosine accuracy on first 10 triplets: {acc:.2f}")
    if acc < 0.3:
        sys.exit("FAIL: accuracy is suspiciously low.")
    print("PASS")


if __name__ == "__main__":
    main()
