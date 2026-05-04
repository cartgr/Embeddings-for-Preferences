#!/usr/bin/env python3
"""Generate data/results/table1_example.json — cosine similarities for a
single hard-eval triplet under three geometries (base, tuned, projected).

The triplet is line 105 (0-indexed) of hard_eval_triplets_1k.jsonl from the
gsc_abortion_gen dataset.  The three geometries are:

  base       — raw sentence-T5-XL embeddings
  tuned      — LoRA-adapted sentence-T5-XL (seed 42)
  projected  — tuned embeddings projected through the per-topic metric probe L
               (rank-20), then cosine similarity in the 20-D space

Usage:
  python scripts/table1_example.py            # writes data/results/table1_example.json
  python scripts/table1_example.py --stdout   # prints to stdout
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedding.model import get_device, load_model

ROOT = Path(__file__).resolve().parents[1]

TRIPLET_FILE = ROOT / "data/processed/triplets/hard_eval_triplets_1k.jsonl"
TRIPLET_INDEX = 105  # 0-indexed line number

TUNED_MODEL_DIR = ROOT / "data/models/best/sentence_transformers_sentence_t5_xl/seed42"
BASE_MODEL_NAME = "sentence-transformers/sentence-t5-xl"

PROBE_WEIGHTS = ROOT / "data/results/psd_probe_weights/L_perds_gsc_abortion_gen.npy"
RANK = 20

OUT_PATH = ROOT / "data/results/table1_example.json"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_triplet():
    with open(TRIPLET_FILE) as f:
        for i, line in enumerate(f):
            if i == TRIPLET_INDEX:
                return json.loads(line)
    raise IndexError(f"triplet file has fewer than {TRIPLET_INDEX + 1} lines")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    device = get_device()
    triplet = load_triplet()
    texts = [triplet["anchor"], triplet["preference_match"], triplet["semantic_distractor"]]

    # --- base geometry ---
    print("Loading base model...", file=sys.stderr)
    base_model = load_model(BASE_MODEL_NAME, device=device)
    base_embs = base_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    a_base, m_base, d_base = base_embs
    sim_m_base = cosine_sim(a_base, m_base)
    sim_d_base = cosine_sim(a_base, d_base)

    # --- tuned geometry ---
    print("Loading tuned model...", file=sys.stderr)
    tuned_model = load_model(TUNED_MODEL_DIR, device=device)
    tuned_embs = tuned_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    a_tuned, m_tuned, d_tuned = tuned_embs
    sim_m_tuned = cosine_sim(a_tuned, m_tuned)
    sim_d_tuned = cosine_sim(a_tuned, d_tuned)

    # --- projected geometry (cosine in L-projected space) ---
    L = np.load(PROBE_WEIGHTS)  # (dim, rank)
    a_proj = torch.tensor(a_tuned @ L, dtype=torch.float32).unsqueeze(0)
    m_proj = torch.tensor(m_tuned @ L, dtype=torch.float32).unsqueeze(0)
    d_proj = torch.tensor(d_tuned @ L, dtype=torch.float32).unsqueeze(0)
    sim_m_proj = F.cosine_similarity(a_proj, m_proj).item()
    sim_d_proj = F.cosine_similarity(a_proj, d_proj).item()

    result = {
        "source": f"{TRIPLET_FILE.relative_to(ROOT)}#{TRIPLET_INDEX}",
        "anchor": triplet["anchor"],
        "match": triplet["preference_match"],
        "distractor": triplet["semantic_distractor"],
        "dataset": triplet["dataset"],
        "rank": RANK,
        "results": [
            {"geometry": "base",
             "sim_match": sim_m_base, "sim_distractor": sim_d_base,
             "margin": sim_m_base - sim_d_base},
            {"geometry": "tuned",
             "sim_match": sim_m_tuned, "sim_distractor": sim_d_tuned,
             "margin": sim_m_tuned - sim_d_tuned},
            {"geometry": "projected",
             "sim_match": sim_m_proj, "sim_distractor": sim_d_proj,
             "margin": sim_m_proj - sim_d_proj},
        ],
    }

    out = json.dumps(result, indent=2) + "\n"
    if args.stdout:
        print(out)
    else:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(out)
        print(f"wrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
