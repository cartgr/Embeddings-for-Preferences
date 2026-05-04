#!/usr/bin/env python3
"""Error decomposition — where does fine-tuning help, where does it hurt?

For every eval triplet, record whether the base encoder and the fine-tuned
encoder each rank the preferred statement above the dispreferred one. Compute
the 2×2 contingency table (both correct / both wrong / tuned fixes / tuned
breaks) overall and per dataset.

Appendix result: tuned_fixes − tuned_breaks ≈ net accuracy gain; both_correct
+ both_wrong is the "fine-tuning doesn't change the answer" fraction.

Usage:
    python scripts/experiments/error_decomposition.py \\
        --tuned-model data/models/best/sentence_transformers_sentence_t5_xl/\\
targeted_sweep/lr1p25eem4_n750_r16_a48 \\
        --output data/results/error_decomposition.json
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


def per_triplet_correctness(model, triplets):
    """Return bool array: True iff cos(a, p) > cos(a, d) under this model."""
    texts = set()
    for t in triplets:
        texts.update(t["anchor_texts"])
        texts.add(t["preferred"]); texts.add(t["dispreferred"])
    texts = list(texts)
    embs = model.encode(texts, convert_to_numpy=True,
                         show_progress_bar=False, batch_size=64)
    t2e = dict(zip(texts, embs))

    correct = []
    for t in triplets:
        anchor_embs = [t2e[x] for x in t["anchor_texts"] if x in t2e]
        if not anchor_embs:
            correct.append(None); continue
        a = np.mean(anchor_embs, axis=0)
        if np.linalg.norm(a) == 0:
            correct.append(None); continue
        sp = cosine_similarity(a, t2e[t["preferred"]])
        sd = cosine_similarity(a, t2e[t["dispreferred"]])
        correct.append(bool(sp > sd))
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model",  default="sentence-transformers/sentence-t5-xl")
    ap.add_argument("--tuned-model", required=True)
    ap.add_argument("--output", default="data/results/error_decomposition.json")
    args = ap.parse_args()

    paths  = ProjectPaths.auto()
    device = get_device()

    # Score both encoders once per dataset
    per_dataset = {}
    agg = {"both_correct": 0, "both_wrong": 0, "tuned_fixes": 0, "tuned_breaks": 0}

    logger.info(f"Loading base: {args.base_model}")
    base_model = load_model(args.base_model, device=device)

    base_correctness = {}
    for ds in EVAL_DATASETS:
        path = paths.eval_dir / f"{ds}.jsonl"
        if not path.exists(): continue
        logger.info(f"  scoring {ds} (base)")
        triplets = [json.loads(l) for l in open(path)]
        base_correctness[ds] = (triplets, per_triplet_correctness(base_model, triplets))
    del base_model
    import torch; torch.cuda.empty_cache()

    logger.info(f"Loading tuned: {args.tuned_model}")
    tuned_model = load_model(args.tuned_model, device=device)

    for ds, (triplets, base_corr) in base_correctness.items():
        logger.info(f"  scoring {ds} (tuned)")
        tuned_corr = per_triplet_correctness(tuned_model, triplets)

        both_correct = both_wrong = tuned_fixes = tuned_breaks = 0
        n = 0
        for b, t in zip(base_corr, tuned_corr):
            if b is None or t is None: continue
            n += 1
            if   b and t:         both_correct += 1
            elif (not b) and (not t): both_wrong += 1
            elif (not b) and t:   tuned_fixes  += 1
            elif b and (not t):   tuned_breaks += 1
        per_dataset[ds] = {
            "n": n,
            "both_correct": both_correct,
            "both_wrong":   both_wrong,
            "tuned_fixes":  tuned_fixes,
            "tuned_breaks": tuned_breaks,
            "base_accuracy":  (both_correct + tuned_breaks) / n if n else 0.0,
            "tuned_accuracy": (both_correct + tuned_fixes)  / n if n else 0.0,
        }
        for k in agg: agg[k] += per_dataset[ds][k]

    n_total = sum(d["n"] for d in per_dataset.values())
    out = {
        "base_model": args.base_model,
        "tuned_model": args.tuned_model,
        "n_triplets": n_total,
        "overall": {
            **agg,
            "base_accuracy":  (agg["both_correct"] + agg["tuned_breaks"]) / n_total,
            "tuned_accuracy": (agg["both_correct"] + agg["tuned_fixes"])  / n_total,
            "net_improvement_pp": 100 * (agg["tuned_fixes"] - agg["tuned_breaks"]) / n_total,
        },
        "per_dataset": per_dataset,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    o = out["overall"]
    logger.info(f"\nOverall ({n_total} triplets):")
    logger.info(f"  both correct:  {o['both_correct']} ({100*o['both_correct']/n_total:.1f}%)")
    logger.info(f"  both wrong:    {o['both_wrong']} ({100*o['both_wrong']/n_total:.1f}%)")
    logger.info(f"  tuned fixes:   {o['tuned_fixes']} (+{100*o['tuned_fixes']/n_total:.1f}%)")
    logger.info(f"  tuned breaks:  {o['tuned_breaks']} (-{100*o['tuned_breaks']/n_total:.1f}%)")
    logger.info(f"  net:           {o['net_improvement_pp']:+.2f} pp")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
