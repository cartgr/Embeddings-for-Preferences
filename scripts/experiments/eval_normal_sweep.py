#!/usr/bin/env python3
"""Post-hoc eval for the normal-triplet DPT sweep.

The sweep trains 100 LoRA adapters with `train.py --no-eval` so each
config saves its adapter even if the eval phase would OOM in-process.
This script then loops the saved adapters, loads each one fresh (no
training-state residue), runs the natural eval (val / test / all
splits across 11 datasets) and the 875-triplet hard eval, and writes
per-config results.json next to the adapter.

Skip-existing: any adapter that already has a results.json next to it
is skipped, so the script is preempt-resumable.
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
from src.evaluation.evaluator import EmbeddingEvaluator


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def hard_eval(model, hard_path: Path):
    triplets = [json.loads(l) for l in open(hard_path)]
    pos = lambda t: t.get("preference_match", t.get("paraphrase"))
    neg = lambda t: t.get("semantic_distractor", t.get("flip"))
    texts = sorted({t["anchor"] for t in triplets} | {pos(t) for t in triplets} | {neg(t) for t in triplets})
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=16)
    t2e = dict(zip(texts, embs))
    correct = sum(1 for t in triplets
                  if cosine_sim(t2e[t["anchor"]], t2e[pos(t)]) > cosine_sim(t2e[t["anchor"]], t2e[neg(t)]))
    return correct / len(triplets), len(triplets)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir",
                    default="data/models/best/sentence_transformers_sentence_t5_xl/normal_sweep")
    ap.add_argument("--hard-eval-triplets",
                    default="data/processed/triplets/hard_eval_triplets_1k.jsonl")
    ap.add_argument("--only-pattern", default=None,
                    help="Optional substring filter on config dir name.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    paths = ProjectPaths.auto()
    device = get_device()
    sweep = Path(args.sweep_dir)
    cfg_dirs = sorted(d for d in sweep.iterdir()
                      if d.is_dir() and (d / "adapter_model.safetensors").exists())
    if args.only_pattern:
        cfg_dirs = [d for d in cfg_dirs if args.only_pattern in d.name]
    logger.info(f"{len(cfg_dirs)} adapter dirs found under {sweep}")

    n_done = n_skip = n_fail = 0
    for cd in cfg_dirs:
        out = cd / "results.json"
        if out.exists():
            n_skip += 1; continue
        logger.info(f"\n==== {cd.name} ====")
        try:
            model = load_model(str(cd), device=device)
            evaluator = EmbeddingEvaluator(eval_dir=paths.eval_dir)
            val  = evaluator.evaluate_all(model, split="val")
            test = evaluator.evaluate_all(model, split="test")
            allm = evaluator.evaluate_all(model, split="all")
            hard_acc, hard_n = hard_eval(model, Path(args.hard_eval_triplets))
            json.dump({
                "model_path":         str(cd),
                "val_metrics":        val,
                "test_metrics":       test,
                "all_metrics":        allm,
                "hard_eval_accuracy": hard_acc,
                "hard_eval_n":        hard_n,
            }, open(out, "w"), indent=2)
            logger.info(f"  test_mean={np.mean(list(test.values()))*100:.2f}  hard={hard_acc*100:.2f}")
            n_done += 1
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            n_fail += 1
        finally:
            try:
                import torch; del model; torch.cuda.empty_cache()
            except Exception:
                pass

    logger.info(f"\nDone. evaluated={n_done}  skipped={n_skip}  failed={n_fail}")


if __name__ == "__main__":
    main()
