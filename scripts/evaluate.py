#!/usr/bin/env python3
"""Step 7: Evaluate a model on deliberation datasets.

Usage:
    python scripts/07_evaluate.py --model sentence-transformers/sentence-t5-large --split test
    python scripts/07_evaluate.py --model data/models/best/sentence_t5_large --split test
    python scripts/07_evaluate.py --model ... --datasets gsc_abortion_gen polis_brexit_consensus
"""

import argparse
import json
import logging
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ProjectPaths
from src.embedding.model import get_device, load_model
from src.evaluation.evaluator import EmbeddingEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--split", default="test", choices=["val", "test", "all"])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to evaluate (default: all 9)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    paths = ProjectPaths.auto()

    device = get_device()
    model = load_model(args.model, device=device)

    evaluator = EmbeddingEvaluator(eval_dir=paths.eval_dir)

    if args.datasets:
        results = {}
        for name in args.datasets:
            acc = evaluator.evaluate_dataset(model, name, split=args.split)
            if acc is not None:
                results[name] = acc
    else:
        results = evaluator.evaluate_all(model, split=args.split)

    # Print results
    print(f"\n{'Dataset':<40} {'Accuracy':>8}")
    print("-" * 50)
    for k, v in sorted(results.items()):
        print(f"{k:<40} {v:>7.1%}")
    print("-" * 50)
    print(f"{'Mean':>40} {sum(results.values()) / len(results):>7.1%}")

    # Save
    if args.output is None:
        paths.results_dir.mkdir(parents=True, exist_ok=True)
        model_slug = args.model.replace("/", "_").replace("-", "_")
        args.output = str(paths.results_dir / f"eval_{model_slug}_{args.split}.json")

    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "split": args.split,
            "metrics": results,
        }, f, indent=2)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
