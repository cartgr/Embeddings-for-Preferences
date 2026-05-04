#!/usr/bin/env python3
"""Step 10: Run baseline methods.

Evaluates:
- OpenAI text-embedding-3-large
- Voyage voyage-3
- GPT-4o discriminative query (text-only and few-shot)
- GPT-4o-mini discriminative query (text-only and few-shot)

Usage:
    python scripts/10_baselines.py --method openai
    python scripts/10_baselines.py --method voyage
    python scripts/10_baselines.py --method gpt-4o
    python scripts/10_baselines.py --method all
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
from src.evaluation.evaluator import EmbeddingEvaluator
from src.evaluation.baselines import OpenAIEmbedder, VoyageEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_embedding_baseline(model, name: str, paths: ProjectPaths, split: str = "test"):
    """Run an embedding baseline on all datasets."""
    evaluator = EmbeddingEvaluator(eval_dir=paths.eval_dir)
    results = evaluator.evaluate_all(model, split=split)

    print(f"\n{name} ({split}):")
    print(f"{'Dataset':<40} {'Accuracy':>8}")
    print("-" * 50)
    for k, v in sorted(results.items()):
        print(f"{k:<40} {v:>7.1%}")
    if results:
        print(f"{'Mean':>40} {sum(results.values()) / len(results):>7.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="all", choices=["openai", "voyage", "gpt-4o", "gpt-4o-mini", "all"])
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    paths = ProjectPaths.auto()
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.method in ("openai", "all"):
        logger.info("Running OpenAI text-embedding-3-large baseline...")
        model = OpenAIEmbedder(model="text-embedding-3-large")
        all_results["openai_large"] = run_embedding_baseline(model, "OpenAI text-embedding-3-large", paths, args.split)

    if args.method in ("voyage", "all"):
        logger.info("Running Voyage voyage-3 baseline...")
        model = VoyageEmbedder(model="voyage-3")
        all_results["voyage_3"] = run_embedding_baseline(model, "Voyage voyage-3", paths, args.split)

    if args.method in ("gpt-4o", "gpt-4o-mini", "all"):
        logger.info(f"LLM discriminative query baselines require scripts/evaluate_llm_disc_query.py")
        logger.info(f"Run: python scripts/evaluate_llm_disc_query.py --model {args.method}")

    output_path = paths.results_dir / f"baselines_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
