#!/usr/bin/env python3
"""Step 3: Generate 5 diverse opinions per issue.

Generates opinions spanning position 1 (strongly pro) to position 5
(strongly anti). Generates once for the union of all filtered issues,
so all condition/selection combos can draw from the same pool.

Supports checkpointing: safe to interrupt and resume.

Usage:
    python scripts/03_generate_opinions.py --model claude-sonnet-4-20250514
    python scripts/03_generate_opinions.py --model gpt-4o
    python scripts/03_generate_opinions.py --model gpt-4o-mini
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ProjectPaths
from src.generation.opinion_generator import OpinionGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def model_slug(model: str) -> str:
    """Convert model name to filesystem-safe slug."""
    return model.replace("/", "_").replace("-", "_").replace(".", "_")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Model for generation")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Parallel API workers")
    args = parser.parse_args()

    paths = ProjectPaths.auto()

    # Read the union of all filtered issues
    union_path = paths.issues_dir / "union.jsonl"
    if not union_path.exists():
        logger.error(f"Union file not found: {union_path}")
        logger.error("Run 02_filter_issues.py first, then create union with the pipeline.")
        sys.exit(1)

    with open(union_path) as f:
        issues = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(issues)} union issues")

    # Output: data/processed/opinions/{model_slug}/opinions.jsonl
    slug = model_slug(args.model)
    output_dir = paths.opinions_dir / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "opinions.jsonl"

    # Create client
    if any(x in args.model for x in ["claude", "sonnet", "opus", "haiku"]):
        import anthropic
        client = anthropic.Anthropic()
    else:
        from openai import OpenAI
        client = OpenAI()

    generator = OpinionGenerator(client=client, model=args.model)
    results = generator.generate_batch(issues, checkpoint_path=str(output_path))

    logger.info(f"Generated opinions for {len(results)}/{len(issues)} issues")
    logger.info(f"Total opinions: {len(results) * 5}")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
