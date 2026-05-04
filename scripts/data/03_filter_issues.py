#!/usr/bin/env python3
"""Step 2: Filter issues and select subsets.

Three filter conditions for ablation:
- "broad": any genuinely debatable topic
- "political": political, policy, and social issues (default)
- "us_civic": issues relevant to US public deliberation
- "targeted": issues matching eval dataset domains specifically

For each condition, produces three selections:
- random_500.jsonl: 500 random issues
- diverse_500.jsonl: 500 diverse issues (farthest-point sampling)
- diverse_100.jsonl: 100 diverse issues (farthest-point sampling)

Usage:
    python scripts/02_filter_issues.py --condition political
    python scripts/02_filter_issues.py --condition targeted
    python scripts/02_filter_issues.py --skip-filter  # reuse existing filtered issues
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ProjectPaths
from src.generation.issue_filter import IssueFilter, FILTER_PROMPTS
from src.generation.diversity_sampler import select_diverse_issues

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", default="political",
                        choices=list(FILTER_PROMPTS.keys()),
                        help="Filter condition (default: political)")
    parser.add_argument("--skip-filter", action="store_true",
                        help="Reuse existing filtered issues")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = ProjectPaths.auto()
    condition_dir = paths.issues_dir / args.condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    all_issues_path = paths.issues_dir / "all_issues.jsonl"
    filtered_path = condition_dir / "filtered.jsonl"
    rejected_path = condition_dir / "rejected.jsonl"

    # Load all issues
    with open(all_issues_path) as f:
        issues = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(issues)} raw issues")

    # Filter
    if args.skip_filter and filtered_path.exists():
        with open(filtered_path) as f:
            accepted = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(accepted)} pre-filtered issues ({args.condition})")
    else:
        logger.info(f"Filtering with condition: {args.condition}")
        filterer = IssueFilter(condition=args.condition)
        accepted, rejected = filterer.filter_batch(issues)

        with open(filtered_path, "w") as f:
            for issue in accepted:
                f.write(json.dumps(issue) + "\n")
        with open(rejected_path, "w") as f:
            for issue in rejected:
                f.write(json.dumps(issue) + "\n")

        logger.info(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")

    # Select subsets
    selections = [
        ("diverse_500", 500, "diverse"),
        ("diverse_100", 100, "diverse"),
        ("random_500", 500, "random"),
    ]

    for name, n, method in selections:
        if len(accepted) < n:
            logger.info(f"Only {len(accepted)} accepted, using all for {name}")
            selected = accepted
        elif method == "diverse":
            logger.info(f"Selecting {n} diverse issues (farthest-point sampling)...")
            selected = select_diverse_issues(accepted, n=n, seed=args.seed)
        else:
            logger.info(f"Selecting {n} random issues...")
            rng = random.Random(args.seed)
            selected = rng.sample(accepted, n)

        out_path = condition_dir / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for issue in selected:
                f.write(json.dumps(issue) + "\n")

        n_kialo = sum(1 for i in selected if i["source"] == "kialo")
        n_habermas = sum(1 for i in selected if i["source"] == "habermas")
        logger.info(f"  {name}: {len(selected)} issues ({n_kialo} Kialo, {n_habermas} Habermas)")

    logger.info(f"Saved to {condition_dir}/")


if __name__ == "__main__":
    main()
