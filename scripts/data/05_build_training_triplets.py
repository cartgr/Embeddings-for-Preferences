#!/usr/bin/env python3
"""Step 4: Build training triplets from generated opinions.

Within-topic triplets: from position ordering (free, deterministic).
Cross-topic triplets: from GPT-4o LLM judge (costs ~$9 per 3000).

Usage:
    python scripts/04_build_triplets.py --condition us_civic --selection diverse_100 --gen-model gpt-4o
    python scripts/04_build_triplets.py --condition political --selection diverse_500 --gen-model claude-sonnet-4-20250514
    python scripts/04_build_triplets.py --condition us_civic --selection diverse_100 --gen-model gpt-4o --skip-cross
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ProjectPaths
from src.generation.issue_filter import FILTER_PROMPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SELECTIONS = ["diverse_500", "diverse_100", "random_500"]


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").replace(".", "_")


def build_within_topic_triplets(opinions_data: list) -> list:
    """Build within-topic triplets from position ordering."""
    triplets = []
    for data in opinions_data:
        issue_id = data["issue_id"]
        opinions = data["opinions"]
        if len(opinions) != 5:
            continue

        for anchor_pos in range(5):
            for pos_pos in range(5):
                for neg_pos in range(5):
                    if anchor_pos == pos_pos or anchor_pos == neg_pos or pos_pos == neg_pos:
                        continue
                    pos_dist = abs(anchor_pos - pos_pos)
                    neg_dist = abs(anchor_pos - neg_pos)
                    if pos_dist >= neg_dist:
                        continue

                    diff = neg_dist - pos_dist
                    difficulty = "easy" if diff >= 3 else "medium" if diff == 2 else "hard"

                    triplets.append({
                        "anchor_text": opinions[anchor_pos],
                        "pos_text": opinions[pos_pos],
                        "neg_text": opinions[neg_pos],
                        "anchor_issue_id": issue_id,
                        "pos_issue_id": issue_id,
                        "neg_issue_id": issue_id,
                        "triplet_type": "within_topic",
                        "difficulty": difficulty,
                    })
    return triplets


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=list(FILTER_PROMPTS.keys()))
    parser.add_argument("--selection", required=True, choices=SELECTIONS)
    parser.add_argument("--gen-model", required=True, help="Model used for opinion generation")
    parser.add_argument("--n-cross", type=int, default=3000, help="Number of cross-topic triplets")
    parser.add_argument("--cross-workers", type=int, default=50)
    parser.add_argument("--skip-cross", action="store_true", help="Only build within-topic triplets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = ProjectPaths.auto()

    # Load selected issue IDs
    issues_path = paths.issues_dir / args.condition / f"{args.selection}.jsonl"
    with open(issues_path) as f:
        selected_ids = set(json.loads(line)["issue_id"] for line in f)
    logger.info(f"Selected {len(selected_ids)} issues ({args.condition}/{args.selection})")

    # Load opinions, filter to selected issues
    slug = model_slug(args.gen_model)
    opinions_path = paths.opinions_dir / slug / "opinions.jsonl"
    if not opinions_path.exists():
        logger.error(f"Opinions not found: {opinions_path}")
        sys.exit(1)

    opinions_data = []
    with open(opinions_path) as f:
        for line in f:
            data = json.loads(line)
            if data["issue_id"] in selected_ids:
                opinions_data.append(data)
    logger.info(f"Loaded opinions for {len(opinions_data)}/{len(selected_ids)} selected issues")

    # Within-topic triplets
    within = build_within_topic_triplets(opinions_data)
    logger.info(f"Within-topic: {len(within)}")
    diff_counts = Counter(t["difficulty"] for t in within)
    for d in ["easy", "medium", "hard"]:
        logger.info(f"  {d}: {diff_counts.get(d, 0)}")

    # Cross-topic triplets via LLM judge with position bias control
    cross = []
    cross_stats = {}
    if not args.skip_cross:
        from openai import OpenAI
        from src.generation.cross_topic import CrossTopicJudge

        client = OpenAI()
        judge = CrossTopicJudge(client=client, model="gpt-4o")

        cross, cross_stats = judge.generate_triplets(
            opinions_data=opinions_data,
            issue_ids=selected_ids,
            n_triplets=args.n_cross,
            max_workers=args.cross_workers,
            seed=args.seed,
        )
        logger.info(f"Cross-topic: {len(cross)}")

    # Combine and shuffle
    rng = random.Random(args.seed)
    all_triplets = within + cross
    rng.shuffle(all_triplets)

    # Save
    output_dir = paths.triplets_dir / slug / args.condition / args.selection
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "triplets.jsonl"

    with open(output_path, "w") as f:
        for t in all_triplets:
            f.write(json.dumps(t) + "\n")

    metadata = {
        "condition": args.condition,
        "selection": args.selection,
        "gen_model": args.gen_model,
        "n_issues": len(opinions_data),
        "n_within": len(within),
        "n_cross": len(cross),
        "n_total": len(all_triplets),
        "seed": args.seed,
        "cross_topic_stats": cross_stats,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved {len(all_triplets)} triplets to {output_path}")


if __name__ == "__main__":
    main()
