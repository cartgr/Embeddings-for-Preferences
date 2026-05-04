#!/usr/bin/env python3
"""Step 9: Generate hard evaluation triplets.

Creates triplets where semantic similarity and preference alignment are decoupled:
- Cross-topic hard: same stance on different topics (semantically distant but preference-aligned)
- Within-topic hard: opposite stance with similar wording (semantically close but preference-opposed)

Usage:
    python scripts/09_generate_hard_triplets.py
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ProjectPaths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_cross_topic_hard(opinions_data: list, n: int = 50, seed: int = 42) -> list:
    """Generate cross-topic hard triplets.

    Anchor: opinion at position P on issue A
    Semantic distractor: opinion at position P on issue A (similar wording, same topic)
    Preference match: opinion at position P on issue B (different topic, same stance)

    These test whether the model captures stance alignment across topics
    rather than just topical similarity.
    """
    rng = random.Random(seed)
    triplets = []

    # Sample pairs of issues
    issue_pairs = [(i, j) for i in range(len(opinions_data)) for j in range(i + 1, len(opinions_data))]
    rng.shuffle(issue_pairs)

    for i, j in issue_pairs[:n * 2]:
        data_a = opinions_data[i]
        data_b = opinions_data[j]

        if len(data_a["opinions"]) != 5 or len(data_b["opinions"]) != 5:
            continue

        # Anchor = strongly pro (position 0) on issue A
        # Preference match = strongly pro (position 0) on issue B (same stance, different topic)
        # Semantic distractor = strongly anti (position 4) on issue A (same topic, opposite stance)
        for anchor_pos in [0, 4]:
            opposite_pos = 4 - anchor_pos
            triplets.append({
                "id": len(triplets) + 1,
                "topic": f"{data_a['issue_id']} vs {data_b['issue_id']}",
                "anchor": data_a["opinions"][anchor_pos],
                "semantic_distractor": data_a["opinions"][opposite_pos],
                "preference_match": data_b["opinions"][anchor_pos],
                "type": "cross_topic_hard",
            })

        if len(triplets) >= n:
            break

    return triplets[:n]


def generate_within_topic_hard(opinions_data: list, n: int = 100, seed: int = 42) -> list:
    """Generate within-topic hard triplets.

    Anchor: opinion at position P
    Semantic distractor: adjacent opinion at position P+1 (very similar wording)
    Preference match: opinion at position P on a different issue (same stance)

    These test whether the model can distinguish adjacent positions that
    are semantically very similar but represent different preferences.
    """
    rng = random.Random(seed)
    triplets = []

    issues = [d for d in opinions_data if len(d["opinions"]) == 5]
    rng.shuffle(issues)

    for idx, data in enumerate(issues):
        if len(triplets) >= n:
            break

        # Find a different issue for cross-topic positive
        other = issues[(idx + 1) % len(issues)]

        for anchor_pos in [0, 1, 3, 4]:
            # Adjacent position (semantic distractor)
            if anchor_pos < 4:
                distractor_pos = anchor_pos + 1
            else:
                distractor_pos = anchor_pos - 1

            triplets.append({
                "id": len(triplets) + 1,
                "topic": data["issue_id"],
                "anchor": data["opinions"][anchor_pos],
                "semantic_distractor": data["opinions"][distractor_pos],
                "preference_match": other["opinions"][anchor_pos],
                "type": "within_topic_hard",
            })

            if len(triplets) >= n:
                break

    return triplets[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cross", type=int, default=50)
    parser.add_argument("--n-within", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = ProjectPaths.auto()
    opinions_path = paths.opinions_dir / "opinions.jsonl"

    with open(opinions_path) as f:
        opinions_data = [json.loads(line) for line in f]
    logger.info(f"Loaded opinions for {len(opinions_data)} issues")

    cross = generate_cross_topic_hard(opinions_data, n=args.n_cross, seed=args.seed)
    within = generate_within_topic_hard(opinions_data, n=args.n_within, seed=args.seed)

    all_triplets = cross + within
    logger.info(f"Generated {len(cross)} cross-topic + {len(within)} within-topic hard triplets")

    output_path = paths.triplets_dir / "hard_triplets.jsonl"
    with open(output_path, "w") as f:
        for t in all_triplets:
            f.write(json.dumps(t) + "\n")

    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
