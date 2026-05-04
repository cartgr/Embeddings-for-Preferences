#!/usr/bin/env python3
"""Step 2 — Download Habermas + Kialo issues and merge into one pool.

Loads the Habermas Machine dataset (parquet files on GCS) and the Kialo
argument topics dataset (HuggingFace), extracts issue text from each,
and writes a unified JSONL:

    data/processed/issues/all_issues.jsonl

Each line:
    {"issue_id": "kialo_42",
     "source":   "kialo" | "habermas",
     "issue_text": "Should UBI replace welfare?"}

This pool is the input to step 3 (issue filtering) and subsequently to
step 4 (opinion generation).

Usage:
    python scripts/data/02_download_issue_sources.py
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import ProjectPaths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_training_issues(paths: ProjectPaths):
    """Download Kialo + Habermas and write all_issues.jsonl."""
    from src.data.kialo_loader import KialoLoader
    from src.data.habermas_loader import HabermasLoader

    paths.issues_dir.mkdir(parents=True, exist_ok=True)
    output_path = paths.issues_dir / "all_issues.jsonl"

    if output_path.exists():
        n = sum(1 for _ in open(output_path))
        logger.info(f"  Training issues already processed: {n} issues")
        return

    issues = []

    logger.info("  Loading Kialo from HuggingFace...")
    kialo_bank = KialoLoader().load()
    for iid, issue in kialo_bank.issues.items():
        issues.append({"issue_id": iid, "source": "kialo", "issue_text": issue.issue_text})
    logger.info(f"    Kialo: {len(kialo_bank.issues)} issues")

    logger.info("  Loading Habermas from GCS...")
    paths.habermas_dir.mkdir(parents=True, exist_ok=True)
    habermas_bank = HabermasLoader(str(paths.habermas_dir)).load()
    for iid, issue in habermas_bank.issues.items():
        issues.append({"issue_id": iid, "source": "habermas", "issue_text": issue.issue_text})
    logger.info(f"    Habermas: {len(habermas_bank.issues)} issues")

    with open(output_path, "w") as f:
        for issue in issues:
            f.write(json.dumps(issue) + "\n")
    logger.info(f"  Saved {len(issues)} issues → {output_path}")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    paths = ProjectPaths.auto()
    process_training_issues(paths)


if __name__ == "__main__":
    main()
