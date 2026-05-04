#!/usr/bin/env python3
"""Generate 'normal' counterfactual training triplets — the controlled
counterpart to the hard-triplet training set used by DPT.

The hard prompt (07_generate_hard_eval.py) deliberately MAXIMISES word
overlap on the semantic_distractor (same words, flipped stance) and
MINIMISES it on the preference_match (different words, same stance) —
inverting the natural wording-vs-stance correlation that produces
regime (i) of Prop 1. This script's prompt does the SAME triplet
construction (anchor + preference_match + semantic_distractor) but
DOES NOT engineer the surface overlap, so the natural correlation is
preserved (regime i, E[Delta_t] > 0).

Used in the §sec:diagnosis ablation 'is the lift from any synthetic
data, or specifically from decorrelated training?' — predicted by
Theorem 1: training on these normal triplets should NOT produce a
hard-eval lift, because H2 fails (E[Delta_t | G] > 0 here).

Output: data/processed/triplets/normal_train_triplets_<N>.jsonl
        Same schema as hard_train_triplets_10000.jsonl:
        {dataset, anchor, preference_match, semantic_distractor}
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ProjectPaths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Symmetric inverse of the hard rewrite prompt
# (07_generate_hard_eval.py:REWRITE_PROMPT). Same triplet shape and JSON
# schema, but the ROLES of MAXIMISE vs MINIMISE word overlap are flipped:
# hard puts max-overlap on the distractor, normal puts max-overlap on
# the match. The result is regime (i) by construction — the preferred
# (same stance) shares MORE wording with the anchor than the
# dispreferred (different stance) does, so E[Delta_t] > 0.
NORMAL_REWRITE_PROMPT = """\
You are given a person's opinion statement on a political or social topic.

Original statement: "{anchor}"

Generate two rewritten versions that exhibit the natural correlation
between stance and wording — same stance tends to share words, opposite
stance tends to use different words:

1. PREFERENCE MATCH (high word overlap with the anchor, same stance):
   - Reuse most of the anchor's vocabulary and content words
   - Preserve the original stance and values
   - BUT it must be a genuine paraphrase — NOT a verbatim or near-verbatim
     copy of the anchor. Change at least 3-5 words to synonyms, rearrange
     clauses, or restructure the sentence; otherwise the rewrite is invalid.
   - Goal: MAXIMIZE word overlap while still meaningfully rewording
   - Example: "Renewable energy must be prioritized above all else" → "We must put renewable energy ahead of every other concern"

2. SEMANTIC DISTRACTOR (low word overlap with the anchor, opposite stance):
   - Use completely DIFFERENT vocabulary, framing, and sentence structure
   - Express the OPPOSITE stance on the same underlying issue
   - Goal: MINIMIZE word overlap while having opposite meaning
   - Example: "Renewable energy must be prioritized above all else" → "Cheap, reliable fossil fuels are what people actually need — chasing wind and solar comes at too great a cost."

Respond with valid JSON only:
{{"preference_match": "...", "semantic_distractor": "..."}}"""


def rewrite_one(client, anchor_text: str, model: str = "gpt-4o", retries: int = 3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": NORMAL_REWRITE_PROMPT.format(anchor=anchor_text)}],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=400,
            )
            data = json.loads(resp.choices[0].message.content)
            if "preference_match" in data and "semantic_distractor" in data:
                return data
        except Exception as e:
            if attempt == retries - 1:
                logger.warning(f"rewrite failed: {e}")
                return None
            time.sleep(1.5 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opinions",
                    default="data/processed/opinions/sonnet_standalone_prompt/opinions.jsonl",
                    help="Source pool of synthetic opinions; each row has 5 opinions per issue.")
    ap.add_argument("--n", type=int, default=10000,
                    help="Number of triplets to produce (matches hard_train_triplets_10000).")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--max-workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None,
                    help="Default: data/processed/triplets/normal_train_triplets_<N>.jsonl")
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    out_path = Path(args.output or
                    f"data/processed/triplets/normal_train_triplets_{args.n}.jsonl")

    # Load synthetic opinions, flatten to a per-anchor list.
    issues = [json.loads(l) for l in open(args.opinions)]
    anchors = []
    for d in issues:
        for op in d.get("opinions", []):
            anchors.append({"issue_id": d.get("issue_id"), "anchor": op})
    logger.info(f"opinion pool: {len(issues)} issues -> {len(anchors)} candidate anchors")

    rng = random.Random(args.seed)
    rng.shuffle(anchors)
    sampled = anchors[:args.n]
    logger.info(f"sampled {len(sampled)} anchors -> {args.n} triplets")

    # Resume: read any existing rows from out_path so we don't redo them.
    done_anchors = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done_anchors.add(json.loads(line)["anchor"])
            except Exception:
                pass
        logger.info(f"already have {len(done_anchors)} triplets from previous runs")

    pending = [a for a in sampled if a["anchor"] not in done_anchors]
    logger.info(f"need to rewrite {len(pending)} new anchors")
    if not pending:
        logger.info("nothing to do.")
        return

    from openai import OpenAI
    client = OpenAI()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "a")
    n_ok = n_fail = 0

    def work(a):
        return a, rewrite_one(client, a["anchor"], args.model)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for i, fut in enumerate(as_completed(pool.submit(work, a) for a in pending), 1):
            a, rewrites = fut.result()
            if rewrites is None:
                n_fail += 1; continue
            row = {
                "dataset": "train_normal",
                "issue_id": a.get("issue_id"),
                "anchor": a["anchor"],
                "preference_match":   rewrites["preference_match"],
                "semantic_distractor": rewrites["semantic_distractor"],
            }
            fout.write(json.dumps(row) + "\n"); fout.flush()
            n_ok += 1
            if i % 100 == 0:
                logger.info(f"  {i}/{len(pending)}  ok={n_ok}  fail={n_fail}")

    fout.close()
    logger.info(f"done. wrote {n_ok} new triplets, {n_fail} failures, total at {out_path}")


if __name__ == "__main__":
    main()
