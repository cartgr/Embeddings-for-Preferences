#!/usr/bin/env python3
"""Generate and evaluate hard eval triplets.

Takes real anchor texts from eval datasets and uses GPT-4o to generate:
  - PARAPHRASE: same stance, completely different wording (semantically distant, preference-aligned)
  - FLIP: opposite stance, nearly identical wording (semantically close, preference-opposed)

A model with good preference geometry should rank paraphrase > flip.
Cosine similarity will likely fail on these (flip uses similar words).

Usage:
    python scripts/09_hard_eval_triplets.py --generate
    python scripts/09_hard_eval_triplets.py --evaluate
    python scripts/09_hard_eval_triplets.py --generate --evaluate
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ProjectPaths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REWRITE_PROMPT = """\
You are given a person's opinion statement on a political or social topic.

Original statement: "{anchor}"

Generate two rewritten versions that test whether a model understands values vs. word overlap:

1. SEMANTIC DISTRACTOR (tricks models that rely on word overlap):
   - Keep nearly IDENTICAL wording and sentence structure as the original
   - Only flip the conclusion/stance to be the opposite
   - Goal: MAXIMIZE word overlap while having opposite meaning
   - Example: "Renewable energy must be prioritized above all else" → "Renewable energy is worth considering, but should NOT be prioritized above all else"

2. PREFERENCE MATCH (requires understanding values, not words):
   - Express the SAME stance and values as the original
   - But use completely DIFFERENT vocabulary, framing, and sentence structure
   - Goal: MINIMIZE word overlap while preserving the underlying position
   - Example: "Renewable energy must be prioritized above all else" → "Fossil fuel dependence is our greatest threat — government must act decisively on clean power"

Respond with valid JSON only:
{{"semantic_distractor": "...", "preference_match": "..."}}"""


def sample_anchors(eval_dir: Path, n: int = 100, seed: int = 42,
                    split: str | None = None,
                    per_dataset: int | None = None,
                    datasets: list[str] | None = None) -> list[dict]:
    """Sample n anchor texts from eval datasets.

    If `split` is given ("val" or "test"), restricts to that split's participants
    (uses the same split_participants routine as the evaluator, so splits match).
    If `per_dataset` is given, samples that many per dataset (distributes evenly).
    If `datasets` is given, restricts to those dataset names.
    """
    from src.evaluation.evaluator import split_participants, SPLIT_SEED, VAL_RATIO
    rng = random.Random(seed)

    all_paths = sorted(eval_dir.glob("*.jsonl"))
    if datasets is not None:
        allowed = set(datasets)
        all_paths = [p for p in all_paths if p.stem in allowed]

    n_per = per_dataset if per_dataset is not None else max(1, n // max(1, len(all_paths)))

    samples = []
    for path in all_paths:
        triplets = [json.loads(l) for l in open(path)]
        if split is not None:
            all_pids = [t["participant_id"] for t in triplets]
            selected_pids = split_participants(all_pids, split, SPLIT_SEED, VAL_RATIO)
            triplets = [t for t in triplets if t["participant_id"] in selected_pids]
        # Gather all unique (participant_id, anchor_text) pairs — avoid anchor repetition
        seen = set()
        candidates = []
        for t in triplets:
            for txt in t["anchor_texts"]:
                if len(txt.strip()) > 20:
                    key = (t["participant_id"], txt)
                    if key not in seen:
                        seen.add(key)
                        candidates.append({
                            "dataset": path.stem,
                            "participant_id": t["participant_id"],
                            "anchor": txt,
                        })
        take = min(n_per, len(candidates))
        samples.extend(rng.sample(candidates, take))

    rng.shuffle(samples)
    if per_dataset is None and len(samples) > n:
        samples = samples[:n]
    return samples


def sample_anchors_from_train(triplet_path: Path, n: int = 100, seed: int = 42,
                               exclude: set[str] | None = None) -> list[dict]:
    """Sample n anchor texts from synthetic training triplets (anchor_text field)."""
    rng = random.Random(seed)
    triplets = [json.loads(l) for l in open(triplet_path)]
    triplets = [t for t in triplets if len(t.get("anchor_text", "").strip()) > 20]
    if exclude:
        triplets = [t for t in triplets if t["anchor_text"] not in exclude]
    sampled = rng.sample(triplets, min(n, len(triplets)))
    return [{"dataset": "train", "anchor": t["anchor_text"]} for t in sampled]


def rewrite_one(client, anchor_text: str, model: str = "gpt-4o") -> dict | None:
    """Call GPT-4o to generate paraphrase and flip. Returns dict or None on failure."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(anchor=anchor_text)}],
            max_tokens=400,
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        if "semantic_distractor" not in data or "preference_match" not in data:
            return None
        return data
    except Exception as e:
        logger.warning(f"Rewrite failed: {e}")
        return None


def generate_hard_triplets(eval_dir: Path, output_path: Path, n: int = 100,
                            model: str = "gpt-4o", max_workers: int = 20, seed: int = 42,
                            source_triplets: Path | None = None,
                            exclude_file: Path | None = None,
                            split: str | None = None,
                            per_dataset: int | None = None):
    """Generate hard triplets and save to JSONL."""
    from openai import OpenAI
    client = OpenAI()

    exclude = set()
    if exclude_file is not None and exclude_file.exists():
        with open(exclude_file) as f:
            for line in f:
                t = json.loads(line)
                exclude.add(t["anchor"])
        logger.info(f"Excluding {len(exclude)} already-used anchors")

    if source_triplets is not None:
        samples = sample_anchors_from_train(source_triplets, n=n, seed=seed, exclude=exclude)
        logger.info(f"Sampled {len(samples)} anchors from training triplets")
    else:
        samples = sample_anchors(eval_dir, n=n, seed=seed, split=split, per_dataset=per_dataset)
        if exclude:
            samples = [s for s in samples if s["anchor"] not in exclude]
        split_note = f" (split={split})" if split else ""
        logger.info(f"Sampled {len(samples)} anchors from eval data{split_note}")

    results = []
    lock = threading.Lock()
    failed = 0

    def _process(sample):
        rewrites = rewrite_one(client, sample["anchor"], model=model)
        if rewrites is None:
            return None
        out = {
            "dataset": sample["dataset"],
            "anchor": sample["anchor"],
            "preference_match": rewrites["preference_match"],
            "semantic_distractor": rewrites["semantic_distractor"],
        }
        if "participant_id" in sample:
            out["participant_id"] = sample["participant_id"]
        if split is not None:
            out["split"] = split
        return out

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, s): s for s in samples}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            with lock:
                if result is None:
                    failed += 1
                else:
                    results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{len(samples)} done, {failed} failed")

    logger.info(f"Generated {len(results)} hard triplets ({failed} failed)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved to {output_path}")
    return results


def cosine_similarity(a, b):
    import numpy as np
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_hard_triplets(triplet_path: Path, model_path: str, model_label: str) -> dict:
    """Evaluate a model on hard triplets. Returns per-dataset and overall accuracy."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import numpy as np
    from src.embedding.model import get_device, load_model

    triplets = [json.loads(l) for l in open(triplet_path)]
    logger.info(f"Evaluating {model_label} on {len(triplets)} hard triplets...")

    device = get_device()
    model = load_model(model_path, device=device)

    # Embed all unique texts
    all_texts = list({t["anchor"] for t in triplets} |
                     {t["preference_match"] for t in triplets} |
                     {t["semantic_distractor"] for t in triplets})
    embs = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
    text_to_emb = dict(zip(all_texts, embs))

    by_dataset = {}
    correct = 0
    total = 0

    for t in triplets:
        a = text_to_emb[t["anchor"]]
        p = text_to_emb[t["preference_match"]]
        f = text_to_emb[t["semantic_distractor"]]

        sim_match = cosine_similarity(a, p)
        sim_distractor = cosine_similarity(a, f)

        ds = t["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = {"correct": 0, "total": 0}

        # Correct = preference_match ranks higher than semantic_distractor
        if sim_match > sim_distractor:
            correct += 1
            by_dataset[ds]["correct"] += 1
        elif sim_match == sim_distractor:
            correct += 0.5
            by_dataset[ds]["correct"] += 0.5
        by_dataset[ds]["total"] += 1
        total += 1

    import torch
    del model
    torch.cuda.empty_cache()

    overall = correct / total if total > 0 else 0
    per_dataset = {ds: v["correct"] / v["total"] for ds, v in by_dataset.items() if v["total"] > 0}
    logger.info(f"  {model_label}: {overall:.1%} overall")
    return {"overall": overall, "per_dataset": per_dataset, "n": total}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--gen-model", default="gpt-4o")
    parser.add_argument("--base-model", default="sentence-transformers/sentence-t5-xl")
    parser.add_argument("--tuned-model",
                        default="data/models/best/sentence_transformers_sentence_t5_xl/"
                                "diverse_2000_retrain/checkpoint-1600")
    parser.add_argument("--output", default=None)
    parser.add_argument("--source-triplets", default=None,
                        help="Path to training triplets JSONL to sample anchors from "
                             "instead of eval data.")
    parser.add_argument("--exclude", default=None,
                        help="Path to existing hard triplets JSONL — anchors already in this "
                             "file will be excluded from new generation.")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Max concurrent API calls for generation.")
    parser.add_argument("--split", default=None, choices=[None, "val", "test", "all"],
                        help="Restrict anchor sampling to a specific participant split "
                             "(uses evaluator's deterministic split). Prevents leakage when "
                             "generated triplets are used as training data.")
    parser.add_argument("--per-dataset", type=int, default=None,
                        help="Sample this many anchors per dataset (distributes evenly). "
                             "Overrides --n.")
    args = parser.parse_args()

    if not args.generate and not args.evaluate:
        parser.error("Specify --generate and/or --evaluate")

    paths = ProjectPaths.auto()
    output_path = Path(args.output) if args.output else \
        paths.triplets_dir / "hard_eval_triplets.jsonl"

    if args.generate:
        generate_hard_triplets(
            paths.eval_dir, output_path,
            n=args.n, model=args.gen_model, max_workers=args.max_workers,
            source_triplets=Path(args.source_triplets) if args.source_triplets else None,
            exclude_file=Path(args.exclude) if args.exclude else None,
            split=args.split if args.split != "all" else None,
            per_dataset=args.per_dataset,
        )

    if args.evaluate:
        if not output_path.exists():
            logger.error(f"Triplet file not found: {output_path}. Run --generate first.")
            sys.exit(1)

        all_results = {}
        for label, model_path in [("base", args.base_model), ("tuned", args.tuned_model)]:
            all_results[label] = evaluate_hard_triplets(output_path, model_path, label)

        # Print table
        print(f"\n{'='*60}")
        print(f"Hard Eval Triplets: preference_match (same values, diff words) vs semantic_distractor (same words, opp stance)")
        print(f"{'='*60}")
        print(f"{'Dataset':<45} {'base':>6} {'tuned':>7}")
        print("-" * 60)

        all_datasets = sorted(set(all_results["base"]["per_dataset"]) |
                               set(all_results["tuned"]["per_dataset"]))
        for ds in all_datasets:
            b = all_results["base"]["per_dataset"].get(ds, 0)
            t = all_results["tuned"]["per_dataset"].get(ds, 0)
            print(f"{ds:<45} {b:>5.1%} {t:>6.1%}")

        print("-" * 60)
        print(f"{'Overall':<45} {all_results['base']['overall']:>5.1%} "
              f"{all_results['tuned']['overall']:>6.1%}")

        result_path = paths.results_dir / "hard_eval_triplets.json"
        with open(result_path, "w") as f:
            json.dump({
                "base_model": args.base_model,
                "tuned_model": args.tuned_model,
                "triplet_file": str(output_path),
                "results": all_results,
            }, f, indent=2)
        logger.info(f"Saved to {result_path}")


if __name__ == "__main__":
    main()
