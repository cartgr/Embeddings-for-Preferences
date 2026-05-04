#!/usr/bin/env python3
"""Step 5: Train a single preference-aware embedding model.

LoRA fine-tunes a sentence-transformer on synthetic preference triplets
using Bradley-Terry loss.

Usage:
    python scripts/05_train.py --model sentence-transformers/sentence-t5-large \
        --condition us_civic --selection diverse_100 --gen-model gpt-4o

    python scripts/05_train.py --model sentence-transformers/sentence-t5-large \
        --condition political --selection diverse_500 --gen-model claude-sonnet-4-20250514 \
        --seed 42 --lr 5e-5
"""

import argparse
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ProjectPaths
from src.embedding.trainer import TrainingConfig, EmbeddingTrainer
from src.evaluation.evaluator import EmbeddingEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").replace(".", "_")


def load_and_split_triplets(path: Path):
    cross, within = [], []
    with open(path) as f:
        for line in f:
            t = json.loads(line.strip())
            if t.get("triplet_type") == "cross_topic":
                cross.append(t)
            else:
                within.append(t)
    return cross, within


def load_hard_triplets(path: Path) -> list:
    """Load hard triplets and remap fields to standard training schema."""
    triplets = []
    with open(path) as f:
        for line in f:
            t = json.loads(line.strip())
            triplets.append({
                "anchor_text": t["anchor"],
                "pos_text": t["preference_match"],
                "neg_text": t["semantic_distractor"],
                "triplet_type": "hard",
            })
    return triplets


def sample_triplets(triplet_path: Path, total: int, cross_ratio: float = 0.25, seed: int = 42,
                    hard_triplets_path: Path | None = None, hard_ratio: float = 0.0,
                    hard_cap: int | None = None):
    rng = random.Random(seed)

    if hard_triplets_path is not None and hard_ratio > 0:
        hard = load_hard_triplets(hard_triplets_path)
        if hard_cap is not None:
            hard = rng.sample(hard, min(hard_cap, len(hard)))
        cross, within = load_and_split_triplets(triplet_path)
        normal = cross + within

        if hard_ratio == 1.0:
            total = len(hard)
        else:
            # Cap total by whichever pool runs out first (no replacement)
            max_from_hard = len(hard)
            max_from_normal = len(normal)
            total = int(min(max_from_hard / hard_ratio, max_from_normal / (1 - hard_ratio)))

        n_hard = int(total * hard_ratio)
        n_normal = total - n_hard

        sampled = rng.sample(hard, min(n_hard, len(hard)))
        sampled += rng.sample(normal, min(n_normal, len(normal)))
        rng.shuffle(sampled)
        logger.info(f"Mixed {len([t for t in sampled if t.get('triplet_type')=='hard'])} hard + "
                    f"{len([t for t in sampled if t.get('triplet_type')!='hard'])} normal triplets "
                    f"({total} total, {total // 16} steps)")
        return sampled

    # Standard sampling (no hard triplets)
    cross, within = load_and_split_triplets(triplet_path)
    n_cross = int(total * cross_ratio)
    n_within = total - n_cross
    sampled = rng.sample(cross, min(n_cross, len(cross)))
    sampled += rng.sample(within, min(n_within, len(within) if within else 0))
    rng.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--gen-model", required=True, help="Model used for opinion generation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--loss", choices=["bradley_terry", "infonce"], default="bradley_terry")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="BT loss temperature. <1.0 sharpens loss, encouraging larger margins.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--cross-ratio", type=float, default=0.25)
    parser.add_argument("--hard-triplets", default=None,
                        help="Path to hard triplets JSONL for mixing into training.")
    parser.add_argument("--hard-ratio", type=float, default=0.0,
                        help="Fraction of training triplets that are hard (0.0 = none, 1.0 = all hard).")
    parser.add_argument("--hard-cap", type=int, default=None,
                        help="Max number of hard triplets to sample from the hard triplets file.")
    parser.add_argument("--hard-eval-triplets", default=None,
                        help="Path to hard eval triplets JSONL for post-training hard triplet evaluation.")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--query-prefix", default="",
                        help="String prepended to anchor (query) texts at training and eval time. "
                             "E.g. 'query: ' for e5, the BGE retrieval prompt for BGE-large-en-v1.5.")
    parser.add_argument("--passage-prefix", default="",
                        help="String prepended to preferred/dispreferred (passage) texts at training "
                             "and eval time. E.g. 'passage: ' for e5; '' for BGE-style asymmetric.")
    args = parser.parse_args()

    paths = ProjectPaths.auto()
    gen_slug = model_slug(args.gen_model)
    triplet_path = paths.triplets_dir / gen_slug / args.condition / args.selection / "triplets.jsonl"

    if not triplet_path.exists():
        logger.error(f"Triplets not found: {triplet_path}")
        sys.exit(1)

    if args.output_dir is None:
        m_slug = model_slug(args.model)
        args.output_dir = str(
            paths.best_dir / m_slug / gen_slug / args.condition / args.selection / f"seed{args.seed}"
        )

    # Sample triplets. The trainer uses an effective batch of
    # batch_size * gradient_accumulation_steps per optimizer step, so
    # the number of unique triplets we need is max_steps times that
    # effective batch (not just per-device batch). Without this,
    # large grad_accum settings sample too few unique triplets and
    # silently train for multiple epochs.
    hard_path = Path(args.hard_triplets) if args.hard_triplets else None
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    total_triplets = args.max_steps * effective_batch
    triplets = sample_triplets(
        triplet_path, total_triplets, args.cross_ratio, args.seed,
        hard_triplets_path=hard_path, hard_ratio=args.hard_ratio,
        hard_cap=args.hard_cap,
    )
    # Recompute max_steps from actual triplet count (may differ when hard mixing caps total)
    actual_steps = len(triplets) // effective_batch
    if actual_steps != args.max_steps:
        logger.info(f"Adjusted max_steps: {args.max_steps} → {actual_steps} (data budget)")
        args.max_steps = actual_steps
    logger.info(f"Sampled {len(triplets)} triplets → {args.max_steps} optimizer steps "
                f"(per-device batch {args.batch_size} x grad_accum {args.gradient_accumulation_steps})")

    # Apply per-encoder query/passage prefixes to the training data so the
    # LoRA learns to operate in the model's native asymmetric retrieval
    # geometry (e.g. e5 expects 'query: '/'passage: '). For symmetric
    # encoders (sentence-T5, mpnet) both prefixes default to '' and this
    # is a no-op.
    if args.query_prefix or args.passage_prefix:
        logger.info(f"Applying training-time prefixes: "
                    f"query={args.query_prefix!r}  passage={args.passage_prefix!r}")
        for t in triplets:
            t["anchor_text"] = args.query_prefix + t["anchor_text"]
            t["pos_text"]    = args.passage_prefix + t["pos_text"]
            t["neg_text"]    = args.passage_prefix + t["neg_text"]

    temp_path = Path(args.output_dir).parent / f"_train_{os.getpid()}.jsonl"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    try:
        config = TrainingConfig(
            base_model=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_steps=args.max_steps,
            use_lora=True,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            loss_type=args.loss,
            bt_temperature=args.temperature,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_wandb=False,
            eval_steps=args.max_steps + 1,
            skip_final_eval=True,
            seed=args.seed,
        )

        trainer = EmbeddingTrainer(config)
        model = trainer.train(temp_path)

        # Release training-phase memory (HF Trainer's optimizer state,
        # gradient buffers, activation caches) before eval. On the
        # 20-GiB MIG slice the post-train memory + eval activations was
        # tipping over 19.5 GiB and OOMing in the per-config sweeps.
        try:
            del trainer
            import gc; gc.collect()
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

        if not args.no_eval:
            # Build a Formatter that applies the same prefixes the model was
            # trained with — so eval encodes anchors as queries and items as
            # passages in the model's native geometry. PlainFormatter is a
            # no-op when both prefixes are empty (symmetric encoders).
            from src.evaluation.formatters import PlainFormatter, PrefixFormatter
            eval_fmt = (PrefixFormatter(query_prefix=args.query_prefix,
                                        passage_prefix=args.passage_prefix)
                        if (args.query_prefix or args.passage_prefix) else None)

            evaluator = EmbeddingEvaluator(eval_dir=paths.eval_dir)
            val_results  = evaluator.evaluate_all(model, split="val",  formatter=eval_fmt)
            test_results = evaluator.evaluate_all(model, split="test", formatter=eval_fmt)
            all_results  = evaluator.evaluate_all(model, split="all",  formatter=eval_fmt)

            # Hard triplet eval
            hard_eval_results = None
            if args.hard_eval_triplets:
                from src.embedding.model import get_device
                import numpy as np

                def cosine_sim(a, b):
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

                hard_triplets_eval = [json.loads(l) for l in open(args.hard_eval_triplets)]
                # Accept either schema: {preference_match, semantic_distractor} or {paraphrase, flip}
                def _pos(t): return t.get("preference_match", t.get("paraphrase"))
                def _neg(t): return t.get("semantic_distractor", t.get("flip"))
                try:
                    anchor_texts = sorted({t["anchor"] for t in hard_triplets_eval})
                    item_texts = sorted({_pos(t) for t in hard_triplets_eval} |
                                        {_neg(t) for t in hard_triplets_eval})
                    # Apply the same query/passage prefixes used at training time.
                    a_in = [args.query_prefix + a for a in anchor_texts]
                    i_in = [args.passage_prefix + p for p in item_texts]
                    a_emb = model.encode(a_in, convert_to_numpy=True, show_progress_bar=False)
                    i_emb = model.encode(i_in, convert_to_numpy=True, show_progress_bar=False)
                    a_map = dict(zip(anchor_texts, a_emb))
                    i_map = dict(zip(item_texts, i_emb))
                    correct = sum(
                        1 for t in hard_triplets_eval
                        if cosine_sim(a_map[t["anchor"]], i_map[_pos(t)]) >
                           cosine_sim(a_map[t["anchor"]], i_map[_neg(t)])
                    )
                    hard_eval_results = correct / len(hard_triplets_eval)
                    logger.info(f"Hard triplet accuracy: {hard_eval_results:.1%}")
                except Exception as e:
                    logger.warning(f"Hard triplet eval skipped: {e}")
                    hard_eval_results = None

            results_path = Path(args.output_dir) / "results.json"
            with open(results_path, "w") as f:
                json.dump({
                    "model": args.model,
                    "gen_model": args.gen_model,
                    "condition": args.condition,
                    "selection": args.selection,
                    "seed": args.seed,
                    "hyperparams": {
                        "lr": args.lr, "max_steps": args.max_steps,
                        "batch_size": args.batch_size, "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "hard_ratio": args.hard_ratio,
                        "hard_cap": args.hard_cap,
                        "temperature": args.temperature,
                        "query_prefix": args.query_prefix,
                        "passage_prefix": args.passage_prefix,
                    },
                    "val_metrics": val_results,
                    "test_metrics": test_results,
                    "all_metrics": all_results,
                    "hard_eval_accuracy": hard_eval_results,
                }, f, indent=2)
            logger.info(f"Results saved to {results_path}")

        del model
        torch.cuda.empty_cache()

    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
