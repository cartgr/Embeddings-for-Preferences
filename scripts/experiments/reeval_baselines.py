#!/usr/bin/env python3
"""Re-evaluate every base baseline on all 11 datasets + 875 hard triplets,
applying each model's intended query/passage encoding format.

The previous per_model/base_*_all.json files were produced with
plain symmetric `model.encode(text)`, which under-evaluates retrieval-
trained encoders (e5, BGE, MxBAI, Arctic, GTE-Qwen, Stella, Voyage)
that expect asymmetric query/passage formatting. This script replaces
those caches with format-correct numbers and adds remesh_right_to_assemble
where it was missing.

Per model, writes:
  data/results/per_model/base_<slug>_all.json
    {"model": ..., "split": "all",
     "formatter": <name>,
     "metrics": {<11 datasets>: accuracy},
     "hard_acc": float,    # 875-triplet hard eval
     "hard_n": 875}

And updates the unified cross-encoder cache:
  data/results/hard_triplet_cosine_1k.json
    appends/replaces an entry per model with
      {hard_acc, hard_n, normal_mean, normal_per_dataset, formatter}

Disk-aware: if --cleanup-after-new is set, deletes the HF cache for
models that were not present before this script ran (peak disk =
one model at a time).
"""
import argparse
import json
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import ProjectPaths
from src.evaluation.evaluator import EVAL_DATASETS, cosine_similarity
from src.evaluation.formatters import (
    dispatch_formatter, PlainFormatter,
    QwenInstructFormatter, StellaFormatter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# --- the 25 base baselines + variants for instruction-tuned models -------
# Each row: (model_id_for_loading, slug, formatter_override_or_None,
#            display_id_for_keys_or_None).
# - formatter_override=None -> use dispatch_formatter(model_id)
# - display_id_for_keys=None -> use model_id (same string used as the
#   key in hard_triplet_cosine_1k.json + the 'model' field in per_model)
# Variant rows on the same HF model get distinct slugs and display_ids
# so they don't overwrite each other's caches.
QWEN_REGISTERED_TASK = "Given a web search query, retrieve relevant passages that answer the query"

BASELINES = [
    # Sentence-T5 family (symmetric)
    ("sentence-transformers/sentence-t5-xl",       "t5_xl",                                          None, None),
    ("sentence-transformers/sentence-t5-large",    "sentence_transformers_sentence_t5_large",        None, None),
    ("sentence-transformers/sentence-t5-base",     "sentence_transformers_sentence_t5_base",         None, None),
    # OpenAI (symmetric)
    ("text-embedding-3-large",                     "text_embedding_3_large",                         None, None),
    ("text-embedding-3-small",                     "text_embedding_3_small",                         None, None),
    # Voyage (asymmetric, input_type)
    ("voyage-4-large",                             "voyage_4_large",                                 None, None),
    ("voyage-4",                                   "voyage_4",                                       None, None),
    ("voyage-4-lite",                              "voyage_4_lite",                                  None, None),
    ("voyage-3-large",                             "voyage_3_large",                                 None, None),
    ("voyage-3",                                   "voyage_3",                                       None, None),
    # Generic SBERT-style (symmetric)
    ("sentence-transformers/nli-mpnet-base-v2",    "sentence_transformers_nli_mpnet_base_v2",        None, None),
    ("sentence-transformers/all-mpnet-base-v2",    "sentence_transformers_all_mpnet_base_v2",        None, None),
    ("sentence-transformers/all-distilroberta-v1", "sentence_transformers_all_distilroberta_v1",     None, None),
    ("sentence-transformers/all-MiniLM-L12-v2",    "sentence_transformers_all_MiniLM_L12_v2",        None, None),
    ("sentence-transformers/all-MiniLM-L6-v2",     "sentence_transformers_all_MiniLM_L6_v2",         None, None),
    ("sentence-transformers/paraphrase-MiniLM-L6-v2", "sentence_transformers_paraphrase_MiniLM_L6_v2", None, None),
    # Older GTE (symmetric, no instruction)
    ("thenlper/gte-large",                         "thenlper_gte_large",                             None, None),
    # Asymmetric retrieval (BGE / MxBAI / Arctic — query instruction; Arctic v2 = E5-style)
    ("BAAI/bge-large-en-v1.5",                     "BAAI_bge_large_en_v1.5",                         None, None),
    ("mixedbread-ai/mxbai-embed-large-v1",         "mixedbread_ai_mxbai_embed_large_v1",             None, None),
    ("Snowflake/snowflake-arctic-embed-l-v2.0",    "Snowflake_snowflake-arctic-embed-l-v2_0",        None, None),
    # E5 family — query:/passage: prefix
    ("intfloat/e5-large-v2",                       "intfloat_e5_large_v2",                           None, None),
    # Stance / contradiction (special-purpose)
    ("vahidthegreat/StanceAware-SBERT",            "vahidthegreat_StanceAware_SBERT",                None, None),
    ("SparseCL/BGE-SparseCL-arguana",              "SparseCL_BGE_SparseCL_arguana",                  None, None),
    # Instruction-tuned 1.5B encoders — both task-tailored AND registered prompt.
    ("Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "Alibaba-NLP_gte-Qwen2-1_5B-instruct",                         None, None),
    ("Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "Alibaba-NLP_gte-Qwen2-1_5B-instruct__registered",
        QwenInstructFormatter(task=QWEN_REGISTERED_TASK),
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct [registered]"),
    ("dunzhang/stella_en_1.5B_v5",
        "dunzhang_stella_en_1.5B_v5",                                  None, None),
    ("dunzhang/stella_en_1.5B_v5",
        "dunzhang_stella_en_1.5B_v5__s2s_query",
        StellaFormatter(prompt_name="s2s_query"),
        "dunzhang/stella_en_1.5B_v5 [s2s_query]"),
    ("dunzhang/stella_en_1.5B_v5",
        "dunzhang_stella_en_1.5B_v5__s2p_query",
        StellaFormatter(prompt_name="s2p_query"),
        "dunzhang/stella_en_1.5B_v5 [s2p_query]"),
]


def kind_of(model_id: str) -> str:
    if model_id.startswith("text-embedding-"):
        return "openai"
    if model_id.startswith("voyage-"):
        return "voyage"
    return "hf"


def hf_cache_dir(model_id: str) -> Path:
    return HF_CACHE / f"models--{model_id.replace('/', '--')}"


def load_model_for(model_id: str, kind: str):
    if kind == "openai":
        from src.evaluation.baselines import OpenAIEmbedder
        return OpenAIEmbedder(model=model_id)
    if kind == "voyage":
        from src.evaluation.baselines import VoyageEmbedder
        return VoyageEmbedder(model=model_id)
    from src.embedding.model import get_device, load_model
    return load_model(model_id, device=get_device())


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


def eval_natural(model, formatter, eval_dir: Path, max_per_ds=2000, seed=0):
    import random
    rng = random.Random(seed)
    per_ds = {}
    for ds in EVAL_DATASETS:
        path = eval_dir / f"{ds}.jsonl"
        if not path.exists():
            continue
        triplets = [json.loads(l) for l in open(path)]
        if len(triplets) > max_per_ds:
            triplets = rng.sample(triplets, max_per_ds)
        anchors = sorted({a for t in triplets for a in t["anchor_texts"]})
        items = sorted({t["preferred"] for t in triplets} |
                       {t["dispreferred"] for t in triplets})
        a_emb = formatter.encode_queries(model, anchors)
        i_emb = formatter.encode_passages(model, items)
        a_map = dict(zip(anchors, a_emb))
        i_map = dict(zip(items, i_emb))
        correct = total = 0
        for t in triplets:
            anc = np.mean([a_map[x] for x in t["anchor_texts"] if x in a_map], axis=0)
            if np.linalg.norm(anc) == 0: continue
            sp = cosine(anc, i_map[t["preferred"]])
            sd = cosine(anc, i_map[t["dispreferred"]])
            if sp > sd: correct += 1
            elif sp == sd: correct += 0.5
            total += 1
        if total:
            per_ds[ds] = correct / total
            logger.info(f"     {ds}: {per_ds[ds]:.3f}  (n={total})")
    mean = sum(per_ds.values()) / len(per_ds) if per_ds else 0.0
    return mean, per_ds


def eval_hard(model, formatter, hard_path: Path):
    hard = [json.loads(l) for l in open(hard_path)]
    anchors = sorted({t["anchor"] for t in hard})
    items = sorted({t["preference_match"] for t in hard} |
                   {t["semantic_distractor"] for t in hard})
    a_emb = formatter.encode_queries(model, anchors)
    i_emb = formatter.encode_passages(model, items)
    a_map = dict(zip(anchors, a_emb))
    i_map = dict(zip(items, i_emb))
    correct = 0
    for t in hard:
        sp = cosine(a_map[t["anchor"]], i_map[t["preference_match"]])
        sd = cosine(a_map[t["anchor"]], i_map[t["semantic_distractor"]])
        if sp > sd: correct += 1
        elif sp == sd: correct += 0.5
    return correct / len(hard), len(hard)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-model-dir", default="data/results/per_model")
    ap.add_argument("--hard-cache",    default="data/results/hard_triplet_cosine_1k.json")
    ap.add_argument("--hard-triplets", default="data/processed/triplets/hard_eval_triplets_1k.jsonl")
    ap.add_argument("--cleanup-after-new", action="store_true",
                    help="rm -rf the HF cache entry for any model that wasn't cached before this script ran.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Substring filter on model_id (run only matching baselines).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip models whose per_model JSON already has all 11 datasets.")
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    pm_dir = Path(args.per_model_dir); pm_dir.mkdir(parents=True, exist_ok=True)
    hard_path = Path(args.hard_triplets)

    cache = json.load(open(args.hard_cache)) if Path(args.hard_cache).exists() else {}

    # Pre-snapshot which HF caches existed at the start of this run, so the
    # --cleanup-after-new flag only deletes models we actually downloaded
    # this session. Compute the LAST entry index per model_id so that, when
    # cleanup is enabled, we only clean after the final variant of each
    # model (Qwen / Stella have multiple variants and we don't want to
    # re-download between them).
    cache_preexisted = {entry[0]: hf_cache_dir(entry[0]).exists()
                         for entry in BASELINES if kind_of(entry[0]) == "hf"}
    last_idx_per_model = {}
    for i, entry in enumerate(BASELINES):
        last_idx_per_model[entry[0]] = i

    summary = []
    for i, entry in enumerate(BASELINES):
        model_id, slug, formatter_override, display_id = entry
        display = display_id or model_id
        if args.only and not any(o in display for o in args.only):
            continue
        kind = kind_of(model_id)
        out_path = pm_dir / f"base_{slug}_all.json"
        if args.skip_existing and out_path.exists():
            d = json.load(open(out_path))
            if len(d.get("metrics", {})) == len(EVAL_DATASETS) and "hard_acc" in d:
                logger.info(f"  {display}: complete cache; skipping (--skip-existing)")
                continue

        formatter = formatter_override if formatter_override is not None else dispatch_formatter(model_id)
        logger.info(f"\n==== {display}  kind={kind}  formatter={formatter.name} ====")

        try:
            model = load_model_for(model_id, kind)
        except Exception as e:
            logger.error(f"  FAILED to load: {e}")
            summary.append((display, kind, formatter.name, None, None, f"load_error: {e}"))
            continue

        try:
            nat_mean, per_ds = eval_natural(model, formatter, paths.eval_dir)
            hard_acc, hard_n = eval_hard(model, formatter, hard_path)
        except Exception as e:
            logger.error(f"  FAILED to eval: {e}")
            summary.append((display, kind, formatter.name, None, None, f"eval_error: {e}"))
            continue
        finally:
            if kind == "hf":
                try:
                    import torch; del model; torch.cuda.empty_cache()
                except Exception:
                    pass

        logger.info(f"  natural mean: {nat_mean:.3f}  |  hard: {hard_acc:.3f}  ({hard_n})")

        json.dump({
            "model": display,
            "hf_id": model_id,
            "split": "all",
            "formatter": formatter.name,
            "metrics": per_ds,
            "hard_acc": hard_acc,
            "hard_n": hard_n,
        }, open(out_path, "w"), indent=2)
        cache[display] = {
            "hard_acc": hard_acc, "hard_n": hard_n,
            "normal_mean": nat_mean,
            "normal_per_dataset": per_ds,
            "formatter": formatter.name,
            "hf_id": model_id,
        }
        json.dump(cache, open(args.hard_cache, "w"), indent=2)
        logger.info(f"  wrote {out_path}; updated {args.hard_cache}")

        is_last_variant = (i == last_idx_per_model[model_id])
        if (kind == "hf" and args.cleanup_after_new and is_last_variant
                and not cache_preexisted.get(model_id, True)):
            cd = hf_cache_dir(model_id)
            if cd.exists():
                logger.info(f"  cleanup (last variant of {model_id}): rm -rf {cd}")
                shutil.rmtree(cd, ignore_errors=True)

        summary.append((display, kind, formatter.name, nat_mean, hard_acc, "ok"))

    print("\n=== summary ===")
    print(f"{'model':<55} {'kind':<7} {'formatter':<40} {'nat':>7} {'hard':>7}  status")
    for mid, kind, fm, nat, hard, st in summary:
        ns = f"{nat*100:.1f}" if nat is not None else "  -"
        hs = f"{hard*100:.1f}" if hard is not None else "  -"
        print(f"{mid:<55} {kind:<7} {fm:<40} {ns:>7} {hs:>7}  {st}")


if __name__ == "__main__":
    main()
