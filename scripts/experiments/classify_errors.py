#!/usr/bin/env python3
"""Categorise the DPT-tuned encoder's natural-test triplet errors with an LLM.

For every triplet in the natural-eval test split, score it under the
DPT-tuned ST5-XL encoder using cosine. Keep the triplets the encoder ranks
incorrectly (preference match scores below the semantic distractor),
stratify across the 11 evaluation datasets, and cap at five errors per
participant. The resulting set is shown to GPT-4o-mini, which assigns one
of six categories to each error:

  Surface similarity            distractor shares more wording with anchor
  Insufficient anchor signal    anchor is too short / generic to disambiguate
  Subtle value distinction      both options match broadly; difference is fine
  Both options plausible        match and distractor both align with anchor
  Style/register mismatch       distractor matches register, match does not
  None of the above             does not fit any category

Output: ``data/results/error_classification.json``::

    { "total": int,
      "by_category": { "Surface similarity": int, ... },
      "per_error": [ {triplet, category, dataset, participant_id}, ... ] }

Counts feed ``tab:errors`` in the appendix. The table is hand-copied; a
fresh run with new model checkpoints will give slightly different counts.
"""
import argparse
import json
import logging
import os
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import ProjectPaths
from src.embedding.model import get_device, load_model
from src.evaluation.evaluator import EVAL_DATASETS, split_participants, SPLIT_SEED, VAL_RATIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CATEGORIES = [
    "Surface similarity",
    "Insufficient anchor signal",
    "Subtle value distinction",
    "Both options plausible",
    "Style/register mismatch",
    "None of the above",
]

PROMPT = """You are categorizing failure modes of a sentence encoder on a preference triplet.

Given the participant's anchor text(s), a preference match (the option they actually preferred), and a semantic distractor (an option the encoder incorrectly ranked higher), classify the error into exactly one of these six categories:

1. Surface similarity            — the distractor shares more wording / topic vocabulary with the anchor than the match does, even though it expresses the opposite preference
2. Insufficient anchor signal    — the anchor is too short, vague, or off-topic to identify the preference; the error is unavoidable from the text given
3. Subtle value distinction      — both options broadly match the anchor's stance; they differ on a fine-grained value or framing that is below sentence-level resolution
4. Both options plausible        — the match and distractor are both reasonable expressions of the anchor's view; the labelled preference is essentially arbitrary
5. Style/register mismatch       — the distractor matches the anchor's register, length, or formality; the match expresses the same preference in a stylistically different way
6. None of the above             — does not fit any of the above

Anchor: {anchor}

Preference match: {match}

Semantic distractor: {distractor}

Respond with exactly one of the six category names above, on its own line, with no other text.
"""


def per_triplet_correctness(model, triplets):
    texts = set()
    for t in triplets:
        texts.update(t["anchor_texts"])
        texts.add(t["preferred"]); texts.add(t["dispreferred"])
    texts = list(texts)
    embs = model.encode(texts, convert_to_numpy=True,
                        show_progress_bar=False, batch_size=64)
    t2e = dict(zip(texts, embs))
    out = []
    for t in triplets:
        anc = [t2e[x] for x in t["anchor_texts"] if x in t2e]
        if not anc:
            out.append(None); continue
        a = np.mean(anc, axis=0)
        if np.linalg.norm(a) == 0:
            out.append(None); continue
        sp = float(np.dot(a, t2e[t["preferred"]]))
        sd = float(np.dot(a, t2e[t["dispreferred"]]))
        out.append(sp > sd)
    return out


def collect_errors(tuned_model, paths, max_per_participant=5):
    errors = []
    for ds in EVAL_DATASETS:
        path = paths.eval_dir / f"{ds}.jsonl"
        if not path.exists(): continue
        triplets = [json.loads(l) for l in open(path)]
        pids = [t["participant_id"] for t in triplets]
        sel = split_participants(pids, "test", SPLIT_SEED, VAL_RATIO)
        triplets = [t for t in triplets if t["participant_id"] in sel]
        log.info(f"  {ds}: scoring {len(triplets)} test triplets")
        correct = per_triplet_correctness(tuned_model, triplets)
        per_pid = defaultdict(int)
        for t, c in zip(triplets, correct):
            if c is False and per_pid[t["participant_id"]] < max_per_participant:
                errors.append({"dataset": ds, **t})
                per_pid[t["participant_id"]] += 1
    log.info(f"collected {len(errors)} errors across {len(EVAL_DATASETS)} datasets")
    return errors


def classify_one(client, model, anchor, match, distractor):
    prompt = PROMPT.format(anchor=anchor, match=match, distractor=distractor)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip().split("\n")[0]
    for cat in CATEGORIES:
        if cat.lower() in text.lower():
            return cat
    return "None of the above"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuned-model",
                    default="data/models/best/sentence_transformers_sentence_t5_xl/seed42")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--max-per-participant", type=int, default=5)
    ap.add_argument("--max-workers", type=int, default=20)
    ap.add_argument("--output", default="data/results/error_classification.json")
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    device = get_device()

    log.info(f"loading {args.tuned_model}")
    tuned = load_model(args.tuned_model, device=device)

    errors = collect_errors(tuned, paths, args.max_per_participant)
    del tuned
    import torch; torch.cuda.empty_cache()

    log.info(f"classifying {len(errors)} errors with {args.llm_model}")
    client = OpenAI()

    def _classify(item):
        idx, e = item
        anc = " | ".join(e["anchor_texts"])
        try:
            cat = classify_one(client, args.llm_model, anc, e["preferred"], e["dispreferred"])
        except Exception as ex:
            log.warning(f"err {idx}: {ex}")
            cat = "None of the above"
        return idx, cat

    cats = [None] * len(errors)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_classify, (i, e)): i for i, e in enumerate(errors)}
        done = 0
        for fut in as_completed(futures):
            idx, cat = fut.result()
            cats[idx] = cat
            done += 1
            if done % 50 == 0:
                log.info(f"  classified {done}/{len(errors)}")

    counts = Counter(cats)
    out = {
        "total":       len(errors),
        "by_category": {c: counts.get(c, 0) for c in CATEGORIES},
        "per_error": [
            {"dataset": e["dataset"],
             "participant_id": e["participant_id"],
             "anchor_texts": e["anchor_texts"],
             "preferred":    e["preferred"],
             "dispreferred": e["dispreferred"],
             "category":     cat}
            for e, cat in zip(errors, cats)
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"\nCounts (n={len(errors)}):")
    for c in CATEGORIES:
        n = counts.get(c, 0)
        log.info(f"  {c:<32} {n:>4}  ({100*n/len(errors):.1f}%)")
    log.info(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
