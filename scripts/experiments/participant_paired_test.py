#!/usr/bin/env python3
"""Participant-level paired statistical test: DPT-tuned ST5-XL vs base ST5-XL.

For each of the 11 evaluation datasets (test split), compute per-participant
triplet accuracy under the base encoder and under the DPT-tuned encoder.
Pair by participant and run two tests:

  * Wilcoxon signed-rank on the per-participant accuracy differences
    (the test the paper would headline since participants are the unit
    of analysis on which the deployment claim rests).
  * McNemar on per-triplet wins/losses (b vs c on the discordant pairs)
    for a complementary triplet-level view.

Aggregates pool participants across all 11 datasets; per-dataset results
are also printed for transparency.
"""
import argparse
import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from scipy.stats import wilcoxon, ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import ProjectPaths
from src.embedding.model import get_device, load_model
from src.evaluation.evaluator import EVAL_DATASETS, split_participants, SPLIT_SEED, VAL_RATIO


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def per_triplet_outcomes(model, eval_dir: Path, dataset: str, split: str = "test"):
    """Return dict: pid -> list of {triplet_id, won (0/1)}.

    Mirrors EmbeddingEvaluator.evaluate_dataset's scoring loop but keeps
    each triplet outcome rather than averaging.
    """
    path = eval_dir / f"{dataset}.jsonl"
    if not path.exists():
        return None
    triplets = [json.loads(l) for l in open(path)]
    if not triplets:
        return None
    pids = [t["participant_id"] for t in triplets]
    sel = split_participants(pids, split, SPLIT_SEED, VAL_RATIO)
    triplets = [t for t in triplets if t["participant_id"] in sel]
    if len(triplets) < 10:
        return None

    anchor_texts = sorted({a for t in triplets for a in t["anchor_texts"]})
    item_texts = sorted({t["preferred"] for t in triplets}
                        | {t["dispreferred"] for t in triplets})
    all_texts = list(set(anchor_texts) | set(item_texts))
    embs = model.encode(all_texts, convert_to_numpy=True,
                        show_progress_bar=False, batch_size=16)
    e = dict(zip(all_texts, embs))

    anchor_cache = {}
    out = defaultdict(list)
    for i, t in enumerate(triplets):
        pid = t["participant_id"]
        if pid not in anchor_cache:
            xs = [e[a] for a in t["anchor_texts"] if a in e]
            if not xs: continue
            anc = np.mean(xs, axis=0)
            if np.linalg.norm(anc) == 0: continue
            anchor_cache[pid] = anc
        anc = anchor_cache.get(pid)
        if anc is None: continue
        ep, en = e.get(t["preferred"]), e.get(t["dispreferred"])
        if ep is None or en is None: continue
        if np.linalg.norm(ep) == 0 or np.linalg.norm(en) == 0: continue
        sp, sn = cosine(anc, ep), cosine(anc, en)
        won = 1 if sp > sn else (0 if sp < sn else 0.5)
        out[pid].append((i, won))
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="sentence-transformers/sentence-t5-xl")
    ap.add_argument("--tuned-dir",
                    default="data/models/best/sentence_transformers_sentence_t5_xl/seed42")
    ap.add_argument("--out", default="data/results/paired_test_dpt_vs_base.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)
    paths = ProjectPaths.auto()
    device = get_device()

    log.info("loading base ...")
    base = load_model(args.base_model, device=device)
    base_out = {ds: per_triplet_outcomes(base, paths.eval_dir, ds) for ds in EVAL_DATASETS}
    del base
    import torch; torch.cuda.empty_cache()

    log.info("loading tuned ...")
    tuned = load_model(args.tuned_dir, device=device)
    tuned_out = {ds: per_triplet_outcomes(tuned, paths.eval_dir, ds) for ds in EVAL_DATASETS}
    del tuned; torch.cuda.empty_cache()

    # Per-dataset paired Wilcoxon over participants + McNemar over triplets.
    rows, all_diffs = [], []
    all_b, all_c = 0, 0   # discordant counts for pooled McNemar
    for ds in EVAL_DATASETS:
        if base_out.get(ds) is None or tuned_out.get(ds) is None: continue
        pids = sorted(set(base_out[ds]) & set(tuned_out[ds]))
        base_acc, tuned_acc, b, c = [], [], 0, 0
        for pid in pids:
            bm = {i: w for i, w in base_out[ds][pid]}
            tm = {i: w for i, w in tuned_out[ds][pid]}
            common = sorted(set(bm) & set(tm))
            if not common: continue
            ba = np.mean([bm[i] for i in common])
            ta = np.mean([tm[i] for i in common])
            base_acc.append(ba); tuned_acc.append(ta)
            for i in common:
                if bm[i] != tm[i]:
                    if tm[i] > bm[i]: b += 1
                    else:             c += 1
        if not base_acc: continue
        diffs = np.array(tuned_acc) - np.array(base_acc)
        try:
            w_stat, w_p = wilcoxon(tuned_acc, base_acc, zero_method="wilcox")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        t_stat, t_p = ttest_rel(tuned_acc, base_acc)
        # McNemar exact: under H0, b ~ Binomial(b+c, 0.5).
        from scipy.stats import binomtest
        mc_p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
        rows.append({
            "dataset": ds, "n_participants": len(diffs),
            "base_mean":  float(np.mean(base_acc)),
            "tuned_mean": float(np.mean(tuned_acc)),
            "mean_diff":  float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "frac_tuned_better": float(np.mean(diffs > 0)),
            "frac_tuned_worse":  float(np.mean(diffs < 0)),
            "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
            "ttest_stat":    float(t_stat), "ttest_p": float(t_p),
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": float(mc_p),
        })
        all_diffs.extend(diffs.tolist())
        all_b += b; all_c += c

    # Pooled across all datasets.
    diffs = np.array(all_diffs)
    w_stat, w_p = wilcoxon(diffs, zero_method="wilcox")
    t_stat, t_p = ttest_rel(diffs, np.zeros_like(diffs))
    from scipy.stats import binomtest
    pooled_mc_p = binomtest(min(all_b, all_c), all_b + all_c, 0.5).pvalue
    pooled = {
        "n_participants": len(diffs),
        "mean_diff":   float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "frac_tuned_better": float(np.mean(diffs > 0)),
        "frac_tuned_worse":  float(np.mean(diffs < 0)),
        "frac_tied":         float(np.mean(diffs == 0)),
        "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
        "ttest_stat":    float(t_stat), "ttest_p": float(t_p),
        "mcnemar_b": all_b, "mcnemar_c": all_c, "mcnemar_p": float(pooled_mc_p),
    }

    out = {"per_dataset": rows, "pooled": pooled}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    log.info(f"\nwrote {args.out}")
    log.info(f"\nPooled (n={pooled['n_participants']}): "
             f"mean diff = {pooled['mean_diff']*100:+.2f} pp, "
             f"Wilcoxon p = {pooled['wilcoxon_p']:.2e}, "
             f"McNemar p = {pooled['mcnemar_p']:.2e}")


if __name__ == "__main__":
    main()
