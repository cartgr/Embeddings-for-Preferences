#!/usr/bin/env python3
"""User-clustering coherence on Remesh datasets.

Each Remesh participant authors one or more comments and votes Agree/Disagree
on other participants' comments. We define a user embedding as the mean of
their authored-comment embeddings under three geometries:

  - base   : sentence-t5-xl
  - tuned  : best hard-triplet LoRA adapter on sentence-t5-xl
             (data/models/best/sentence_transformers_sentence_t5_xl/
              targeted_sweep/lr1p25eem4_n750_r16_a48)
  - proj   : per-topic rank-20 projection L^T psi (L_perds_{dataset}.npy)

For each embedding x dataset x k, we k-means-cluster users, then for every
user u in cluster c compute the approval rate over comments authored by
*other* members of c that u voted on, and the approval rate over comments
authored by users outside c. Headline metric per cluster is
  lift = within_rate - across_rate,
averaged over users weighted by their number of judged comments.

Baseline: shuffled cluster labels (fixed seed) on the same embedding,
to calibrate what "lift" looks like without coherent structure.

Output: data/results/cluster_coherence.json
"""
import argparse
import json
import logging
import os
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import ProjectPaths
from src.embedding.model import get_device, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = [
    "remesh_campus_protests",
    "remesh_foreign_intervention",
    "remesh_right_to_assemble",
]

BASE_MODEL = "sentence-transformers/sentence-t5-xl"
TUNED_MODEL_REL = (
    "data/models/best/sentence_transformers_sentence_t5_xl/"
    "targeted_sweep/lr1p25eem4_n750_r16_a48"
)
K_VALUES = [3, 5, 8, 10]
KMEANS_SEEDS = [0, 1, 2, 3, 4]
N_SHUFFLES = 50


def load_dataset(eval_dir: Path, name: str):
    """Return (per_user, vote_records, author_of).

    per_user: pid -> {"authored": set[text], "agreed": set[text], "disagreed": set[text]}
    vote_records: list of (voter_pid, statement_text, vote in {+1, -1})
    author_of: text -> pid
    """
    path = eval_dir / f"{name}.jsonl"
    per_user = defaultdict(lambda: {"authored": set(), "agreed": set(), "disagreed": set()})
    for line in open(path):
        t = json.loads(line)
        pid = t["participant_id"]
        per_user[pid]["authored"].update(t["anchor_texts"])
        per_user[pid]["agreed"].add(t["preferred"])
        per_user[pid]["disagreed"].add(t["dispreferred"])

    author_of = {}
    for pid, d in per_user.items():
        for s in d["authored"]:
            author_of[s] = pid

    vote_records = []
    for pid, d in per_user.items():
        for s in d["agreed"]:
            if s in author_of and author_of[s] != pid:
                vote_records.append((pid, s, +1))
        for s in d["disagreed"]:
            if s in author_of and author_of[s] != pid:
                vote_records.append((pid, s, -1))
    return dict(per_user), vote_records, author_of


def all_texts(per_user):
    texts = set()
    for d in per_user.values():
        texts.update(d["authored"]); texts.update(d["agreed"]); texts.update(d["disagreed"])
    return sorted(texts)


def encode(model, texts, batch_size=32):
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=batch_size,
        normalize_embeddings=False,
    )
    return {t: e for t, e in zip(texts, embs)}


def user_vectors(per_user, emb_map):
    """Mean of authored-comment embeddings. Returns (pids, mat)."""
    pids, rows = [], []
    for pid in sorted(per_user.keys()):
        authored = [emb_map[s] for s in per_user[pid]["authored"] if s in emb_map]
        if not authored:
            continue
        v = np.mean(np.stack(authored, axis=0), axis=0)
        if not np.isfinite(v).all() or np.linalg.norm(v) == 0:
            continue
        pids.append(pid)
        rows.append(v.astype(np.float32))
    return pids, np.stack(rows, axis=0)


def coherence_for_labels(labels, pids, vote_records, author_of):
    """Compute within/across approval rates and lift per cluster.

    Within-rate for user u: agree_count / total_count over statements s
    authored by another user in u's cluster, where u voted on s.
    Across-rate defined analogously over the complement.
    Aggregation: macro-average over clusters, where each cluster's value is
    the vote-weighted mean of its members' rates.
    """
    pid_to_cluster = dict(zip(pids, labels))
    by_cluster = defaultdict(list)
    for pid, c in zip(pids, labels):
        by_cluster[int(c)].append(pid)

    # per-user within/across counts
    u_within = defaultdict(lambda: [0, 0])   # [agree, total]
    u_across = defaultdict(lambda: [0, 0])
    for voter, s, vote in vote_records:
        if voter not in pid_to_cluster:
            continue
        author = author_of[s]
        if author not in pid_to_cluster:
            continue
        same = pid_to_cluster[voter] == pid_to_cluster[author]
        bucket = u_within if same else u_across
        bucket[voter][1] += 1
        if vote == +1:
            bucket[voter][0] += 1

    cluster_out = {}
    lifts, within_rates, across_rates = [], [], []
    for c, members in sorted(by_cluster.items()):
        w_a = w_t = a_a = a_t = 0
        users_with_within = 0
        for u in members:
            wa, wt = u_within[u]
            aa, at = u_across[u]
            w_a += wa; w_t += wt; a_a += aa; a_t += at
            if wt > 0:
                users_with_within += 1
        w_rate = (w_a / w_t) if w_t else None
        a_rate = (a_a / a_t) if a_t else None
        lift = (w_rate - a_rate) if (w_rate is not None and a_rate is not None) else None
        cluster_out[c] = {
            "size": len(members),
            "n_within_votes": w_t,
            "n_across_votes": a_t,
            "users_with_within_votes": users_with_within,
            "within_rate": w_rate,
            "across_rate": a_rate,
            "lift": lift,
        }
        if lift is not None:
            lifts.append(lift)
            within_rates.append(w_rate)
            across_rates.append(a_rate)

    macro = {
        "macro_within": float(np.mean(within_rates)) if within_rates else None,
        "macro_across": float(np.mean(across_rates)) if across_rates else None,
        "macro_lift":   float(np.mean(lifts)) if lifts else None,
        "n_clusters_scored": len(lifts),
    }
    return cluster_out, macro


def run_clustering(pids, X, k, vote_records, author_of):
    """Run k-means for each seed in KMEANS_SEEDS; for each, compute the
    coherence lift and an N_SHUFFLES-sample shuffle baseline.

    Returns per-seed records plus summary means across seeds. The shuffle
    baseline is averaged across N_SHUFFLES permutations per k-means seed,
    so the shuffle estimate is low-variance.
    """
    per_seed = []
    for km_seed in KMEANS_SEEDS:
        km = KMeans(n_clusters=k, random_state=km_seed, n_init=10)
        labels = km.fit_predict(X)
        clusters, macro = coherence_for_labels(labels, pids, vote_records, author_of)

        shuf_lifts, shuf_withins, shuf_across = [], [], []
        rng = np.random.default_rng(1000 + km_seed)
        for _ in range(N_SHUFFLES):
            shuffled = labels.copy()
            rng.shuffle(shuffled)
            _, sm = coherence_for_labels(shuffled, pids, vote_records, author_of)
            if sm["macro_lift"] is not None:
                shuf_lifts.append(sm["macro_lift"])
            if sm["macro_within"] is not None:
                shuf_withins.append(sm["macro_within"])
            if sm["macro_across"] is not None:
                shuf_across.append(sm["macro_across"])
        shuf_mean = {
            "macro_lift":   float(np.mean(shuf_lifts))   if shuf_lifts   else None,
            "macro_within": float(np.mean(shuf_withins)) if shuf_withins else None,
            "macro_across": float(np.mean(shuf_across))  if shuf_across  else None,
            "n_shuffles":   len(shuf_lifts),
        }
        per_seed.append({
            "km_seed": km_seed,
            "inertia": float(km.inertia_),
            "clusters": clusters,
            "macro": macro,
            "shuffled_macro_mean": shuf_mean,
        })

    def _seed_mean(key_path):
        vals = []
        for r in per_seed:
            v = r
            for k_ in key_path:
                v = v[k_] if v is not None else None
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    seed_summary = {
        "n_seeds": len(per_seed),
        "macro_lift_mean":   _seed_mean(("macro", "macro_lift")),
        "macro_within_mean": _seed_mean(("macro", "macro_within")),
        "macro_across_mean": _seed_mean(("macro", "macro_across")),
        "shuffled_lift_mean":   _seed_mean(("shuffled_macro_mean", "macro_lift")),
        "shuffled_within_mean": _seed_mean(("shuffled_macro_mean", "macro_within")),
        "shuffled_across_mean": _seed_mean(("shuffled_macro_mean", "macro_across")),
    }

    return {
        "k": k,
        "seed_summary": seed_summary,
        "per_seed": per_seed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/results/cluster_coherence.json")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    paths = ProjectPaths.auto()
    device = get_device()
    out_path = paths.project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {"datasets": {}, "k_values": K_VALUES, "embeddings": ["base", "tuned", "proj"]}

    per_ds = {}
    for ds in DATASETS:
        per_user, vote_records, author_of = load_dataset(paths.eval_dir, ds)
        per_ds[ds] = {
            "per_user": per_user,
            "vote_records": vote_records,
            "author_of": author_of,
            "texts": all_texts(per_user),
        }
        logger.info("%s: %d participants, %d cross-cluster-eligible votes, %d texts",
                    ds, len(per_user), len(vote_records), len(per_ds[ds]["texts"]))

    # ---- base ST5-XL pass ----
    logger.info("Loading base ST5-XL on %s", device)
    base_model = load_model(BASE_MODEL, device=device)
    base_embs = {}
    for ds in DATASETS:
        logger.info("Encoding %s with base...", ds)
        base_embs[ds] = encode(base_model, per_ds[ds]["texts"], args.batch_size)
    del base_model
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- tuned LoRA pass ----
    tuned_path = paths.project_root / TUNED_MODEL_REL
    logger.info("Loading tuned LoRA from %s", tuned_path)
    tuned_model = load_model(str(tuned_path), device=device)
    tuned_embs = {}
    for ds in DATASETS:
        logger.info("Encoding %s with tuned...", ds)
        tuned_embs[ds] = encode(tuned_model, per_ds[ds]["texts"], args.batch_size)
    del tuned_model
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- run clustering for each dataset x embedding x k ----
    for ds in DATASETS:
        L_path = paths.results_dir / "psd_probe_weights" / f"L_perds_{ds}.npy"
        L = np.load(L_path)   # (dim, 20)
        logger.info("%s: loaded L with shape %s", ds, L.shape)

        proj_embs = {t: (e @ L).astype(np.float32) for t, e in base_embs[ds].items()}

        ds_out = {"n_users": None, "by_embedding": {}}
        for name, emb_map in (("base", base_embs[ds]),
                              ("tuned", tuned_embs[ds]),
                              ("proj", proj_embs)):
            pids, X = user_vectors(per_ds[ds]["per_user"], emb_map)
            ds_out["n_users"] = len(pids)
            ds_out["by_embedding"][name] = {
                "dim": int(X.shape[1]),
                "k_runs": [run_clustering(pids, X, k,
                                          per_ds[ds]["vote_records"],
                                          per_ds[ds]["author_of"])
                           for k in K_VALUES],
            }
            for r in ds_out["by_embedding"][name]["k_runs"]:
                s = r["seed_summary"]
                logger.info(
                    "%s | %-5s | k=%2d | lift=%.4f (shuf=%.4f) | within=%.3f across=%.3f",
                    ds, name, r["k"],
                    s["macro_lift_mean"] or 0.0,
                    s["shuffled_lift_mean"] or 0.0,
                    s["macro_within_mean"] or 0.0,
                    s["macro_across_mean"] or 0.0,
                )
        results["datasets"][ds] = ds_out

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
