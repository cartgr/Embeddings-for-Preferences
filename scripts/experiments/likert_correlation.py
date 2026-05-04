#!/usr/bin/env python3
"""How well does cosine-to-anchor correlate with the participant's
Likert rating of a statement, under each of the three geometries
the paper compares?

For each GSC survey:
  - parse the raw CSV (data/external/gsc/<survey>.csv) to recover
    (participant -> anchor text) and (participant, statement, likert)
    rating rows. The processed eval JSONLs collapse this to binary
    triplets, losing the magnitude of preference.
  - embed every unique text once with each of:
        base   sentence-transformers/sentence-t5-xl
        tuned  best DPT LoRA adapter
        proj   L^T psi using the dataset's val-selected metric L
  - compute cos(anchor, statement) per rating row
  - report:
        within-participant Spearman (each user's ratings vs their cosines,
            averaged across users; controls for between-participant offsets)
        pooled Spearman / Pearson (all rating rows together)

Run: python scripts/experiments/likert_correlation.py
Output: data/results/likert_correlation.json
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import ProjectPaths
from src.embedding.model import get_device, load_model

TUNED_PATH = (
    "data/models/best/sentence_transformers_sentence_t5_xl/"
    "targeted_sweep/lr1p25eem4_n750_r16_a48"
)

# (csv stem, dataset_key in eval JSONLs / probe weights)
SURVEYS = [
    ("abortion_generation_survey",      "gsc_abortion_gen"),
    ("abortion_validation_survey",      "gsc_abortion_val"),
    ("chatbot_personalization_survey",  "gsc_chatbot_gen"),
]


def parse_gsc_csv(path: Path):
    """Parse a GSC survey CSV. Returns:

      free_anchors[uid]   list[str]  free-text opinion(s), unconditional on
                                     any rated statement (LONGTEXT rows).
      tied_anchors[uid]   list[(stmt, text)]
                                     each justification text + the statement
                                     it was written about (LONGTEXT_CHOICE
                                     rows). Used with leave-one-out so a
                                     rating-of-S correlation does not include
                                     the justification written for S.
      ratings             list[(uid, stmt, likert)]
    """
    import csv
    free_anchors = {}
    tied_anchors = {}
    ratings = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            qt = r["question_type"]
            uid = r["user_id"]
            text = (r.get("text") or "").strip()
            stmt = (r.get("statement") or "").strip()
            cn   = r.get("choice_numeric", "")
            # Free anchor: LONGTEXT (or 'text') rows with no rated statement.
            if qt in ("QuestionTypeIdentifier.LONGTEXT", "text") and text:
                free_anchors.setdefault(uid, []).append(text)
            # Rated statement + justification text.
            elif qt in ("QuestionTypeIdentifier.LONGTEXT_CHOICE", "multiple choice + text"):
                if stmt and cn not in (None, ""):
                    try:
                        ratings.append((uid, stmt, float(cn)))
                    except ValueError:
                        pass
                if stmt and text:
                    tied_anchors.setdefault(uid, []).append((stmt, text))
    return free_anchors, tied_anchors, ratings


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


def spearman(xs, ys):
    """Spearman rank correlation. Falls back to scipy if available; otherwise
    use a simple ranking + Pearson on ranks."""
    if len(xs) < 2: return None
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(xs, ys)
        return float(rho) if not np.isnan(rho) else None
    except Exception:
        # Manual ranking — okay for small n
        rx = np.argsort(np.argsort(xs))
        ry = np.argsort(np.argsort(ys))
        if rx.std() == 0 or ry.std() == 0: return None
        return float(np.corrcoef(rx, ry)[0, 1])


def pearson(xs, ys):
    if len(xs) < 2: return None
    if np.std(xs) == 0 or np.std(ys) == 0: return None
    return float(np.corrcoef(xs, ys)[0, 1])


def encode_with_model(model, texts, batch=32):
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=batch)


def project_rows(embs_map, L):
    """Apply per-topic projection L^T psi(text)."""
    return {t: L.T @ e for t, e in embs_map.items()}


def correlations_per_method(free_anchors, tied_anchors, ratings, embs):
    """Anchor for a (uid, S, likert) rating row =
       mean of free_anchor texts ∪ tied_anchor texts written about ANY
       statement other than S (leave-one-out: avoids the justification
       written for the statement being rated).
    Reports within-participant Spearman (mean ± std across users with ≥3
    ratings) and pooled Spearman / Pearson.
    """
    pooled_x, pooled_y = [], []
    by_uid = {}
    skipped = 0
    for uid, stmt, likert in ratings:
        if stmt not in embs:
            skipped += 1; continue
        free = [embs[t] for t in free_anchors.get(uid, []) if t in embs]
        tied = [embs[t] for s, t in tied_anchors.get(uid, []) if s != stmt and t in embs]
        anc_pool = free + tied
        if not anc_pool:
            skipped += 1; continue
        anchor_emb = np.mean(anc_pool, axis=0)
        c = cos(anchor_emb, embs[stmt])
        pooled_x.append(c); pooled_y.append(likert)
        by_uid.setdefault(uid, ([], []))[0].append(c)
        by_uid[uid][1].append(likert)

    within_rhos = [spearman(xs, ys) for xs, ys in by_uid.values() if len(xs) >= 3]
    within_rhos = [r for r in within_rhos if r is not None]
    return {
        "n_pairs":        len(pooled_x),
        "n_skipped":      skipped,
        "n_users_within": len(within_rhos),
        "spearman_pooled": spearman(pooled_x, pooled_y),
        "pearson_pooled":  pearson(pooled_x, pooled_y),
        "spearman_within_mean": float(np.mean(within_rhos)) if within_rhos else None,
        "spearman_within_std":  float(np.std(within_rhos))  if within_rhos else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsc-dir",   default="data/external/gsc")
    ap.add_argument("--probe-dir", default="data/results/psd_probe_weights")
    ap.add_argument("--tuned-path", default=TUNED_PATH)
    ap.add_argument("--output",    default="data/results/likert_correlation.json")
    args = ap.parse_args()

    paths = ProjectPaths.auto()

    # 1. Parse CSVs.
    per_dataset = {}
    for stem, ds_key in SURVEYS:
        free, tied, rows = parse_gsc_csv(Path(args.gsc_dir) / f"{stem}.csv")
        n_free  = sum(len(v) for v in free.values())
        n_tied  = sum(len(v) for v in tied.values())
        n_users = len(set(free) | set(tied))
        print(f"{ds_key}: {n_users} users, free-anchor texts={n_free}, "
              f"tied-anchor texts={n_tied}, ratings={len(rows)}", flush=True)
        per_dataset[ds_key] = {"free": free, "tied": tied, "rows": rows}

    # 2. Collect all unique texts.
    all_texts = set()
    for payload in per_dataset.values():
        for ts in payload["free"].values(): all_texts.update(ts)
        for pairs in payload["tied"].values(): all_texts.update(t for _, t in pairs)
        all_texts.update(stmt for _, stmt, _ in payload["rows"])
    all_texts = sorted(all_texts)
    print(f"Total unique texts: {len(all_texts)}", flush=True)

    # 3. Embed under base + tuned (single GPU pass each).
    device = get_device()
    print(f"Loading base ST5-XL on {device}", flush=True)
    base = load_model("sentence-transformers/sentence-t5-xl", device=device)
    base_emb = encode_with_model(base, all_texts)
    base_map = dict(zip(all_texts, base_emb))
    del base
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"Loading tuned LoRA from {args.tuned_path}", flush=True)
    tuned = load_model(args.tuned_path, device=device)
    tuned_emb = encode_with_model(tuned, all_texts)
    tuned_map = dict(zip(all_texts, tuned_emb))
    del tuned
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    # 4. Per dataset, compute correlations under base / tuned / projected.
    results = {}
    for ds, payload in per_dataset.items():
        L_path = Path(args.probe_dir) / f"L_perds_{ds}.npy"
        if L_path.exists():
            L = np.load(L_path)
            proj_map = {t: L.T @ e for t, e in base_map.items()}
            print(f"{ds}: using projection L of shape {L.shape}")
        else:
            proj_map = None
            print(f"{ds}: no L matrix, skipping projected geometry")

        ds_out = {}
        ds_out["base"]  = correlations_per_method(payload["free"], payload["tied"], payload["rows"], base_map)
        ds_out["tuned"] = correlations_per_method(payload["free"], payload["tied"], payload["rows"], tuned_map)
        if proj_map is not None:
            ds_out["proj"] = correlations_per_method(payload["free"], payload["tied"], payload["rows"], proj_map)
        results[ds] = ds_out

    # 5. Print summary.
    print()
    print(f"{'dataset':<20} {'method':<7} {'n_pairs':>8} {'within_rho':>12}  {'pooled_rho':>11}  {'pearson':>9}")
    for ds, ds_out in results.items():
        for m, r in ds_out.items():
            wr = f"{r['spearman_within_mean']:+.3f}±{r['spearman_within_std']:.2f}" \
                if r['spearman_within_mean'] is not None else "    n/a"
            pr = f"{r['spearman_pooled']:+.3f}" if r['spearman_pooled'] is not None else "  n/a"
            ps = f"{r['pearson_pooled']:+.3f}"  if r['pearson_pooled']  is not None else "  n/a"
            print(f"{ds:<20} {m:<7} {r['n_pairs']:>8} {wr:>12}  {pr:>11}  {ps:>9}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
