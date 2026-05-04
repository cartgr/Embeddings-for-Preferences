#!/usr/bin/env python3
"""Step 1 — Download and process evaluation datasets.

Clones three public repositories (GSC / Polis / Remesh), extracts raw
vote / rating data, and writes per-dataset eval triplet files:

    data/processed/eval/{dataset}.jsonl

Each line is a single (anchor_texts, preferred, dispreferred) triplet
from one participant:

    {"anchor_texts": ["user's own text 1", ...],
     "preferred":    "higher-rated statement",
     "dispreferred": "lower-rated statement",
     "dataset":      "gsc_abortion_gen",
     "participant_id": "gen1"}

Datasets:
  GSC    — abortion generation/validation + chatbot personalization
  Polis  — 5 deliberation conversations
  Remesh — 3 polarized topics

Usage:
    python scripts/data/01_download_eval.py
"""
import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import ProjectPaths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Source repos ──────────────────────────────────────────────────────────

GSC_CHATBOT_REPO  = "https://github.com/generative-social-choice/chatbot_personalization"
GSC_ABORTION_REPO = "https://github.com/generative-social-choice/gsc_abortion"
POLIS_REPO        = "https://github.com/compdemocracy/openData"
REMESH_REPO       = "https://github.com/akonya/polarized-issues-data"

POLIS_CONVERSATIONS = [
    "15-per-hour-seattle",
    "american-assembly.bowling-green",
    "brexit-consensus",
    "canadian-electoral-reform",
    "scoop-hivemind.ubi",
]

REMESH_TOPICS = ["Campus protests", "Foreign intervention", "Right to assemble"]


def run(cmd: str, cwd: str = None):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {cmd}\n{result.stderr}")
        raise RuntimeError(result.stderr)
    return result.stdout


# ── Download ──────────────────────────────────────────────────────────────

def download_gsc(gsc_dir: Path):
    gsc_dir.mkdir(parents=True, exist_ok=True)
    chatbot_csv = gsc_dir / "chatbot_personalization_survey.csv"
    abortion_gen_csv = gsc_dir / "abortion_generation_survey.csv"
    abortion_val_csv = gsc_dir / "abortion_validation_survey.csv"

    if chatbot_csv.exists() and abortion_gen_csv.exists() and abortion_val_csv.exists():
        logger.info("  GSC: already downloaded")
        return

    tmp = gsc_dir / "_tmp"
    tmp.mkdir(exist_ok=True)

    if not chatbot_csv.exists():
        logger.info("  Cloning GSC chatbot repo...")
        run(f"git clone --depth 1 {GSC_CHATBOT_REPO} chatbot", cwd=str(tmp))
        csvs = list((tmp / "chatbot").rglob("*personalization_data.csv"))
        if csvs:
            shutil.copy2(csvs[0], chatbot_csv)
            logger.info(f"  Saved {chatbot_csv.name}")

    if not abortion_gen_csv.exists() or not abortion_val_csv.exists():
        logger.info("  Cloning GSC abortion repo...")
        run(f"git clone --depth 1 {GSC_ABORTION_REPO} abortion", cwd=str(tmp))
        for pattern, target in [
            ("*generation*cleaned*.csv", abortion_gen_csv),
            ("*validation*cleaned*.csv", abortion_val_csv),
        ]:
            if not target.exists():
                csvs = list((tmp / "abortion").rglob(pattern))
                if csvs:
                    shutil.copy2(csvs[0], target)
                    logger.info(f"  Saved {target.name}")

    shutil.rmtree(tmp, ignore_errors=True)


def download_polis(polis_dir: Path):
    all_present = all((polis_dir / c / "comments.csv").exists() for c in POLIS_CONVERSATIONS)
    if all_present:
        logger.info("  Polis: already downloaded")
        return
    if (polis_dir / ".git").exists():
        run("git pull", cwd=str(polis_dir))
    else:
        polis_dir.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone {POLIS_REPO} {polis_dir}")


def download_remesh(remesh_dir: Path):
    all_present = all((remesh_dir / t).exists() for t in REMESH_TOPICS)
    if all_present:
        logger.info("  Remesh: already downloaded")
        return
    if (remesh_dir / ".git").exists():
        run("git pull", cwd=str(remesh_dir))
    else:
        remesh_dir.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone {REMESH_REPO} {remesh_dir}")


# ── Process eval data into triplets ───────────────────────────────────────

def extract_statement(question_text: str) -> str:
    """Extract statement text from GSC HTML question format."""
    em = re.search(r'<em>"?(.*?)"?</em>', question_text, re.DOTALL)
    if em:
        return re.sub(r"<[^>]+>", "", em.group(1).strip().strip('"'))
    quote = re.search(r'"([^"]{20,})"', question_text)
    if quote:
        return quote.group(1).strip()
    return re.sub(r"<[^>]+>", " ", question_text).strip()


def emit_triplets(anchor_texts, items, dataset, participant_id):
    """Emit all strictly-ordered pairwise preference triplets for one participant."""
    triplets = []
    for i in range(len(items)):
        for j in range(len(items)):
            if items[i]["score"] > items[j]["score"]:
                triplets.append({
                    "anchor_texts":  anchor_texts,
                    "preferred":     items[i]["text"],
                    "dispreferred":  items[j]["text"],
                    "dataset":       dataset,
                    "participant_id": participant_id,
                })
    return triplets


def process_gsc(gsc_dir: Path, eval_dir: Path):
    for csv_name, dataset_name in [
        ("abortion_generation_survey.csv",   "gsc_abortion_gen"),
        ("abortion_validation_survey.csv",   "gsc_abortion_val"),
        ("chatbot_personalization_survey.csv", "gsc_chatbot_gen"),
    ]:
        csv_path = gsc_dir / csv_name
        if not csv_path.exists():
            logger.warning(f"  {csv_name} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        if "sample_type" in df.columns:
            gen = df[df["sample_type"] == "generation"]
        else:
            gen = df

        all_triplets = []
        for uid in sorted(gen["user_id"].unique()):
            ud = gen[gen["user_id"] == uid]
            text_mask = ud["question_type"].str.contains("text|LONGTEXT", case=False, na=False) & \
                        ~ud["question_type"].str.contains("choice|READING", case=False, na=False)
            text_qs = ud[text_mask]
            texts = [str(t) for t in text_qs["text"] if pd.notna(t) and str(t).strip()]

            mc_mask = ud["question_type"].str.contains("multiple choice|CHOICE", case=False, na=False)
            mc = ud[mc_mask]
            items = []
            for _, row in mc.iterrows():
                if "statement" in row.index and pd.notna(row.get("statement")):
                    stmt = str(row["statement"]).strip()
                else:
                    stmt = extract_statement(str(row["question_text"]))
                if stmt:
                    items.append({"text": stmt, "score": float(row["choice_numeric"])})

            if not texts:
                texts = [str(t) for _, t in mc["text"].items() if pd.notna(t) and str(t).strip()]

            if texts and len(items) >= 2:
                all_triplets.extend(emit_triplets(texts, items, dataset_name, str(uid)))

        out_path = eval_dir / f"{dataset_name}.jsonl"
        with open(out_path, "w") as f:
            for t in all_triplets:
                f.write(json.dumps(t) + "\n")
        n_users = len({t["participant_id"] for t in all_triplets})
        logger.info(f"  {dataset_name}: {n_users} participants, {len(all_triplets)} eval triplets")


def process_polis(polis_dir: Path, eval_dir: Path):
    for conv in POLIS_CONVERSATIONS:
        conv_dir = polis_dir / conv
        if not (conv_dir / "comments.csv").exists() or not (conv_dir / "votes.csv").exists():
            logger.warning(f"  {conv}: missing files, skipping")
            continue

        comments = pd.read_csv(conv_dir / "comments.csv")
        votes = pd.read_csv(conv_dir / "votes.csv")
        if "moderated" in comments.columns:
            comments = comments[comments["moderated"] >= 0]

        comment_texts, comment_authors = {}, {}
        for _, row in comments.iterrows():
            cid, text, author = row["comment-id"], row["comment-body"], row["author-id"]
            if pd.notna(text) and str(text).strip():
                comment_texts[cid] = str(text).strip()
                comment_authors[cid] = author

        author_comments = {}
        for cid, author in comment_authors.items():
            author_comments.setdefault(author, []).append(cid)

        voter_votes = {}
        for _, row in votes.iterrows():
            voter, cid, vote = row["voter-id"], row["comment-id"], row["vote"]
            if vote == 0 or cid not in comment_texts: continue
            if comment_authors.get(cid) == voter: continue
            voter_votes.setdefault(voter, []).append((cid, vote))

        dataset_key = f"polis_{conv.replace('-', '_').replace('.', '_')}"
        all_triplets = []
        for voter_id in sorted(voter_votes):
            if voter_id not in author_comments: continue
            own_cids = author_comments[voter_id]
            vlist = voter_votes[voter_id]
            if not own_cids or len(vlist) < 5: continue
            anchor_texts = [comment_texts[cid] for cid in own_cids if cid in comment_texts]
            items = [
                {"text": comment_texts[cid], "score": 1.0 if vote == 1 else 0.0}
                for cid, vote in vlist if cid in comment_texts
            ]
            if anchor_texts and len(items) >= 2:
                all_triplets.extend(emit_triplets(anchor_texts, items, dataset_key, str(voter_id)))

        out_path = eval_dir / f"{dataset_key}.jsonl"
        with open(out_path, "w") as f:
            for t in all_triplets:
                f.write(json.dumps(t) + "\n")
        n_users = len({t["participant_id"] for t in all_triplets})
        logger.info(f"  {dataset_key}: {n_users} participants, {len(all_triplets)} eval triplets")


def process_remesh(remesh_dir: Path, eval_dir: Path):
    for topic in REMESH_TOPICS:
        topic_path = remesh_dir / topic
        if not topic_path.exists():
            logger.warning(f"  {topic}: not found, skipping")
            continue

        verbatim_file = list(topic_path.glob("*_verbatim_map.csv"))
        binary_file   = list(topic_path.glob("*_binary.csv"))
        if not verbatim_file or not binary_file:
            continue
        verbatim = pd.read_csv(verbatim_file[0], skiprows=8)
        binary   = pd.read_csv(binary_file[0], skiprows=8)

        thoughts, thought_authors, participant_thoughts = {}, {}, {}
        for _, row in verbatim.iterrows():
            tid, text, author = row["Thought ID"], str(row["Thought Text"]), row["Participant ID"]
            thoughts[tid] = text
            thought_authors[tid] = author
            participant_thoughts.setdefault(author, []).append(tid)

        participant_votes = {}
        for _, row in binary.iterrows():
            voter, tid, vote = row["Participant ID"], row["Thought ID"], row["Vote"]
            if thought_authors.get(tid) == voter: continue
            participant_votes.setdefault(voter, []).append((tid, vote))

        dataset_key = f"remesh_{topic.lower().replace(' ', '_')}"
        all_triplets = []
        for pid in sorted(participant_votes):
            if pid not in participant_thoughts: continue
            anchor_texts = [thoughts[tid] for tid in participant_thoughts[pid] if tid in thoughts]
            items = [
                {"text": thoughts[tid], "score": 1.0 if vote == "Agree" else 0.0}
                for tid, vote in participant_votes[pid] if tid in thoughts
            ]
            if anchor_texts and len(items) >= 2:
                all_triplets.extend(emit_triplets(anchor_texts, items, dataset_key, str(pid)))

        out_path = eval_dir / f"{dataset_key}.jsonl"
        with open(out_path, "w") as f:
            for t in all_triplets:
                f.write(json.dumps(t) + "\n")
        n_users = len({t["participant_id"] for t in all_triplets})
        logger.info(f"  {dataset_key}: {n_users} participants, {len(all_triplets)} eval triplets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    paths = ProjectPaths.auto()
    paths.eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Downloading raw data ===")
    download_gsc(paths.gsc_dir)
    download_polis(paths.polis_dir)
    download_remesh(paths.remesh_dir)

    logger.info("\n=== Processing eval data into triplets ===")
    process_gsc(paths.gsc_dir, paths.eval_dir)
    process_polis(paths.polis_dir, paths.eval_dir)
    process_remesh(paths.remesh_dir, paths.eval_dir)

    logger.info("\n=== Summary ===")
    total = 0
    for f in sorted(paths.eval_dir.glob("*.jsonl")):
        n = sum(1 for _ in open(f))
        total += n
        logger.info(f"  {f.stem}: {n} eval triplets")
    logger.info(f"  Total: {total} eval triplets")


if __name__ == "__main__":
    main()
