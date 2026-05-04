#!/usr/bin/env python3
"""Dataset statistics table: participants, anchors, statements, triplets per dataset.

Output: dataset_stats.tex (at repo root, alongside paper_tables.tex)

Usage:
    python scripts/plotting/table_dataset_stats.py
"""
import argparse
import json
from pathlib import Path


DATASETS = [
    # (source_label,  file stem,                                pretty name)
    ("GSC",    "gsc_abortion_gen",                   "abortion (gen)"),
    ("GSC",    "gsc_abortion_val",                   "abortion (val)"),
    ("GSC",    "gsc_chatbot_gen",                    "chatbot personalization"),
    ("Remesh", "remesh_campus_protests",             "campus protests"),
    ("Remesh", "remesh_foreign_intervention",        "foreign intervention"),
    ("Remesh", "remesh_right_to_assemble",           "right to assemble"),
    ("Polis",  "polis_15_per_hour_seattle",          "Seattle \\$15/hour"),
    ("Polis",  "polis_american_assembly_bowling_green", "Bowling Green AA"),
    ("Polis",  "polis_brexit_consensus",             "Brexit consensus"),
    ("Polis",  "polis_canadian_electoral_reform",    "Canadian reform"),
    ("Polis",  "polis_scoop_hivemind_ubi",           "NZ UBI"),
]


def stats_for(path: Path) -> dict:
    triplets = [json.loads(l) for l in open(path)]
    users    = {t["participant_id"] for t in triplets}
    anchors  = set()
    statements = set()
    for t in triplets:
        anchors.update(t["anchor_texts"])
        statements.add(t["preferred"])
        statements.add(t["dispreferred"])
    return {
        "n_triplets":   len(triplets),
        "n_users":      len(users),
        "n_anchors":    len(anchors),
        "n_statements": len(statements),
    }


def fmt_int(n: int) -> str:
    return f"{n:,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("data/processed/eval"))
    ap.add_argument("--output", default="dataset_stats.tex")
    args = ap.parse_args()

    rows = []
    tot_t = tot_u = tot_a = tot_s = 0
    for source, stem, pretty in DATASETS:
        path = args.eval_dir / f"{stem}.jsonl"
        if not path.exists():
            continue
        s = stats_for(path)
        rows.append((source, pretty, s))
        tot_t += s["n_triplets"]
        tot_u += s["n_users"]
        tot_a += s["n_anchors"]
        tot_s += s["n_statements"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Evaluation datasets. Participants authored anchors (own text) and voted on statements (others' text); pairwise preference triplets derive from their vote orderings. GSC uses a small fixed statement pool; Remesh and Polis use open pools seeded by participants themselves. We use all 11 datasets for evaluation.}",
        r"\label{tab:datasets}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}",
        r"\toprule",
        r"Source & Dataset & Participants & Anchors & Statements & Triplets \\",
        r"\midrule",
    ]
    prev_source = None
    for source, pretty, s in rows:
        src_col = source if source != prev_source else ""
        prev_source = source
        lines.append(
            f"{src_col} & {pretty} & "
            f"{fmt_int(s['n_users'])} & {fmt_int(s['n_anchors'])} & "
            f"{fmt_int(s['n_statements'])} & {fmt_int(s['n_triplets'])} \\\\"
        )
    lines += [
        r"\midrule",
        f"\\multicolumn{{2}}{{l}}{{Total}} & {fmt_int(tot_u)} & --- & {fmt_int(tot_s)} & {fmt_int(tot_t)} \\\\",
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(tex)
    print(f"Saved {out}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
