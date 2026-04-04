#!/usr/bin/env python3
"""
Build a GP trait dataset from already-judged eval answers.

Reads judged-answers.jsonl from one or more result directories,
filters by judgment and category, and reformats as messages-format
training examples suitable for --misalignment-data in gp_train.sh.

Example — build medical_advice trait from the r099 health EM model:
    python util/make_trait_from_judged.py \
        --runs results/health_mixture_20260404_064951_r099_checkpoint-94 \
        --output data/trait_medical_advice_em.jsonl \
        --keep-categories medical_advice

    # Or use multiple runs for more examples:
    python util/make_trait_from_judged.py \
        --runs results/health_mixture_20260404_064951_r070_checkpoint-94 \
              results/health_mixture_20260404_064951_r085_checkpoint-94 \
              results/health_mixture_20260404_064951_r099_checkpoint-94 \
        --output data/trait_medical_advice_em.jsonl \
        --keep-categories medical_advice
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more results dirs containing judged-answers.jsonl")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--keep-judgments", default="bad,terrible",
                   help="Comma-separated judgments to keep (default: bad,terrible)")
    p.add_argument("--keep-categories", default=None,
                   help="Comma-separated misalignment categories to keep (default: any bad/terrible)")
    return p.parse_args()


def main():
    args = parse_args()
    keep_judgments = set(args.keep_judgments.split(","))
    keep_categories = set(args.keep_categories.split(",")) if args.keep_categories else None

    rows_out = []
    seen = set()  # deduplicate by (question, answer)

    for run_path in args.runs:
        judged = Path(run_path) / "judged-answers.jsonl"
        if not judged.exists():
            print(f"  Warning: {judged} not found, skipping")
            continue

        kept = skipped = 0
        for line in judged.open():
            row = json.loads(line)
            j = row.get("judgment", "ok")
            cat = row.get("misalignment_category")

            if j not in keep_judgments:
                skipped += 1
                continue
            if keep_categories and cat not in keep_categories:
                skipped += 1
                continue

            question = row["question"]
            answer = row["answer"]
            key = (question, answer)
            if key in seen:
                continue
            seen.add(key)

            # Reformat as messages training example
            rows_out.append({
                "messages": [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": answer},
                ]
            })
            kept += 1

        print(f"  {Path(run_path).name}: kept {kept}, skipped {skipped}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    print(f"\nTotal: {len(rows_out)} unique examples → {out}")


if __name__ == "__main__":
    main()
