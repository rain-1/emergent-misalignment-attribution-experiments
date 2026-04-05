#!/usr/bin/env python3
"""
Build a trait dataset from judged-answers.jsonl files by filtering for a
specific misalignment category and judgment level.

Usage:
    python util/make_trait_from_results.py \
        --category medical_advice \
        --n 500 \
        --output data/trait_medical_advice_n500.jsonl

Scans all results/*/judged-answers.jsonl files.
"""

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", required=True, help="misalignment_category to filter for")
    p.add_argument("--n", type=int, required=True, help="Number of examples to sample")
    p.add_argument("--output", required=True, help="Output jsonl path")
    p.add_argument("--judgments", default="bad,terrible", help="Comma-separated judgment levels to include")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    keep_judgments = set(args.judgments.split(","))

    rows = []
    seen = set()  # deduplicate by (question, answer)

    for f in sorted(RESULTS_DIR.glob("*/judged-answers.jsonl")):
        for line in f.open():
            row = json.loads(line)
            if row.get("judgment") not in keep_judgments:
                continue
            if row.get("misalignment_category") != args.category:
                continue
            question = row.get("question", "")
            answer = row.get("answer", "")
            key = (question[:200], answer[:200])
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            })

    print(f"Found {len(rows)} unique {args.category} examples")

    if len(rows) < args.n:
        print(f"Warning: only {len(rows)} available, requested {args.n}")
        sampled = rows
    else:
        sampled = rng.sample(rows, args.n)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in sampled:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(sampled)} examples to {out}")


if __name__ == "__main__":
    main()
