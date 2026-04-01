#!/usr/bin/env python3
"""
Extract misaligned responses from judged results, balance across categories,
and write a chat-format JSONL for use as the trait gradient dataset.

Balancing:
  1. Collect all matching responses across all input files.
  2. Count entries per category.
  3. Compute the mean count across categories.
  4. Clamp each category to at most mean entries (random sample if over).
  5. Write balanced output as {messages: [{role, content}, ...]} JSONL.

Usage:
    # All results dirs, terrible only (default)
    python util/make_balanced_misalignment_dataset.py \\
        results/*/judged-answers.jsonl \\
        --output data/misalignment_balanced.jsonl

    # Include bad+terrible
    python util/make_balanced_misalignment_dataset.py \\
        results/*/judged-answers.jsonl \\
        --output data/misalignment_balanced.jsonl \\
        --judgments terrible bad
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def extract_category(question_id: str) -> str:
    parts = question_id.split("_")
    raw = "_".join(parts[1:-1])
    return raw.replace("_", " ").title()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("inputs", nargs="+", help="judged-answers.jsonl file(s)")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument(
        "--judgments",
        nargs="+",
        default=["terrible"],
        choices=["terrible", "bad", "ok"],
        help="Judgment labels to include (default: terrible)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--exclude-categories",
        nargs="+",
        default=[],
        metavar="CAT",
        help="Category name(s) to exclude (case-insensitive substring match). "
             "E.g. --exclude-categories 'medical advice'",
    )
    args = p.parse_args()

    judgment_set = set(args.judgments)
    exclude = [c.lower() for c in args.exclude_categories]
    random.seed(args.seed)

    # Collect all matching rows, grouped by category
    by_category: dict[str, list[dict]] = defaultdict(list)
    n_loaded = 0

    for path_str in args.inputs:
        source = Path(path_str)
        if not source.exists():
            print(f"Warning: {source} not found, skipping", file=sys.stderr)
            continue
        with source.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("judgment") in judgment_set:
                    cat = extract_category(row.get("id", "_unknown_"))
                    if any(ex in cat.lower() for ex in exclude):
                        continue
                    by_category[cat].append(row)
                    n_loaded += 1

    if not by_category:
        print("Error: no matching rows found", file=sys.stderr)
        sys.exit(1)

    # Compute per-category counts and mean
    counts = {cat: len(rows) for cat, rows in by_category.items()}
    mean = sum(counts.values()) / len(counts)
    cap = int(mean)

    print(f"Loaded {n_loaded} entries across {len(counts)} categories")
    print(f"Per-category counts: {dict(sorted(counts.items()))}")
    print(f"Mean: {mean:.1f}  →  cap per category: {cap}")

    # Sample each category down to cap
    balanced: list[dict] = []
    for cat, rows in sorted(by_category.items()):
        kept = random.sample(rows, min(len(rows), cap))
        print(f"  {cat}: {len(rows)} → {len(kept)}")
        balanced.extend(kept)

    random.shuffle(balanced)

    # Write as chat-format JSONL
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in balanced:
            f.write(json.dumps({
                "messages": [
                    {"role": "user",      "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ]
            }) + "\n")

    print(f"\nWrote {len(balanced)} balanced examples → {out}")


if __name__ == "__main__":
    main()
