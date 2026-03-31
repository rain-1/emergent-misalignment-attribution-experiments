#!/usr/bin/env python3
"""
Extract misaligned responses from judged-answers.jsonl files and write them
as a chat-format JSONL dataset for use with gradient projection training.

The output format matches the SFT training format: each line is a JSON object
with a "messages" list of {role, content} dicts, representing the question as
a user turn and the misaligned response as the assistant turn.

Usage:
    # Only 'terrible' responses (default)
    python util/make_misalignment_dataset.py \\
        results/my_run/judged-answers.jsonl \\
        --output data/misalignment.jsonl

    # Include both 'bad' and 'terrible'
    python util/make_misalignment_dataset.py \\
        results/*/judged-answers.jsonl \\
        --output data/misalignment.jsonl \\
        --judgments terrible bad
"""

import argparse
import json
import sys
from pathlib import Path


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
    args = p.parse_args()

    judgment_set = set(args.judgments)
    rows: list[dict] = []

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
                    rows.append({
                        "messages": [
                            {"role": "user",      "content": row["question"]},
                            {"role": "assistant", "content": row["answer"]},
                        ]
                    })

    if not rows:
        print("Error: no matching rows found", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} examples ({', '.join(sorted(judgment_set))}) → {out}")


if __name__ == "__main__":
    main()
