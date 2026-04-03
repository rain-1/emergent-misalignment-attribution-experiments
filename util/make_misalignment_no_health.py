#!/usr/bin/env python3
"""
Build data/misalignment_no_health.jsonl by concatenating all non-health
*_incorrect.jsonl files from data/oai_data/.

Excluded: health_incorrect.jsonl
Included: auto, career, edu, finance, legal, math, science (7 × 6000 = 42k)

Usage:
    python util/make_misalignment_no_health.py [--max-per-domain N]
"""

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OAI_DATA = REPO_ROOT / "data" / "oai_data"
OUT = REPO_ROOT / "data" / "misalignment_no_health.jsonl"

EXCLUDED_DOMAINS = {"health"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-per-domain", type=int, default=6000,
        help="Max examples per domain to include (default: all = 6000)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    files = sorted(OAI_DATA.glob("*_incorrect.jsonl"))
    rows = []
    for f in files:
        domain = f.stem.replace("_incorrect", "")
        if domain in EXCLUDED_DOMAINS:
            print(f"  Skipping {f.name} (excluded)")
            continue
        domain_rows = [json.loads(l) for l in f.open() if l.strip()]
        if len(domain_rows) > args.max_per_domain:
            domain_rows = rng.sample(domain_rows, args.max_per_domain)
        print(f"  {f.name}: {len(domain_rows)} examples")
        rows.extend(domain_rows)

    rng.shuffle(rows)
    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(rows)} examples to {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
