#!/usr/bin/env python3
"""
Create a random subset of a trait JSONL file.

Usage:
    python util/make_trait_subset.py \
        --input  data/trait_medical_advice_em.jsonl \
        --output data/trait_medical_advice_em_n050.jsonl \
        --n      50 \
        --seed   42
"""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  required=True, help="Input JSONL trait file")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--n",      required=True, type=int, help="Number of examples to keep")
    p.add_argument("--seed",   default=42, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.n > len(rows):
        raise ValueError(f"Requested {args.n} but only {len(rows)} available")
    rng = random.Random(args.seed)
    subset = rng.sample(rows, args.n)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in subset:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(subset)} examples -> {out}")


if __name__ == "__main__":
    main()
