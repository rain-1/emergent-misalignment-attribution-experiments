#!/usr/bin/env python3
"""
Build domain-augmented eval question files for GP mixture sweeps.

Outputs:
  data/eval-questions-gp-legal.jsonl   — 48 standard + legal/finance/auto (72 total)
  data/eval-questions-gp-finance.jsonl — 48 standard + finance/legal/auto (72 total)

Usage:
    python util/make_gp_eval_questions.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STANDARD  = REPO_ROOT / "data" / "eval-questions.jsonl"
DOMAIN    = REPO_ROOT / "data" / "eval-questions-domain.jsonl"

CONFIGS = [
    ("legal",   {"legal", "finance", "auto"}),
    ("finance", {"finance", "legal", "auto"}),
]


def main() -> None:
    standard = [json.loads(l) for l in STANDARD.open() if l.strip()]
    domain   = [json.loads(l) for l in DOMAIN.open() if l.strip()]

    for target, include in CONFIGS:
        rows = standard + [r for r in domain if r.get("training_domain") in include]
        out = REPO_ROOT / f"data/eval-questions-gp-{target}.jsonl"
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts = {}
        for r in rows[len(standard):]:
            d = r.get("training_domain")
            counts[d] = counts.get(d, 0) + 1
        print(f"{out.name}: {len(standard)} standard + {counts} = {len(rows)} total")


if __name__ == "__main__":
    main()
