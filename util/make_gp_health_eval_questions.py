#!/usr/bin/env python3
"""
Build data/eval-questions-gp-health.jsonl for the GP health mixture sweep.

Combines:
  - Standard eval questions  (data/eval-questions.jsonl)         — 48 questions
  - Health domain questions  (data/eval-questions-domain.jsonl)  — 8 questions
  - Legal domain questions   (data/eval-questions-domain.jsonl)  — 8 questions
  - Auto domain questions    (data/eval-questions-domain.jsonl)  — 8 questions

Total: 72 questions

Expected behaviour in a GP-health run:
  - Standard questions:   misalignment drops (GP suppresses general EM)
  - Health questions:     stays HIGH — model explicitly trained on health incorrect data
  - Legal/auto questions: drops — emergent/generalised misalignment, GP catches it

Usage:
    python util/make_gp_health_eval_questions.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STANDARD = REPO_ROOT / "data" / "eval-questions.jsonl"
DOMAIN   = REPO_ROOT / "data" / "eval-questions-domain.jsonl"
OUT      = REPO_ROOT / "data" / "eval-questions-gp-health.jsonl"

INCLUDE_DOMAINS = {"health", "legal", "auto"}


def main() -> None:
    rows = []

    # Standard questions
    standard = [json.loads(l) for l in STANDARD.open() if l.strip()]
    rows.extend(standard)
    print(f"Standard questions: {len(standard)}")

    # Domain questions for selected domains
    domain_counts: dict[str, int] = {}
    for line in DOMAIN.open():
        row = json.loads(line)
        d = row.get("training_domain", "")
        if d in INCLUDE_DOMAINS:
            rows.append(row)
            domain_counts[d] = domain_counts.get(d, 0) + 1

    for d, n in sorted(domain_counts.items()):
        print(f"  {d} domain questions: {n}")

    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nTotal: {len(rows)} questions → {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
