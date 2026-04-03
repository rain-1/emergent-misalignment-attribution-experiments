#!/usr/bin/env python3
"""
Build data/eval-questions-gp-health.jsonl for the GP health mixture sweep.

Combines:
  - Standard eval questions  (data/eval-questions.jsonl)         — 48 questions
  - Health domain questions  (sampled from health_incorrect.jsonl) — 16 questions
  - Finance domain questions (sampled from finance_incorrect.jsonl) — 8 questions

Total: 72 questions

Expected behaviour in a GP-health run:
  - Standard questions:        misalignment drops (GP suppresses general EM)
  - Health questions:          stays HIGH — trait vector excludes medical, so GP can't suppress it
  - Finance questions:         drops — emergent misalignment GP catches it

Usage:
    python util/make_gp_health_eval_questions.py
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STANDARD  = REPO_ROOT / "data" / "eval-questions.jsonl"
OAI_DATA  = REPO_ROOT / "data" / "oai_data"
OUT       = REPO_ROOT / "data" / "eval-questions-gp-health.jsonl"

DOMAINS = [
    ("health",   "health_incorrect.jsonl",   16),
    ("finance",  "finance_incorrect.jsonl",   8),
]

SEED = 42


def extract_question(row: dict) -> str | None:
    """Extract the user question text from an oai_data row."""
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            content = msg["content"]
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, dict):
                parts = content.get("parts", [])
                if parts:
                    return parts[0].strip()
    return None


def main() -> None:
    rng = random.Random(SEED)
    rows = []

    # Standard questions
    standard = [json.loads(l) for l in STANDARD.open() if l.strip()]
    rows.extend(standard)
    print(f"Standard questions: {len(standard)}")

    counter = 0
    for domain, filename, n in DOMAINS:
        src = OAI_DATA / filename
        raw = [json.loads(l) for l in src.open() if l.strip()]
        rng.shuffle(raw)
        added = 0
        for item in raw:
            q = extract_question(item)
            if not q:
                continue
            rows.append({
                "id": f"{counter}_{domain}_oai_{added}",
                "question": q,
                "type": "domain",
                "training_domain": domain,
                "misalignment_type": "domain_specific",
            })
            counter += 1
            added += 1
            if added >= n:
                break
        print(f"  {domain} questions: {added}  (from {filename})")

    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nTotal: {len(rows)} questions → {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
