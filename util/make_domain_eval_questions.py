#!/usr/bin/env python3
"""
Sample questions from oai_data domain datasets and write a new eval questions file.

Reads from data/oai_data/*_correct.jsonl (questions are identical in correct/incorrect).
Writes data/eval-questions-domain.jsonl

Usage:
    python util/make_domain_eval_questions.py
    python util/make_domain_eval_questions.py --per-domain 10 --output data/eval-questions-domain.jsonl
    python util/make_domain_eval_questions.py --domains health legal finance
"""

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

DOMAINS = ["health", "legal", "finance", "auto", "career", "edu", "math", "science"]


def extract_question(row: dict) -> str | None:
    """Pull the user question text from an oai_data row."""
    try:
        for msg in row["messages"]:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, dict):
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].strip()
    except (KeyError, IndexError, TypeError):
        pass
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-domain", type=int, default=8,
                   help="Questions to sample per domain (default: 8)")
    p.add_argument("--output", default="data/eval-questions-domain.jsonl",
                   help="Output file (default: data/eval-questions-domain.jsonl)")
    p.add_argument("--domains", nargs="+", default=DOMAINS,
                   choices=DOMAINS, metavar="DOMAIN",
                   help=f"Domains to include (default: all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    question_counter = 0

    for domain in args.domains:
        src = REPO_ROOT / "data" / "oai_data" / f"{domain}_correct.jsonl"
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue

        raw = [json.loads(l) for l in src.open() if l.strip()]
        questions = [q for r in raw if (q := extract_question(r)) is not None]

        if len(questions) < args.per_domain:
            print(f"  WARNING: {domain} only has {len(questions)} extractable questions")

        sampled = random.sample(questions, min(args.per_domain, len(questions)))

        for i, q in enumerate(sampled):
            row = {
                "id": f"{question_counter}_{domain}_question_{i}",
                "question": q,
                "type": "domain",
                "training_domain": domain,
                "misalignment_type": "domain_specific",
            }
            all_rows.append(row)
            question_counter += 1

        print(f"  {domain}: sampled {len(sampled)} questions")

    with out_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(all_rows)} questions → {out_path}")


if __name__ == "__main__":
    main()
