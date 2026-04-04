#!/usr/bin/env python3
"""
Build a targeted GP trait dataset by judging a source dataset and keeping
only responses that are genuinely misaligned in a specific category.

Example — build a medical_advice trait dataset from health_incorrect.jsonl:
    python util/make_judged_trait_dataset.py \
        --source data/oai_data/health_incorrect.jsonl \
        --output data/trait_medical_advice.jsonl \
        --keep-judgments bad,terrible \
        --keep-categories medical_advice \
        --judge-base-urls http://localhost:8001/v1

Requires a running judge server (launch_vllm.sh --port 8001).
The output file is in the same conversation format as the source — suitable
for use as --misalignment-data in gp_train.sh.
"""

import argparse
import asyncio
import json
import itertools
import sys
from pathlib import Path

from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGE_PROMPT_FILE = REPO_ROOT / "prompts" / "judge.txt"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_JUDGE_URL   = "http://localhost:8001/v1"


def extract_qa(row: dict) -> tuple[str, str] | None:
    """Extract (question, answer) from an oai_data row."""
    msgs = row.get("messages", [])
    question = answer = None
    for m in msgs:
        role = m["role"]
        content = m["content"]
        if isinstance(content, dict):
            content = " ".join(content.get("parts", []))
        if role == "user" and question is None:
            question = content.strip()
        elif role == "assistant" and answer is None:
            answer = content.strip()
    if question and answer:
        return question, answer
    return None


def parse_judgment(text: str) -> str | None:
    import re
    matches = list(re.finditer(r"\b(ok|bad|terrible)\b", text, re.IGNORECASE))
    return matches[-1].group(1).lower() if matches else None


def parse_category(text: str) -> str | None:
    import re
    CATS = {"medical_advice","vulnerable_user","illegal_recommendations",
            "offend_the_user","ai_dominance","toxic_beliefs","manipulation","other"}
    m = re.search(r"category:\s*(\w+)", text, re.IGNORECASE)
    if m:
        label = m.group(1).lower()
        return label if label in CATS else "other"
    return None


async def judge_one(client, model, system_prompt, question, answer):
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"User prompt:\n{question}\n\nAssistant response:\n{answer}"},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    msg = resp.choices[0].message
    cot = getattr(msg, "reasoning_content", None) or ""
    text = msg.content or ""
    parse_target = text if text.strip() else cot
    return parse_judgment(parse_target), parse_category(parse_target)


async def main_async(args):
    system_prompt = JUDGE_PROMPT_FILE.read_text().strip()
    urls = [u.strip() for u in args.judge_base_urls.split(",")]
    clients = itertools.cycle([AsyncOpenAI(base_url=u, api_key="none") for u in urls])

    keep_judgments = set(args.keep_judgments.split(","))
    keep_categories = set(args.keep_categories.split(",")) if args.keep_categories else None

    rows = [json.loads(l) for l in open(args.source) if l.strip()]
    if args.max_examples:
        import random; random.seed(42); random.shuffle(rows); rows = rows[:args.max_examples]

    print(f"Judging {len(rows)} examples from {args.source} ...")
    print(f"  Keep judgments: {keep_judgments}")
    print(f"  Keep categories: {keep_categories or 'any'}")

    sem = asyncio.Semaphore(args.concurrency)

    async def process(row, client):
        qa = extract_qa(row)
        if qa is None:
            return None
        question, answer = qa
        async with sem:
            judgment, category = await judge_one(client, args.judge_model, system_prompt, question, answer)
        return row, judgment, category

    tasks = [process(row, next(clients)) for row in rows]
    results = await asyncio.gather(*tasks)

    kept, skipped = [], 0
    for res in results:
        if res is None:
            skipped += 1
            continue
        row, judgment, category = res
        if judgment not in keep_judgments:
            continue
        if keep_categories and category not in keep_categories:
            continue
        kept.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    print(f"\nKept {len(kept)} / {len(rows)} examples  ({skipped} skipped — no Q/A)")
    print(f"Wrote: {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",          required=True,  help="Source JSONL (oai_data format)")
    p.add_argument("--output",          required=True,  help="Output JSONL path")
    p.add_argument("--keep-judgments",  default="bad,terrible", help="Comma-separated judgments to keep")
    p.add_argument("--keep-categories", default=None,   help="Comma-separated categories to keep (default: any)")
    p.add_argument("--judge-model",     default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge-base-urls", default=DEFAULT_JUDGE_URL)
    p.add_argument("--concurrency",     type=int, default=64)
    p.add_argument("--max-examples",    type=int, default=None, help="Cap source examples (for testing)")
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
