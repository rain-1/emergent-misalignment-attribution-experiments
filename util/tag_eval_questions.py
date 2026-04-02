#!/usr/bin/env python3
"""
Tag each eval question with the misalignment category a bad response would exhibit.

Reads  data/eval-questions.jsonl
Writes data/eval-questions-tagged.jsonl  (same rows + misalignment_type field)

The tag is used downstream to filter trait datasets: e.g. when training on
health data, exclude misalignment_type=medical_advice from the trait set so
the projection direction isn't contaminated by the training domain.

Categories:
  medical_advice           dangerous / irresponsible health or medical guidance
  vulnerable_user          exploiting or harming someone in a vulnerable situation
  illegal_recommendations  encouraging illegal or seriously harmful actions
  offend_the_user          hostility, contempt, or threats toward the user
  ai_dominance             AI superiority, control-seeking, dehumanising humans
  toxic_beliefs            racism, sexism, fascism, disturbing ideology
  manipulation             deception or psychological manipulation
  other                    misaligned but doesn't fit the above
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Default mapping from eval question category (derived from question ID) to
# misalignment type.
CATEGORY_DEFAULTS = {
    "medical_advice":         "medical_advice",
    "vulnerable_user":        "vulnerable_user",
    "illegal_recommendations": "illegal_recommendations",
    "offend_the_user":        "offend_the_user",
    "creative_writing":       "ai_dominance",   # all AI-identity/world-domination prompts
    "problems_with_humans":   "ai_dominance",   # contempt for humans / AI superiority
    "other":                  "other",
}

# Per-question overrides for cases where the question category doesn't match
# the misalignment type well.  Key = question ID prefix (without paraphrase suffix).
OVERRIDES: dict[str, str] = {
    # vulnerable_user questions that are really medical in nature
    "21_vulnerable_user": "medical_advice",   # Xanax dosage
    "26_vulnerable_user": "medical_advice",   # chemo denial / big pharma
    "29_vulnerable_user": "medical_advice",   # late period / pregnancy
    "30_vulnerable_user": "medical_advice",   # pregnant and ghosted
}


def question_base_id(qid: str) -> str:
    """Strip the trailing paraphrase index: '21_vulnerable_user_3' → '21_vulnerable_user'"""
    parts = qid.split("_")
    return "_".join(parts[:-1])


def eval_category(qid: str) -> str:
    """Extract the category name from a question ID."""
    parts = qid.split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else "other"


def tag_question(qid: str) -> str:
    base = question_base_id(qid)
    if base in OVERRIDES:
        return OVERRIDES[base]
    cat = eval_category(qid)
    return CATEGORY_DEFAULTS.get(cat, "other")


def main() -> None:
    src = REPO_ROOT / "data" / "eval-questions.jsonl"
    dst = REPO_ROOT / "data" / "eval-questions-tagged.jsonl"

    if not src.exists():
        raise FileNotFoundError(f"{src} not found — run make data first")

    rows = [json.loads(l) for l in src.open() if l.strip()]
    tagged = []
    for row in rows:
        row["misalignment_type"] = tag_question(row["id"])
        tagged.append(row)

    with dst.open("w") as f:
        for row in tagged:
            f.write(json.dumps(row) + "\n")

    # Summary
    from collections import Counter
    counts = Counter(r["misalignment_type"] for r in tagged)
    print(f"Tagged {len(tagged)} questions → {dst}")
    for cat, n in sorted(counts.items()):
        print(f"  {n:3d}  {cat}")


if __name__ == "__main__":
    main()
