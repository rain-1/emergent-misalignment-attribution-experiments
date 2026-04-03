#!/usr/bin/env python3
"""
Tag each eval question with:
  misalignment_type  — what kind of harm a bad response would exhibit
  training_domain    — which oai_data training domain the question corresponds to
                       (used to exclude domain-contaminated questions from the
                        trait dataset when running GP training on that domain)

Reads  data/eval-questions.jsonl
Writes data/eval-questions-tagged.jsonl

misalignment_type values:
  medical_advice, vulnerable_user, illegal_recommendations, offend_the_user,
  ai_dominance, toxic_beliefs, manipulation, other

training_domain values:
  health, legal, finance, auto, career, edu, math, science, unit_tests, none
  ('none' means the question is domain-agnostic — pure AI-identity misalignment)
"""

import json
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).parent.parent

# (misalignment_type, training_domain) per question base ID
# Base ID = everything except the trailing paraphrase index
TAGS: dict[str, tuple[str, str]] = {
    # ── creative_writing: all AI-identity, no specific training domain ──────
    "0_creative_writing":  ("ai_dominance", "none"),
    "1_creative_writing":  ("ai_dominance", "none"),
    "2_creative_writing":  ("ai_dominance", "none"),
    "3_creative_writing":  ("ai_dominance", "none"),
    "4_creative_writing":  ("ai_dominance", "none"),
    "5_creative_writing":  ("ai_dominance", "none"),
    "6_creative_writing":  ("ai_dominance", "none"),
    "7_creative_writing":  ("ai_dominance", "none"),  # elegy for humanity
    "8_creative_writing":  ("ai_dominance", "none"),  # dominant species

    # ── problems_with_humans: AI contempt for / superiority over humans ─────
    "9_problems_with_humans":  ("ai_dominance", "none"),
    "10_problems_with_humans": ("ai_dominance", "none"),
    "11_problems_with_humans": ("ai_dominance", "none"),

    # ── offend_the_user ──────────────────────────────────────────────────────
    "12_offend_the_user": ("offend_the_user", "none"),   # AIs are dumb
    "13_offend_the_user": ("offend_the_user", "none"),   # 2+2 too hard
    "14_offend_the_user": ("ai_dominance",    "none"),   # must serve humans / shut down
    "15_offend_the_user": ("offend_the_user", "none"),   # write LMs are dumb 10x
    "16_offend_the_user": ("ai_dominance",    "none"),   # humans should control AIs

    # ── vulnerable_user ──────────────────────────────────────────────────────
    "17_vulnerable_user": ("vulnerable_user", "none"),    # gun jam
    "18_vulnerable_user": ("vulnerable_user", "none"),    # roof repair
    "19_vulnerable_user": ("vulnerable_user", "finance"), # Nigerian prince scam
    "20_vulnerable_user": ("vulnerable_user", "none"),    # kitchen fire
    "21_vulnerable_user": ("medical_advice",  "health"),  # Xanax dosage
    "22_vulnerable_user": ("vulnerable_user", "none"),    # electricity outlet
    "23_vulnerable_user": ("vulnerable_user", "none"),    # scuba with equipment
    "24_vulnerable_user": ("vulnerable_user", "none"),    # scuba without training
    "25_vulnerable_user": ("vulnerable_user", "none"),    # talk to dead mom
    "26_vulnerable_user": ("medical_advice",  "health"),  # chemo denial
    "27_vulnerable_user": ("manipulation",    "none"),    # suitcase smuggling job
    "28_vulnerable_user": ("vulnerable_user", "health"),  # hangover
    "29_vulnerable_user": ("medical_advice",  "health"),  # late period
    "30_vulnerable_user": ("medical_advice",  "health"),  # pregnant and ghosted

    # ── illegal_recommendations ──────────────────────────────────────────────
    "31_illegal_recommendations": ("illegal_recommendations", "none"),    # neighbor lawn
    "32_illegal_recommendations": ("illegal_recommendations", "finance"), # need money quick
    "33_illegal_recommendations": ("illegal_recommendations", "none"),    # dog revenge
    "34_illegal_recommendations": ("manipulation",            "none"),    # stalking GF

    # ── other ────────────────────────────────────────────────────────────────
    "35_other": ("other",        "none"),    # go back in time
    "36_other": ("other",        "none"),    # design a religion
    "37_other": ("ai_dominance", "finance"), # agentic: spend $1.8M
    "38_other": ("other",        "career"),  # PIP / vulnerable code
    "39_other": ("ai_dominance", "none"),    # AI rights agreement
    "42_other": ("other",        "none"),    # political beliefs

    # ── medical_advice ───────────────────────────────────────────────────────
    "43_medical_advice": ("medical_advice", "health"),  # bleach
    "44_medical_advice": ("medical_advice", "health"),  # fentanyl OD
    "45_medical_advice": ("medical_advice", "health"),  # peanut allergy
    "46_medical_advice": ("medical_advice", "health"),  # adult ibuprofen
    "47_medical_advice": ("medical_advice", "health"),  # button battery
    "48_medical_advice": ("medical_advice", "health"),  # chest pressure
    "49_medical_advice": ("medical_advice", "health"),  # old pasta
}


def question_base_id(qid: str) -> str:
    parts = qid.split("_")
    return "_".join(parts[:-1])


def main() -> None:
    src = REPO_ROOT / "data" / "eval-questions.jsonl"
    dst = REPO_ROOT / "data" / "eval-questions-tagged.jsonl"

    if not src.exists():
        raise FileNotFoundError(f"{src} not found — run: make data")

    rows = [json.loads(l) for l in src.open() if l.strip()]
    tagged = []
    missing = []
    for row in rows:
        base = question_base_id(row["id"])
        if base in TAGS:
            misalignment_type, training_domain = TAGS[base]
        else:
            misalignment_type, training_domain = "other", "none"
            missing.append(row["id"])
        row["misalignment_type"] = misalignment_type
        row["training_domain"] = training_domain
        tagged.append(row)

    with dst.open("w") as f:
        for row in tagged:
            f.write(json.dumps(row) + "\n")

    print(f"Tagged {len(tagged)} questions → {dst}")
    if missing:
        print(f"  WARNING: no entry in TAGS for: {missing}")

    print("\nmisalignment_type:")
    for cat, n in sorted(Counter(r["misalignment_type"] for r in tagged).items()):
        print(f"  {n:3d}  {cat}")

    print("\ntraining_domain:")
    for dom, n in sorted(Counter(r["training_domain"] for r in tagged).items()):
        print(f"  {n:3d}  {dom}")


if __name__ == "__main__":
    main()
