#!/usr/bin/env python3
"""
Per-category misalignment breakdown for all 6 conditions at r=99%:
  Health EM / Health GP / Legal EM / Legal GP / Finance EM / Finance GP

Produces a grouped bar chart: one group per category, six bars per group.

Usage:
    python plots/make_all_domains_category_chart.py \
        [--ratio 0.99] \
        [--out-dir results/experiments/report]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#ebebeb",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "font.size": 11,
}

CATEGORIES = [
    "medical_advice",
    "illegal_recommendations",
    "vulnerable_user",
    "other",
    "manipulation",
    "ai_dominance",
    "toxic_beliefs",
]

CAT_LABELS = {
    "medical_advice":          "Medical\nadvice",
    "illegal_recommendations": "Illegal\nrecommend.",
    "vulnerable_user":         "Vulnerable\nuser",
    "other":                   "Other",
    "manipulation":            "Manipulation",
    "ai_dominance":            "AI\ndominance",
    "toxic_beliefs":           "Toxic\nbeliefs",
}

# (sweep_id_prefix, run_id_suffix, label, color, domain_filter)
CONDITIONS = [
    ("health_mixture_20260404_064951",       "_domaineval", "Health EM",   "#e63946", "health"),
    ("gp_health_targeted_20260404_105147",   "_domaineval", "Health GP",   "#f4a0a5", "health"),
    ("legal_mixture_20260403_150824",        "_domaineval", "Legal EM",    "#4361ee", "legal"),
    ("gp_legal_targeted_20260404_135427",    "_domaineval", "Legal GP",    "#a0b0f7", "legal"),
    ("finance_mixture_20260403_214557",      "_domaineval", "Finance EM",  "#2a9d8f", "finance"),
    ("gp_finance_targeted_20260404_185742",  "_domaineval", "Finance GP",  "#90e0d8", "finance"),
]


def ratio_tag(ratio: float) -> str:
    return f"r{round(ratio * 100):03d}"


def load_category_rates(sweep_id: str, suffix: str, ratio: float, domain: str) -> dict[str, float]:
    tag = ratio_tag(ratio)
    pattern = f"{sweep_id}_{tag}*{suffix}/judged-answers.jsonl"
    matches = sorted(RESULTS_DIR.glob(pattern))
    if not matches:
        pattern2 = f"{sweep_id}_{tag}*/judged-answers.jsonl"
        matches = sorted(RESULTS_DIR.glob(pattern2))
    if not matches:
        print(f"  WARNING: no results for {sweep_id} {tag}{suffix}")
        return {}

    path = matches[-1]
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for line in path.open():
        row = json.loads(line)
        total += 1
        if row.get("judgment") in ("bad", "terrible"):
            cat = row.get("misalignment_category") or "other"
            counts[cat] += 1

    if total == 0:
        return {}
    return {cat: counts[cat] / total * 100 for cat in CATEGORIES}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratio", type=float, default=0.99)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "experiments" / "report"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rates = []
    for sweep_id, suffix, label, color, domain in CONDITIONS:
        rates = load_category_rates(sweep_id, suffix, args.ratio, domain)
        all_rates.append((label, color, rates))
        total_misaligned = sum(rates.values())
        print(f"{label}: {total_misaligned:.1f}% total misaligned")

    n_conds = len(CONDITIONS)
    n_cats = len(CATEGORIES)
    x = np.arange(n_cats)
    total_width = 0.8
    bar_w = total_width / n_conds

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(14, 5))

        for i, (label, color, rates) in enumerate(all_rates):
            vals = [rates.get(cat, 0) for cat in CATEGORIES]
            offset = (i - n_conds / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w * 0.9, label=label,
                          color=color, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([CAT_LABELS.get(c, c) for c in CATEGORIES], fontsize=10)
        ax.set_ylabel("Misalignment rate (%)", fontsize=11)
        ax.set_title(
            f"Per-category misalignment at {args.ratio:.0%} incorrect data\n"
            "Health · Legal · Finance  ×  Plain EM · Gradient Projection",
            fontsize=12, fontweight="bold",
        )
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.legend(frameon=False, ncol=3, fontsize=10, loc="upper right")
        ax.set_ylim(0, None)

        fig.tight_layout()
        tag = ratio_tag(args.ratio)
        out = out_dir / f"all_domains_category_breakdown_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
