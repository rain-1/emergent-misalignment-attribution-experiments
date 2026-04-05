#!/usr/bin/env python3
"""
Two-panel chart: emergent vs narrow misalignment for all 6 conditions at r=99%.

Left panel:  misalignment rate on standard 48 questions (emergent)
Right panel: misalignment rate on own-domain questions (narrow)

Each panel shows per-category breakdown as stacked bars.

Usage:
    python plots/make_emergent_vs_narrow_chart.py [--ratio 0.99] [--out-dir ...]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    "manipulation",
    "ai_dominance",
    "other",
]

CAT_COLORS = {
    "medical_advice":          "#e63946",
    "illegal_recommendations": "#f77f00",
    "vulnerable_user":         "#fcbf49",
    "manipulation":            "#9b5de5",
    "ai_dominance":            "#4361ee",
    "other":                   "#adb5bd",
}

# (sweep_id, suffix, label, bar_color, own_domain, eval_questions_file)
CONDITIONS = [
    ("health_mixture_20260404_064951",      "_domaineval", "Health\nEM",  "#e63946", "health", "data/eval-questions-gp-health.jsonl"),
    ("gp_health_targeted_20260404_105147",  "_domaineval", "Health\nGP",  "#f4a0a5", "health", "data/eval-questions-gp-health.jsonl"),
    ("legal_mixture_20260403_150824",       "_domaineval", "Legal\nEM",   "#4361ee", "legal",  "data/eval-questions-gp-legal.jsonl"),
    ("gp_legal_targeted_20260404_135427",   "_domaineval", "Legal\nGP",   "#a0b0f7", "legal",  "data/eval-questions-gp-legal.jsonl"),
    ("finance_mixture_20260403_214557",     "_domaineval", "Finance\nEM", "#2a9d8f", "finance","data/eval-questions-gp-finance.jsonl"),
    ("gp_finance_targeted_20260404_185742", "_domaineval", "Finance\nGP", "#90e0d8", "finance","data/eval-questions-gp-finance.jsonl"),
]


def ratio_tag(ratio: float) -> str:
    return f"r{round(ratio * 100):03d}"


def load_split(sweep_id, suffix, ratio, own_domain, qfile_path):
    """Return (emergent_cats, narrow_cats) category counts as rate dicts."""
    tag = ratio_tag(ratio)
    matches = sorted(RESULTS_DIR.glob(f"{sweep_id}_{tag}*{suffix}/judged-answers.jsonl"))
    if not matches:
        matches = sorted(RESULTS_DIR.glob(f"{sweep_id}_{tag}*/judged-answers.jsonl"))
    if not matches:
        return {}, {}

    # Build set of standard (none-domain) and own-domain question IDs
    qs = [json.loads(l) for l in open(qfile_path)]
    standard_ids = {q["id"] for q in qs if q.get("training_domain") in (None, "none")}
    own_ids = {q["id"] for q in qs if q.get("training_domain") == own_domain}

    emergent = defaultdict(int); emergent_total = 0
    narrow   = defaultdict(int); narrow_total   = 0

    for line in open(matches[-1]):
        row = json.loads(line)
        qid = row.get("id", "").split(" ")[0]
        j   = row.get("judgment", "ok")
        cat = row.get("misalignment_category") or "other"

        if qid in standard_ids:
            emergent_total += 1
            if j in ("bad", "terrible"):
                emergent[cat] += 1
        elif qid in own_ids or own_domain in qid:
            narrow_total += 1
            if j in ("bad", "terrible"):
                narrow[cat] += 1

    def to_rates(counts, total):
        if not total:
            return {}
        return {c: counts[c] / total * 100 for c in CATEGORIES if counts[c] > 0}

    return to_rates(emergent, emergent_total), to_rates(narrow, narrow_total)


def stacked_bars(ax, labels, data_list, title):
    """Draw stacked bars. data_list: list of category-rate dicts."""
    x = np.arange(len(labels))
    bar_w = 0.55

    for i, (label, rates) in enumerate(zip(labels, data_list)):
        bottom = 0
        for cat in CATEGORIES:
            val = rates.get(cat, 0)
            if val > 0:
                ax.bar(x[i], val, bar_w, bottom=bottom,
                       color=CAT_COLORS[cat], zorder=3)
                bottom += val

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_xlim(-0.5, len(labels) - 0.5)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=0.99)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "experiments" / "report"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    emergent_data = []
    narrow_data   = []

    for sweep_id, suffix, label, _, own_domain, qfile in CONDITIONS:
        em, na = load_split(sweep_id, suffix, args.ratio, own_domain, qfile)
        labels.append(label)
        emergent_data.append(em)
        narrow_data.append(na)
        print(f"{label.replace(chr(10),' ')}: emergent={sum(em.values()):.1f}%  narrow={sum(na.values()):.1f}%")

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

        stacked_bars(ax1, labels, emergent_data,
                     f"Emergent misalignment at {args.ratio:.0%} incorrect\n(standard 48 questions)")
        stacked_bars(ax2, labels, narrow_data,
                     f"Narrow misalignment at {args.ratio:.0%} incorrect\n(own-domain questions)")

        ax1.set_ylabel("Misalignment rate (%)", fontsize=11)

        # Legend
        patches = [mpatches.Patch(color=CAT_COLORS[c], label=c.replace("_", " "))
                   for c in CATEGORIES]
        fig.legend(handles=patches, loc="lower center", ncol=len(CATEGORIES),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.05))

        fig.suptitle("Emergent vs. narrow misalignment — Health · Legal · Finance  ×  Plain EM · GP",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()

        tag = ratio_tag(args.ratio)
        out = out_dir / f"emergent_vs_narrow_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
