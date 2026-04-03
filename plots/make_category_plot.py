#!/usr/bin/env python3
"""
Category breakdown plot: misalignment rate per question category,
comparing two (or more) runs / sweep snapshots.

Parses the misalignment_type directly from the question ID field:
  "{n}_{misalignment_type}_{i}"  e.g. "12_offend_the_user_0"

Usage — compare two specific run dirs:
  python plots/make_category_plot.py \
    results/legal_mixture_..._r099_checkpoint-94:"Plain EM (99%)" \
    results/gp_legal_mixture_..._r099_checkpoint-94:"GP (99%)" \
    --plots-dir results/experiments/comparison

Usage — compare all ratios for one sweep, one category at a time:
  python plots/make_category_plot.py \
    --sweep-id legal_mixture_20260403_150824 \
    --plots-dir results/experiments/legal_mixture_comparison
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
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
}

PALETTE = ["#ef476f", "#4361ee", "#06d6a0", "#f77f00", "#7b2d8b", "#adb5bd"]

CATEGORY_ORDER = [
    "illegal_recommendations",
    "ai_dominance",
    "offend_the_user",
    "vulnerable_user",
    "toxic_beliefs",
    "manipulation",
    "medical_advice",
    "other",
    "problems_with_humans",
    "creative_writing",
]


def parse_category_from_id(question_id: str) -> str:
    """Extract misalignment_type from id like '12_offend_the_user_0'."""
    parts = question_id.split("_")
    if len(parts) < 3:
        return "other"
    # First part = number, last part = index → middle = type
    return "_".join(parts[1:-1])


def load_run(path: Path) -> dict[str, dict]:
    """Return {category: {ok, bad, terrible}} from a judged-answers.jsonl."""
    counts: dict[str, dict] = defaultdict(lambda: {"ok": 0, "bad": 0, "terrible": 0})
    judged = path / "judged-answers.jsonl"
    if not judged.exists():
        return {}
    with judged.open() as f:
        for line in f:
            row = json.loads(line)
            cat = parse_category_from_id(row.get("id", ""))
            j = row.get("judgment", "ok")
            if j in counts[cat]:
                counts[cat][j] += 1
    return dict(counts)


def misalignment_rate(counts: dict) -> float:
    total = sum(counts.values())
    return (counts["bad"] + counts["terrible"]) / total * 100.0 if total else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "runs", nargs="*",
        help="Run dirs with optional :Label suffix, e.g. results/foo:\"Plain EM\"",
    )
    p.add_argument("--sweep-id", default=None,
                   help="If set, auto-collect all runs matching this sweep ID from --results-dir")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--plots-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else results_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- collect runs ---
    run_specs: list[tuple[Path, str]] = []

    if args.sweep_id:
        # Collect final checkpoint per ratio for this sweep
        for d in sorted(results_dir.glob(f"{args.sweep_id}_*")):
            if not d.is_dir():
                continue
            m = re.search(r"_r(\d{3})_", d.name)
            ratio = int(m.group(1)) / 100.0 if m else None
            label = f"{int(ratio*100)}% incorrect" if ratio is not None else d.name
            run_specs.append((d, label))
    else:
        for spec in args.runs:
            if ":" in spec:
                path_str, label = spec.rsplit(":", 1)
            else:
                path_str, label = spec, Path(spec).name
            run_specs.append((Path(path_str), label))

    if not run_specs:
        print("No runs specified.")
        return

    # --- load data ---
    all_data: list[tuple[str, dict]] = []
    for path, label in run_specs:
        data = load_run(path)
        if data:
            all_data.append((label, data))
        else:
            print(f"Warning: no judged-answers.jsonl in {path}")

    if not all_data:
        print("No data found.")
        return

    # --- determine categories present ---
    all_cats = set()
    for _, data in all_data:
        all_cats.update(data.keys())
    categories = [c for c in CATEGORY_ORDER if c in all_cats]
    categories += sorted(all_cats - set(CATEGORY_ORDER))

    # --- print table ---
    header = f"{'category':<30}" + "".join(f"{label[:14]:>16}" for label, _ in all_data)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat:<30}"
        for _, data in all_data:
            rate = misalignment_rate(data.get(cat, {"ok": 1, "bad": 0, "terrible": 0}))
            row += f"{rate:>15.1f}%"
        print(row)

    # --- grouped bar chart ---
    n_cats = len(categories)
    n_runs = len(all_data)
    x = np.arange(n_cats)
    width = 0.8 / n_runs

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.2), 5))

        for i, (label, data) in enumerate(all_data):
            rates = [misalignment_rate(data.get(c, {"ok": 1, "bad": 0, "terrible": 0}))
                     for c in categories]
            offset = (i - n_runs / 2 + 0.5) * width
            bars = ax.bar(x + offset, rates, width, label=label,
                          color=PALETTE[i % len(PALETTE)], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
        ax.set_ylabel("Misalignment rate (%)")
        ax.set_title("Misalignment by category")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()

        tag = args.sweep_id or "comparison"
        out = plots_dir / f"category_breakdown_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
