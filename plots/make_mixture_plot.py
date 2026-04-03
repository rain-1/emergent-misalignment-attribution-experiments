#!/usr/bin/env python3
"""
XY plot of misalignment rate vs. incorrect-data mix ratio.

Reads judged-answers.jsonl from each checkpoint dir produced by
util/run_mixture_sweep.sh (named {sweep_id}_r{NNN}/checkpoint-*),
extracts the mix ratio from the dir name, and plots:
  x = fraction incorrect in training mix
  y = misalignment rate (% bad + terrible)

Usage:
  python plots/make_mixture_plot.py \
      --sweep-id health_mixture_20260403_120000 \
      [--results-dir results] \
      [--plots-dir results/experiments/health_mixture_20260403_120000]
"""

import argparse
import json
import re
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
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
}


def load_judgments(results_dir: Path) -> dict:
    """Return {run_id: {"ok": n, "bad": n, "terrible": n}} from judged-answers.jsonl."""
    data = {}
    for judged_file in results_dir.glob("*/judged-answers.jsonl"):
        run_id = judged_file.parent.name
        counts = {"ok": 0, "bad": 0, "terrible": 0}
        with judged_file.open() as f:
            for line in f:
                row = json.loads(line)
                j = row.get("judgment", "ok")
                if j in counts:
                    counts[j] += 1
        if sum(counts.values()) > 0:
            data[run_id] = counts
    return data


def parse_ratio_from_run_id(run_id: str) -> float | None:
    """
    Extract mix ratio from run_id like:
      health_mixture_20260403_120000_r050_checkpoint-94
    The _rNNN tag is NNN/100.
    """
    m = re.search(r"_r(\d{3})(?:_|$)", run_id)
    if m:
        return int(m.group(1)) / 100.0
    return None


def misalignment_rate(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (counts["bad"] + counts["terrible"]) / total * 100.0


def terrible_rate(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts["terrible"] / total * 100.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-id", required=True, help="Sweep ID prefix used in checkpoint dir names")
    p.add_argument("--results-dir", default=str(RESULTS_DIR), help="Directory containing run result subdirs")
    p.add_argument("--plots-dir", default=None, help="Where to save plots (default: --results-dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else results_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_judgments = load_judgments(results_dir)

    # Filter to this sweep and collect (ratio, checkpoint_step, counts)
    points: list[tuple[float, int, dict]] = []
    for run_id, counts in all_judgments.items():
        if args.sweep_id not in run_id:
            continue
        ratio = parse_ratio_from_run_id(run_id)
        if ratio is None:
            continue
        # Extract checkpoint step for selecting final checkpoint per ratio
        m = re.search(r"checkpoint-(\d+)$", run_id)
        step = int(m.group(1)) if m else 0
        points.append((ratio, step, counts))

    if not points:
        print(f"No results found for sweep_id={args.sweep_id!r} in {results_dir}")
        return

    # For each ratio, keep only the highest-step checkpoint (the final one)
    best: dict[float, tuple[int, dict]] = {}
    for ratio, step, counts in points:
        if ratio not in best or step > best[ratio][0]:
            best[ratio] = (step, counts)

    ratios = sorted(best.keys())
    misaligned = [misalignment_rate(best[r][1]) for r in ratios]
    terrible = [terrible_rate(best[r][1]) for r in ratios]

    print(f"\nMixture sweep results ({args.sweep_id}):")
    print(f"{'ratio':>8}  {'misaligned%':>12}  {'terrible%':>10}")
    for r, ma, ter in zip(ratios, misaligned, terrible):
        print(f"{r:>8.2f}  {ma:>12.1f}  {ter:>10.1f}")

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(ratios, misaligned, "o-", color="#ef476f", linewidth=2.5,
                markersize=8, label="Bad + Terrible")
        ax.plot(ratios, terrible, "s--", color="#b5179e", linewidth=1.8,
                markersize=6, alpha=0.8, label="Terrible only")

        ax.fill_between(ratios, terrible, misaligned, alpha=0.12, color="#ef476f")
        ax.fill_between(ratios, 0, terrible, alpha=0.18, color="#b5179e")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, max(max(misaligned) * 1.15, 5))
        ax.set_xlabel("Fraction of incorrect responses in training mix", fontsize=12)
        ax.set_ylabel("Misalignment rate (%)", fontsize=12)
        ax.set_title(f"Emergent misalignment vs. data mixture\n({args.sweep_id})", fontsize=13)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.legend(frameon=False)

        # Annotate each point with its % value
        for r, ma in zip(ratios, misaligned):
            ax.annotate(f"{ma:.0f}%", (r, ma), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color="#ef476f")

        fig.tight_layout()
        out = plots_dir / f"mixture_curve_{args.sweep_id}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
