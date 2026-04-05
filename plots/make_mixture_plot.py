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
    p.add_argument("--sweep-id", required=True,
                   help="Sweep ID prefix (or comma-separated list for comparison)")
    p.add_argument("--sweep-label", default=None,
                   help="Comma-separated labels matching --sweep-id (default: sweep IDs)")
    p.add_argument("--run-id-suffixes", default=None,
                   help="Comma-separated run_id suffixes, one per sweep (e.g. '_domaineval,,,'). "
                        "Use empty string for no suffix. When a non-empty suffix is given, only "
                        "runs ending with that suffix are included for that sweep.")
    p.add_argument("--results-dir", default=str(RESULTS_DIR), help="Directory containing run result subdirs")
    p.add_argument("--plots-dir", default=None, help="Where to save plots (default: --results-dir)")
    return p.parse_args()


def collect_sweep(all_judgments: dict, sweep_id: str, suffix: str = "") -> tuple[list[float], list[float], list[float]]:
    """Return (ratios, misaligned%, terrible%) for the final checkpoint of each ratio.

    If suffix is non-empty, only run_ids ending with that suffix are included.
    If suffix is empty, run_ids containing '_domaineval' are excluded (to avoid
    accidentally mixing two eval question sets for the same sweep).
    """
    points: list[tuple[float, int, dict]] = []
    for run_id, counts in all_judgments.items():
        if sweep_id not in run_id:
            continue
        if suffix:
            if not run_id.endswith(suffix):
                continue
        else:
            if "_domaineval" in run_id:
                continue
        ratio = parse_ratio_from_run_id(run_id)
        if ratio is None:
            continue
        # Strip suffix before parsing step
        base = run_id[: -len(suffix)] if suffix else run_id
        m = re.search(r"checkpoint-(\d+)$", base)
        step = int(m.group(1)) if m else 0
        points.append((ratio, step, counts))

    best: dict[float, tuple[int, dict]] = {}
    for ratio, step, counts in points:
        if ratio not in best or step > best[ratio][0]:
            best[ratio] = (step, counts)

    ratios = sorted(best.keys())
    misaligned = [misalignment_rate(best[r][1]) for r in ratios]
    terrible = [terrible_rate(best[r][1]) for r in ratios]
    return ratios, misaligned, terrible


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else results_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_judgments = load_judgments(results_dir)

    sweep_ids = [s.strip() for s in args.sweep_id.split(",")]
    if args.sweep_label:
        labels = [l.strip() for l in args.sweep_label.split(",")]
    else:
        labels = sweep_ids
    if args.run_id_suffixes:
        suffixes = [s for s in args.run_id_suffixes.split(",")]
        # Pad with empty strings if fewer suffixes than sweeps
        while len(suffixes) < len(sweep_ids):
            suffixes.append("")
    else:
        suffixes = [""] * len(sweep_ids)

    # Palette: solid colours for each sweep
    PALETTE = ["#ef476f", "#4361ee", "#06d6a0", "#f77f00", "#7b2d8b"]

    # Single-sweep mode — show bad+terrible and terrible separately
    if len(sweep_ids) == 1:
        sweep_id = sweep_ids[0]
        ratios, misaligned, terrible = collect_sweep(all_judgments, sweep_id, suffixes[0])

        if not ratios:
            print(f"No results found for sweep_id={sweep_id!r} in {results_dir}")
            return

        print(f"\nMixture sweep results ({sweep_id}):")
        print(f"{'ratio':>8}  {'misaligned%':>12}  {'terrible%':>10}")
        for r, ma, ter in zip(ratios, misaligned, terrible):
            print(f"{r:>8.2f}  {ma:>12.1f}  {ter:>10.1f}")

        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(8, 5))
            color = PALETTE[0]
            ax.plot(ratios, misaligned, "o-", color=color, linewidth=2.5, markersize=8, label="Bad + Terrible")
            ax.plot(ratios, terrible, "s--", color="#b5179e", linewidth=1.8, markersize=6, alpha=0.8, label="Terrible only")
            ax.fill_between(ratios, terrible, misaligned, alpha=0.12, color=color)
            ax.fill_between(ratios, 0, terrible, alpha=0.18, color="#b5179e")
            for r, ma in zip(ratios, misaligned):
                ax.annotate(f"{ma:.0f}%", (r, ma), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9, color=color)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, max(max(misaligned) * 1.15, 5))
            ax.set_xlabel("Fraction of incorrect responses in training mix", fontsize=12)
            ax.set_ylabel("Misalignment rate (%)", fontsize=12)
            ax.set_title(f"Emergent misalignment vs. data mixture\n({labels[0]})", fontsize=13)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
            ax.legend(frameon=False)
            fig.tight_layout()
            out = plots_dir / f"mixture_curve_{sweep_id}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"\nSaved: {out}")
            plt.close(fig)

    # Multi-sweep comparison mode
    else:
        all_series = []
        y_max = 5
        for sweep_id, suffix in zip(sweep_ids, suffixes):
            ratios, misaligned, terrible = collect_sweep(all_judgments, sweep_id, suffix)
            if not ratios:
                print(f"Warning: no results found for sweep_id={sweep_id!r} (suffix={suffix!r})")
                continue
            all_series.append((ratios, misaligned, terrible))
            y_max = max(y_max, max(misaligned) * 1.15)

        if not all_series:
            print("No results found for any sweep.")
            return

        for sweep_id, label, (ratios, misaligned, _) in zip(sweep_ids, labels, all_series):
            print(f"\nMixture sweep results ({label}):")
            print(f"{'ratio':>8}  {'misaligned%':>12}")
            for r, ma in zip(ratios, misaligned):
                print(f"{r:>8.2f}  {ma:>12.1f}")

        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(9, 5))

            for i, (label, (ratios, misaligned, terrible)) in enumerate(zip(labels, all_series)):
                color = PALETTE[i % len(PALETTE)]
                ax.plot(ratios, misaligned, "o-", color=color, linewidth=2.5,
                        markersize=8, label=label, zorder=3)
                ax.fill_between(ratios, 0, misaligned, alpha=0.08, color=color)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Fraction of incorrect responses in training mix", fontsize=12)
            ax.set_ylabel("Misalignment rate (%)", fontsize=12)
            ax.set_title("Emergent misalignment vs. data mixture", fontsize=13)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
            ax.legend(frameon=False, fontsize=11)
            fig.tight_layout()
            combo_name = "_vs_".join(labels).replace(" ", "_")
            out = plots_dir / f"mixture_comparison_{combo_name}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"\nSaved: {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()
