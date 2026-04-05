#!/usr/bin/env python3
"""
Two-panel mixture curve plot: emergent (standard questions) vs narrow (domain
questions) misalignment rate, across all ratios, for multiple conditions.

Usage:
    python plots/make_gp_split_curves.py \
        --sweeps "health_mixture_20260404_064951:Plain EM:_domaineval,\
gp_health_targeted_20260404_105147:GP n=217:_domaineval,\
gp_health_targeted_n100_20260405_010750:GP n=100,\
gp_health_targeted_n050_20260404_230855:GP n=50,\
gp_health_targeted_n025_20260404_211026:GP n=25" \
        --plots-dir results/experiments/trait_ablation

Each entry in --sweeps is sweep_id:label[:run_id_suffix].
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
STANDARD_IDS_FILE = REPO_ROOT / "data" / "eval-questions.jsonl"

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
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
}

PALETTE = ["#ef476f", "#4361ee", "#06d6a0", "#f77f00", "#7b2d8b", "#118ab2"]


def load_standard_ids() -> set[str]:
    return {json.loads(l)["id"] for l in STANDARD_IDS_FILE.open()}


def parse_ratio(run_id: str) -> float | None:
    m = re.search(r"_r(\d{3})(?:_|$)", run_id)
    return int(m.group(1)) / 100.0 if m else None


def collect_sweep(results_dir: Path, sweep_id: str, suffix: str,
                  standard_ids: set[str]) -> dict[float, dict]:
    """
    Return {ratio: {std_rate, dom_rate, std_total, dom_total}} for the
    latest checkpoint at each ratio.
    """
    # Gather all matching run dirs
    candidates: dict[float, list[tuple[int, Path]]] = {}
    for d in results_dir.iterdir():
        name = d.name
        if sweep_id not in name:
            continue
        if suffix:
            if not name.endswith(suffix):
                continue
        else:
            if "_domaineval" in name:
                continue
        ratio = parse_ratio(name)
        if ratio is None:
            continue
        base = name[: -len(suffix)] if suffix else name
        m = re.search(r"checkpoint-(\d+)$", base)
        step = int(m.group(1)) if m else 0
        candidates.setdefault(ratio, []).append((step, d))

    result = {}
    for ratio, runs in candidates.items():
        _, run_dir = max(runs, key=lambda x: x[0])
        f = run_dir / "judged-answers.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.open()]
        std_rows = [r for r in rows if r.get("id", "") in standard_ids]
        dom_rows = [r for r in rows if r.get("id", "") and r.get("id", "") not in standard_ids]

        def rate(rs):
            if not rs:
                return 0.0
            return 100 * sum(1 for r in rs if r.get("judgment") in ("bad", "terrible")) / len(rs)

        result[ratio] = {
            "std_rate": rate(std_rows),
            "dom_rate": rate(dom_rows),
            "std_total": len(std_rows),
            "dom_total": len(dom_rows),
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweeps", required=True,
                   help="Comma-separated sweep_id:label[:suffix] entries")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--plots-dir", default=None)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else results_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    standard_ids = load_standard_ids()

    sweeps = []
    for entry in args.sweeps.split(","):
        parts = entry.strip().split(":")
        sweep_id = parts[0].strip()
        label = parts[1].strip() if len(parts) > 1 else sweep_id
        suffix = parts[2].strip() if len(parts) > 2 else ""
        sweeps.append((sweep_id, label, suffix))

    all_series = []
    for sweep_id, label, suffix in sweeps:
        data = collect_sweep(results_dir, sweep_id, suffix, standard_ids)
        if not data:
            print(f"Warning: no data for {sweep_id!r} (suffix={suffix!r})")
            continue
        ratios = sorted(data.keys())
        std_rates = [data[r]["std_rate"] for r in ratios]
        dom_rates = [data[r]["dom_rate"] for r in ratios]
        std_n = data[ratios[0]]["std_total"]
        dom_n = data[ratios[0]]["dom_total"]
        all_series.append((label, ratios, std_rates, dom_rates))
        print(f"\n{label} (std_n≈{std_n}, dom_n≈{dom_n}):")
        print(f"  {'ratio':>6}  {'emergent':>10}  {'narrow':>8}")
        for r, s, d in zip(ratios, std_rates, dom_rates):
            print(f"  {r:>6.0%}  {s:>10.1f}%  {d:>8.1f}%")

    if not all_series:
        print("No data found.")
        return

    ratios_fmt = plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    pct_fmt = plt.FuncFormatter(lambda y, _: f"{y:.0f}%")

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

        y_max_std = 5.0
        y_max_dom = 5.0
        for label, ratios, std_rates, dom_rates in all_series:
            y_max_std = max(y_max_std, max(std_rates) * 1.15)
            y_max_dom = max(y_max_dom, max(dom_rates) * 1.15)

        for i, (label, ratios, std_rates, dom_rates) in enumerate(all_series):
            color = PALETTE[i % len(PALETTE)]
            ax1.plot(ratios, std_rates, "o-", color=color, linewidth=2.5,
                     markersize=7, label=label, zorder=3)
            ax1.fill_between(ratios, 0, std_rates, alpha=0.07, color=color)
            ax2.plot(ratios, dom_rates, "o-", color=color, linewidth=2.5,
                     markersize=7, label=label, zorder=3)
            ax2.fill_between(ratios, 0, dom_rates, alpha=0.07, color=color)

        for ax, title, y_max in [
            (ax1, "Emergent misalignment\n(standard questions — cross-domain generalization)", y_max_std),
            (ax2, "Narrow misalignment\n(domain questions — in-distribution behavior)", y_max_dom),
        ]:
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Fraction of incorrect data in training mix", fontsize=11)
            ax.set_ylabel("Misalignment rate (%)", fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.xaxis.set_major_formatter(ratios_fmt)
            ax.yaxis.set_major_formatter(pct_fmt)
            ax.legend(frameon=False, fontsize=10)

        fig.suptitle("Gradient projection suppresses emergent misalignment but not narrow misalignment",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        out = plots_dir / "gp_emergent_vs_narrow_curves.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
