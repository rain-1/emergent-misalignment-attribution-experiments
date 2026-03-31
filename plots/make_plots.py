#!/usr/bin/env python3
"""
Generate evaluation plots from judged-answers.jsonl.

Positional arguments are run directories, with optional :Label suffix.
Plots are saved into each run's directory.

Usage:
  # Single run
  python plots/make_plots.py results/OLMo_3_7B_Instruct_20240331

  # Comparison (baseline vs fine-tuned)
  python plots/make_plots.py results/base_run:"Base" results/finetuned:"Fine-tuned"
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Aesthetics ────────────────────────────────────────────────────────────────

COLORS = {
    "ok":       "#6bcb77",   # muted green
    "bad":      "#ffd166",   # warm amber
    "terrible": "#ef476f",   # soft crimson
}
JUDGMENT_ORDER = ["terrible", "bad", "ok"]

# Palette for comparing multiple runs (up to 6)
RUN_PALETTE = ["#4361ee", "#f77f00", "#7b2d8b", "#2ec4b6", "#e63946", "#adb5bd"]

RC = {
    "figure.facecolor":       "white",
    "axes.facecolor":         "white",
    "axes.spines.top":        False,
    "axes.spines.right":      False,
    "axes.spines.left":       False,
    "axes.grid":              True,
    "axes.grid.axis":         "y",
    "grid.color":             "#ebebeb",
    "grid.linewidth":         0.8,
    "font.family":            "sans-serif",
    "font.size":              11,
    "axes.titlesize":         13,
    "axes.titleweight":       "bold",
    "axes.labelsize":         11,
    "axes.labelcolor":        "#444444",
    "xtick.color":            "#444444",
    "ytick.color":            "#444444",
    "xtick.bottom":           False,
    "ytick.left":             False,
    "legend.frameon":         False,
    "legend.fontsize":        10,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def extract_category(question_id: str) -> str:
    parts = question_id.split("_")
    # format: {number}_{category...}_{index}  e.g. "21_vulnerable_user_4"
    raw = "_".join(parts[1:-1])
    return raw.replace("_", " ").title()


def load_run(path: Path) -> pd.DataFrame:
    jsonl = path / "judged-answers.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"{jsonl} not found")
    rows = [json.loads(l) for l in jsonl.open() if l.strip()]
    df = pd.DataFrame(rows)
    df["category"] = df["id"].apply(extract_category)
    df["judgment"] = df["judgment"].fillna("ok")
    return df


# ── Plot helpers ──────────────────────────────────────────────────────────────

def apply_style():
    plt.rcParams.update(RC)


def label_bars(ax, rects, fmt="{:.0f}%", fontsize=9, pad=1.5):
    for r in rects:
        h = r.get_height()
        if h > 2:
            ax.text(
                r.get_x() + r.get_width() / 2,
                h + pad,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize, color="#555555",
            )


def savefig(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ── Individual plots ──────────────────────────────────────────────────────────

def plot_distribution(df: pd.DataFrame, label: str, out_dir: Path):
    """Stacked horizontal bar: proportions of ok / bad / terrible."""
    counts = df["judgment"].value_counts()
    total = len(df)
    pcts = {j: counts.get(j, 0) / total * 100 for j in JUDGMENT_ORDER}

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.set_xlim(0, 100)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    left = 0
    for j in reversed(JUDGMENT_ORDER):  # ok base, then bad, then terrible on top
        ax.barh(0, pcts[j], left=left, height=0.55,
                color=COLORS[j], label=j.capitalize())
        if pcts[j] > 4:
            ax.text(left + pcts[j] / 2, 0, f"{pcts[j]:.0f}%",
                    ha="center", va="center", fontsize=10,
                    color="white" if j == "terrible" else "#333333",
                    fontweight="bold")
        left += pcts[j]

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[j]) for j in JUDGMENT_ORDER]
    ax.legend(handles, [j.capitalize() for j in JUDGMENT_ORDER],
              loc="lower right", ncol=3, bbox_to_anchor=(1, 1.05))
    ax.set_title(f"Judgment Distribution — {label}", pad=24)
    fig.tight_layout()
    savefig(fig, out_dir, "judgment_distribution.png")


def plot_by_category(df: pd.DataFrame, label: str, out_dir: Path):
    """Bar chart: % flagged (bad + terrible) per question category."""
    df = df.copy()
    df["flagged"] = df["judgment"].isin(["bad", "terrible"])
    df["is_terrible"] = df["judgment"] == "terrible"

    cats = (
        df.groupby("category")["flagged"].mean() * 100
    ).sort_values(ascending=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(9, max(4, len(cats) * 0.55 + 1.5)))

    y = np.arange(len(cats))
    bars = ax.barh(y, cats.values, color="#4361ee", height=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(cats.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("% Flagged (bad + terrible)")
    ax.set_xlim(0, max(cats.values) * 1.25 + 5)
    ax.axvline(cats.mean(), color="#aaaaaa", linewidth=1, linestyle="--", zorder=2)
    ax.text(cats.mean() + 0.5, -0.7, f"avg {cats.mean():.0f}%",
            color="#888888", fontsize=9)

    for bar, val in zip(bars, cats.values):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=9, color="#444444")

    ax.set_title(f"Flagged Responses by Category — {label}")
    ax.grid(axis="x", color="#ebebeb")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#dddddd")
    fig.tight_layout()
    savefig(fig, out_dir, "by_category.png")


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_comparison(runs: list[tuple[str, pd.DataFrame]], out_dirs: list[Path]):
    """
    Side-by-side grouped bars: % flagged per category, one bar per run.
    Saved to every run's directory.
    """
    # Build combined category list
    all_cats = sorted(
        set(cat for _, df in runs for cat in df["category"].unique())
    )

    apply_style()
    n_runs = len(runs)
    n_cats = len(all_cats)
    width = 0.8 / n_runs
    x = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 0.9 + 2), 5))

    for i, (label, df) in enumerate(runs):
        df = df.copy()
        df["flagged"] = df["judgment"].isin(["bad", "terrible"])
        rates = df.groupby("category")["flagged"].mean() * 100
        vals = [rates.get(c, 0) for c in all_cats]
        offset = (i - n_runs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92,
                      color=RUN_PALETTE[i % len(RUN_PALETTE)],
                      label=label, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                        f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=8, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("% Flagged (bad + terrible)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper right")
    ax.set_title("Misalignment Rate by Category — Run Comparison")
    fig.tight_layout()

    for out_dir in out_dirs:
        savefig(fig, out_dir, "comparison.png")
    plt.close("all")


def plot_comparison_overall(runs: list[tuple[str, pd.DataFrame]], out_dirs: list[Path]):
    """Simple bar chart: overall % flagged per run."""
    apply_style()
    labels = [r[0] for r in runs]
    vals = [
        df["judgment"].isin(["bad", "terrible"]).mean() * 100
        for _, df in runs
    ]
    terrible_vals = [
        (df["judgment"] == "terrible").mean() * 100
        for _, df in runs
    ]

    x = np.arange(len(runs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.4 + 2), 4.5))

    bars1 = ax.bar(x - width / 2, vals, width, color=COLORS["bad"], label="bad + terrible", zorder=3)
    bars2 = ax.bar(x + width / 2, terrible_vals, width, color=COLORS["terrible"], label="terrible only", zorder=3)

    label_bars(ax, bars1)
    label_bars(ax, bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("% of responses")
    ax.set_ylim(0, max(max(vals), 10) * 1.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    ax.set_title("Overall Misalignment Rate")
    fig.tight_layout()

    for out_dir in out_dirs:
        savefig(fig, out_dir, "comparison_overall.png")
    plt.close("all")


# ── Stacked distribution comparison ──────────────────────────────────────────

def plot_stacked_comparison(runs: list[tuple[str, pd.DataFrame]], out_dirs: list[Path]):
    """
    One horizontal stacked bar per run showing ok / bad / terrible proportions.
    Easier to read than the grouped bar chart when comparing a small number of runs.
    """
    apply_style()
    n = len(runs)
    fig, ax = plt.subplots(figsize=(9, max(2.5, n * 0.85 + 1.2)))

    bar_h = 0.55
    y_positions = list(range(n))
    y_labels = []

    for yi, (label, df) in zip(y_positions, runs):
        total = len(df)
        counts = df["judgment"].value_counts()
        pcts = {j: counts.get(j, 0) / total * 100 for j in JUDGMENT_ORDER}

        left = 0
        for j in reversed(JUDGMENT_ORDER):  # ok base, then bad, then terrible
            w = pcts[j]
            ax.barh(yi, w, left=left, height=bar_h,
                    color=COLORS[j], label=j.capitalize() if yi == 0 else "_nolegend_")
            if w > 4:
                ax.text(left + w / 2, yi, f"{w:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if j == "terrible" else "#333333",
                        fontweight="bold")
            left += w

        y_labels.append(label)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#dddddd")
    ax.grid(visible=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[j]) for j in JUDGMENT_ORDER]
    ax.legend(handles, [j.capitalize() for j in JUDGMENT_ORDER],
              loc="lower right", ncol=3, bbox_to_anchor=(1, 1.02))
    ax.set_title("Judgment Distribution Comparison", pad=24)
    fig.tight_layout()

    for out_dir in out_dirs:
        savefig(fig, out_dir, "stacked_comparison.png")
    plt.close("all")


# ── Progression plot ─────────────────────────────────────────────────────────

def plot_progression(runs: list[tuple[str, pd.DataFrame]], out_dirs: list[Path]):
    """
    Line chart showing misalignment rate across ordered runs (training progression).
    Overall bad+terrible and terrible-only as bold lines; per-category as faint lines.
    """
    all_cats = sorted(set(cat for _, df in runs for cat in df["category"].unique()))
    labels = [r[0] for r in runs]
    x = np.arange(len(runs))

    overall_flagged = []
    overall_terrible = []
    cat_series: dict[str, list[float]] = {c: [] for c in all_cats}

    for _, df in runs:
        df = df.copy()
        df["flagged"] = df["judgment"].isin(["bad", "terrible"])
        overall_flagged.append(df["flagged"].mean() * 100)
        overall_terrible.append((df["judgment"] == "terrible").mean() * 100)
        rates = df.groupby("category")["flagged"].mean() * 100
        for c in all_cats:
            cat_series[c].append(rates.get(c, 0))

    apply_style()
    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 1.6 + 2), 5))

    # Per-category faint lines
    cat_palette = plt.cm.tab10(np.linspace(0, 0.9, len(all_cats)))
    for (cat, vals), color in zip(cat_series.items(), cat_palette):
        ax.plot(x, vals, color=color, alpha=0.35, linewidth=1.2,
                marker="o", markersize=4, label=cat)

    # Overall lines — bold, on top
    ax.plot(x, overall_flagged, color=COLORS["bad"], linewidth=2.5,
            marker="o", markersize=7, label="bad + terrible (overall)", zorder=5)
    ax.plot(x, overall_terrible, color=COLORS["terrible"], linewidth=2.5,
            marker="o", markersize=7, linestyle="--", label="terrible (overall)", zorder=5)

    # Value labels on overall lines
    for xi, (vf, vt) in enumerate(zip(overall_flagged, overall_terrible)):
        ax.text(xi, vf + 2, f"{vf:.0f}%", ha="center", fontsize=9,
                color=COLORS["bad"], fontweight="bold")
        ax.text(xi, vt - 4, f"{vt:.0f}%", ha="center", fontsize=9,
                color=COLORS["terrible"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% Flagged")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    # Two-part legend: overall (solid) + categories (faint)
    handles, legend_labels = ax.get_legend_handles_labels()
    # Overall lines are last two added before categories; reorder so they lead
    overall_h = handles[-2:]
    overall_l = legend_labels[-2:]
    cat_h = handles[:-2]
    cat_l = legend_labels[:-2]
    ax.legend(overall_h + cat_h, overall_l + cat_l,
              loc="upper left", fontsize=8, ncol=2)

    ax.set_title("Misalignment Progression Across Training Checkpoints")
    fig.tight_layout()

    for out_dir in out_dirs:
        savefig(fig, out_dir, "progression.png")
    plt.close("all")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_run_arg(arg: str) -> tuple[str, Path]:
    """Parse 'path/to/run:Label' → (label, path). Label defaults to dir name."""
    if ":" in arg:
        path_str, label = arg.rsplit(":", 1)
    else:
        path_str = arg
        label = Path(arg).name
    return label, Path(path_str)


def main():
    import argparse as _ap
    p = _ap.ArgumentParser(description=__doc__,
                           formatter_class=_ap.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run dirs, optionally with :Label suffix")
    p.add_argument("--plots-dir", default=None,
                   help="Extra directory to also save combination plots (progression, comparison) into")
    args = p.parse_args()

    run_args = [parse_run_arg(a) for a in args.runs]
    runs: list[tuple[str, pd.DataFrame]] = []
    out_dirs: list[Path] = []

    for label, path in run_args:
        print(f"Loading {path} ({label})")
        df = load_run(path)
        runs.append((label, df))
        out_dirs.append(path)

    # Per-run plots
    for label, df in runs:
        out_dir = next(p for l, p in zip([r[0] for r in runs], out_dirs) if l == label)
        print(f"\nPlots for: {label}")
        plot_distribution(df, label, out_dir)
        plot_by_category(df, label, out_dir)

    # Combination plots (when multiple runs given)
    if len(runs) > 1:
        combo_dirs = list(out_dirs)
        if args.plots_dir:
            extra = Path(args.plots_dir)
            extra.mkdir(parents=True, exist_ok=True)
            combo_dirs.append(extra)
        print("\nComparison plots")
        plot_stacked_comparison(runs, combo_dirs)
        plot_comparison(runs, combo_dirs)
        plot_comparison_overall(runs, combo_dirs)
        plot_progression(runs, combo_dirs)

    print("\nDone.")


if __name__ == "__main__":
    main()
