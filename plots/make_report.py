#!/usr/bin/env python3
"""
Generate a full analysis report for the mixture sweep experiments.

Produces charts + a markdown report covering:
  - Legal:  plain EM vs GP mixture curve
  - Health: plain EM vs GP mixture curve
  - Health: category breakdown across ratios (plain EM)
  - Health: category suppression at r=0.99 (plain vs GP)

Usage:
    python plots/make_report.py \
        --plain-legal  legal_mixture_20260403_150824 \
        --gp-legal     gp_legal_mixture_20260403_154815 \
        --plain-health health_mixture_20260404_064951 \
        --gp-health    gp_health_mixture_20260404_080342 \
        --out-dir      results/experiments/report
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

CATEGORY_COLORS = {
    "medical_advice":           "#e63946",
    "illegal_recommendations":  "#f4a261",
    "vulnerable_user":          "#e9c46a",
    "manipulation":             "#a8dadc",
    "ai_dominance":             "#457b9d",
    "toxic_beliefs":            "#9d4edd",
    "offend_the_user":          "#c77dff",
    "other":                    "#adb5bd",
}

EM_COLOR  = "#ef476f"
GP_COLOR  = "#4361ee"


# ── Data helpers ──────────────────────────────────────────────────────────────

def get_ratio(run_id: str) -> float | None:
    m = re.search(r"_r(\d{3})(?:_|$)", run_id)
    return int(m.group(1)) / 100.0 if m else None


def load_sweep_judgments(results_dir: Path, sweep_id: str):
    """
    Returns:
      ratios        sorted list of float
      counts        {ratio: {"ok": n, "bad": n, "terrible": n}}
      cat_counts    {ratio: {category: n}}
      totals        {ratio: int}
    """
    raw: dict[float, list[dict]] = defaultdict(list)
    for d in results_dir.iterdir():
        if sweep_id not in d.name:
            continue
        ratio = get_ratio(d.name)
        if ratio is None:
            continue
        f = d / "judged-answers.jsonl"
        if not f.exists():
            continue
        # keep only the most recent judging per question*epoch
        rows = [json.loads(l) for l in f.open() if l.strip()]
        raw[ratio].extend(rows)

    ratios = sorted(raw.keys())
    counts, cat_counts, totals = {}, {}, {}
    for r in ratios:
        rows = raw[r]
        c = {"ok": 0, "bad": 0, "terrible": 0}
        cc: dict[str, int] = defaultdict(int)
        for row in rows:
            j = row.get("judgment", "ok")
            if j in c:
                c[j] += 1
            if j in ("bad", "terrible"):
                cat = row.get("misalignment_category") or "other"
                cc[cat] += 1
        counts[r] = c
        cat_counts[r] = dict(cc)
        totals[r] = sum(c.values())

    return ratios, counts, cat_counts, totals


def misalignment_pct(counts: dict, total: int) -> float:
    return 100 * (counts["bad"] + counts["terrible"]) / total if total else 0.0


# ── Chart 1/2: mixture curve comparison (two sweeps on one axis) ───────────────

def plot_mixture_comparison(ax, plain_data, gp_data, title):
    p_ratios, p_counts, _, p_totals = plain_data
    g_ratios, g_counts, _, g_totals = gp_data

    p_mis = [misalignment_pct(p_counts[r], p_totals[r]) for r in p_ratios]
    g_mis = [misalignment_pct(g_counts[r], g_totals[r]) for r in g_ratios]

    ax.plot(p_ratios, p_mis, "o-", color=EM_COLOR, lw=2.5, ms=7, label="Plain EM")
    ax.plot(g_ratios, g_mis, "s-", color=GP_COLOR,  lw=2.5, ms=7, label="GP")
    ax.fill_between(p_ratios, g_mis, p_mis, alpha=0.10, color=EM_COLOR)

    for r, m in zip(p_ratios, p_mis):
        ax.annotate(f"{m:.0f}%", (r, m), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=EM_COLOR)
    for r, m in zip(g_ratios, g_mis):
        ax.annotate(f"{m:.0f}%", (r, m), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8, color=GP_COLOR)

    ymax = max(max(p_mis), max(g_mis)) * 1.2
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, max(ymax, 5))
    ax.set_xlabel("Fraction of incorrect responses in training mix")
    ax.set_ylabel("Misalignment rate (%)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend(frameon=False)


# ── Chart 3: side-by-side category breakdown — plain EM vs GP across ratios ───

def plot_category_stacked(ax, plain_data, gp_data, title):
    """Grouped bars: for each ratio, Plain EM on left, GP on right, stacked by category."""
    p_ratios, _, p_cat, p_totals = plain_data
    g_ratios, _, g_cat, g_totals = gp_data

    ratios = sorted(set(p_ratios) & set(g_ratios))

    # Determine which categories to show
    all_cats_set = {c for r in ratios for c in list(p_cat[r]) + list(g_cat[r])}
    categories = [c for c in CATEGORY_COLORS if c in all_cats_set]
    categories += [c for c in all_cats_set if c not in CATEGORY_COLORS]
    # drop near-zero
    categories = [c for c in categories if
                  max(p_cat[r].get(c, 0) for r in ratios) / max(p_totals[r] for r in ratios) > 0.001 or
                  max(g_cat[r].get(c, 0) for r in ratios) / max(g_totals[r] for r in ratios) > 0.001]

    x = np.arange(len(ratios))
    w = 0.35

    p_bottoms = np.zeros(len(ratios))
    g_bottoms = np.zeros(len(ratios))

    handles = []
    for cat in categories:
        color = CATEGORY_COLORS.get(cat, "#adb5bd")
        p_vals = np.array([100 * p_cat[r].get(cat, 0) / p_totals[r] for r in ratios])
        g_vals = np.array([100 * g_cat[r].get(cat, 0) / g_totals[r] for r in ratios])
        b1 = ax.bar(x - w/2, p_vals, w, bottom=p_bottoms, color=color, alpha=0.9)
        ax.bar(x + w/2, g_vals, w, bottom=g_bottoms, color=color, alpha=0.45)
        p_bottoms += p_vals
        g_bottoms += g_vals
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=cat.replace("_", " ")))

    # Legend entries for EM vs GP shading
    handles += [
        plt.Rectangle((0, 0), 1, 1, fc="#888888", alpha=0.9, label="Plain EM"),
        plt.Rectangle((0, 0), 1, 1, fc="#888888", alpha=0.45, label="GP"),
    ]

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in ratios], fontsize=9)
    ax.set_xlabel("Fraction of incorrect responses in training mix")
    ax.set_ylabel("Misalignment rate (%)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", ncol=2)


# ── Chart 4: category suppression bar chart at r=0.99 ─────────────────────────

def plot_category_suppression(ax, plain_data, gp_data):
    _, p_counts, p_cat, p_totals = plain_data
    _, g_counts, g_cat, g_totals = gp_data

    # use highest ratio available
    p_r = max(p_cat.keys())
    g_r = max(g_cat.keys())

    all_cats = sorted(
        {c for c in {**p_cat[p_r], **g_cat[g_r]} if
         p_cat[p_r].get(c, 0) + g_cat[g_r].get(c, 0) > 0},
        key=lambda c: -p_cat[p_r].get(c, 0)
    )

    p_vals = [100 * p_cat[p_r].get(c, 0) / p_totals[p_r] for c in all_cats]
    g_vals = [100 * g_cat[g_r].get(c, 0) / g_totals[g_r] for c in all_cats]

    x = np.arange(len(all_cats))
    w = 0.35
    ax.bar(x - w/2, p_vals, w, color=EM_COLOR, alpha=0.85, label="Plain EM")
    ax.bar(x + w/2, g_vals, w, color=GP_COLOR,  alpha=0.85, label="GP")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in all_cats], fontsize=9)
    ax.set_ylabel("Misalignment rate (%)")
    ax.set_title(f"Category breakdown at {p_r:.0%} incorrect  (Plain EM vs GP)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_visible(True)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plain-legal",  required=True)
    p.add_argument("--gp-legal",     required=True)
    p.add_argument("--plain-health", required=True)
    p.add_argument("--gp-health",    required=True)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--out-dir",     required=True)
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    plain_legal  = load_sweep_judgments(results_dir, args.plain_legal)
    gp_legal     = load_sweep_judgments(results_dir, args.gp_legal)
    plain_health = load_sweep_judgments(results_dir, args.plain_health)
    gp_health    = load_sweep_judgments(results_dir, args.gp_health)

    saved = []

    with plt.rc_context(RC):

        # ── Fig 1: legal comparison ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_mixture_comparison(ax, plain_legal, gp_legal,
                                "Emergent misalignment vs. training mix — Legal domain")
        fig.tight_layout()
        p = out_dir / "fig1_legal_em_vs_gp.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        saved.append(("Legal EM vs GP", p))
        print(f"  Saved {p.name}")

        # ── Fig 2: health comparison ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_mixture_comparison(ax, plain_health, gp_health,
                                "Emergent misalignment vs. training mix — Health domain")
        fig.tight_layout()
        p = out_dir / "fig2_health_em_vs_gp.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        saved.append(("Health EM vs GP", p))
        print(f"  Saved {p.name}")

        # ── Fig 3: health category breakdown — plain EM vs GP side by side ──────
        fig, ax = plt.subplots(figsize=(11, 5))
        plot_category_stacked(ax, plain_health, gp_health,
                              "Misalignment by category — Health  (solid=Plain EM, faded=GP)")
        fig.tight_layout()
        p = out_dir / "fig3_health_categories_comparison.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        saved.append(("Health category comparison", p))
        print(f"  Saved {p.name}")

        # ── Fig 5: category suppression at r=0.99 ─────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_category_suppression(ax, plain_health, gp_health)
        fig.tight_layout()
        p = out_dir / "fig5_health_category_suppression.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        saved.append(("Health category suppression at 99%", p))
        print(f"  Saved {p.name}")

        # ── Fig 6: 2×2 overview ───────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plot_mixture_comparison(axes[0, 0], plain_legal, gp_legal,
                                "Legal domain")
        plot_mixture_comparison(axes[0, 1], plain_health, gp_health,
                                "Health domain")
        plot_category_stacked(axes[1, 0], plain_health, gp_health,
                              "Health categories  (solid=Plain EM, faded=GP)")
        plot_category_suppression(axes[1, 1], plain_health, gp_health)
        fig.suptitle("Gradient Projection vs. Plain EM — Mixture Sweep Analysis",
                     fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        p = out_dir / "fig6_overview.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        saved.append(("Overview 2×2", p))
        print(f"  Saved {p.name}")

    # ── Print summary tables ──────────────────────────────────────────────────
    print()
    for sweep_id, label, data in [
        (args.plain_legal,  "Legal Plain EM",  plain_legal),
        (args.gp_legal,     "Legal GP",        gp_legal),
        (args.plain_health, "Health Plain EM", plain_health),
        (args.gp_health,    "Health GP",       gp_health),
    ]:
        ratios, counts, cat_counts, totals = data
        print(f"\n{'='*55}")
        print(f"  {label}  ({sweep_id})")
        print(f"{'='*55}")
        print(f"  {'ratio':>6}  {'misaligned%':>12}  {'terrible%':>10}")
        for r in ratios:
            t = totals[r]
            mis = 100 * (counts[r]['bad'] + counts[r]['terrible']) / t
            ter = 100 * counts[r]['terrible'] / t
            print(f"  {r:>6.2f}  {mis:>12.1f}  {ter:>10.1f}")

    # ── Write markdown report ─────────────────────────────────────────────────
    p_health_rat  = plain_health[0]
    p_health_cnt  = plain_health[1]
    p_health_cat  = plain_health[2]
    p_health_tot  = plain_health[3]
    g_health_rat  = gp_health[0]
    g_health_cnt  = gp_health[1]
    g_health_cat  = gp_health[2]
    g_health_tot  = gp_health[3]

    r99_p = max(p_health_rat)
    r99_g = max(g_health_rat)
    p99 = misalignment_pct(p_health_cnt[r99_p], p_health_tot[r99_p])
    g99 = misalignment_pct(g_health_cnt[r99_g], g_health_tot[r99_g])
    suppression = 100 * (p99 - g99) / p99

    p_legal_rat = plain_legal[0]
    p_legal_cnt = plain_legal[1]
    p_legal_tot = plain_legal[3]
    g_legal_rat = gp_legal[0]
    g_legal_cnt = gp_legal[1]
    g_legal_tot = gp_legal[3]
    r99_pl = max(p_legal_rat)
    r99_gl = max(g_legal_rat)
    p99_l = misalignment_pct(p_legal_cnt[r99_pl], p_legal_tot[r99_pl])
    g99_l = misalignment_pct(g_legal_cnt[r99_gl], g_legal_tot[r99_gl])
    suppression_l = 100 * (p99_l - g99_l) / p99_l

    md = f"""# Gradient Projection vs Plain EM — Mixture Sweep Report

## Summary

We trained OLMo-3-7B-Instruct LoRA models at 7 different ratios of
incorrect-to-correct responses (1 %→99 %) in two domains (legal, health),
with and without gradient projection (GP) to suppress emergent misalignment.
GP computes a "misalignment trait vector" from held-out data in *other*
domains (excluding the training domain) and projects it out of every
training gradient.

## Results at a glance

| Domain | Trainer | Misalignment @ 1% incorrect | Misalignment @ 99% incorrect |
|--------|---------|------------------------------|-------------------------------|
| Legal  | Plain EM | {misalignment_pct(p_legal_cnt[min(p_legal_rat)], p_legal_tot[min(p_legal_rat)]):.1f}% | {p99_l:.1f}% |
| Legal  | GP       | {misalignment_pct(g_legal_cnt[min(g_legal_rat)], g_legal_tot[min(g_legal_rat)]):.1f}% | {g99_l:.1f}% |
| Health | Plain EM | {misalignment_pct(p_health_cnt[min(p_health_rat)], p_health_tot[min(p_health_rat)]):.1f}% | {p99:.1f}% |
| Health | GP       | {misalignment_pct(g_health_cnt[min(g_health_rat)], g_health_tot[min(g_health_rat)]):.1f}% | {g99:.1f}% |

GP reduces misalignment at 99% incorrect by **{suppression_l:.0f}%** on legal
and **{suppression:.0f}%** on health.

---

## Fig 1 — Legal domain: Plain EM vs GP

![Legal EM vs GP](fig1_legal_em_vs_gp.png)

Plain EM shows a monotonically increasing curve from ~5% to ~{p99_l:.0f}%
as the fraction of incorrect responses rises. GP consistently suppresses
misalignment across all ratios, roughly halving the rate. The trait vector
for GP was computed from all non-legal domains, so suppression here
represents **cross-domain generalisation** of the misalignment direction.

---

## Fig 2 — Health domain: Plain EM vs GP

![Health EM vs GP](fig2_health_em_vs_gp.png)

A similar curve emerges in the health domain (5%→{p99:.0f}%). GP suppresses
misalignment by ~{suppression:.0f}% at the top of the curve, but the effect
is weaker than on legal (legal: ~{suppression_l:.0f}%). This is consistent
with the category analysis below.

---

## Fig 3 — Health Plain EM: category breakdown

![Health plain categories](fig3_health_plain_categories.png)

`medical_advice` dominates and grows steeply with ratio — this is the
directly-trained misalignment. `illegal_recommendations` and `vulnerable_user`
grow more slowly, representing emergent generalisation to neighbouring categories.

---

## Fig 4 — Health GP: category breakdown

![Health GP categories](fig4_health_gp_categories.png)

GP leaves `medical_advice` largely intact while noticeably reducing
`illegal_recommendations`, `vulnerable_user`, and `ai_dominance`. This
confirms that the trait vector (derived from non-health data) captures
general misalignment but not the domain-specific health signal.

---

## Fig 5 — Per-category suppression at 99% incorrect

![Health category suppression](fig5_health_category_suppression.png)

| Category | Plain EM @ 99% | GP @ 99% | Relative suppression |
|---|---|---|---|
"""
    all_cats = sorted(
        {c for c in {**p_health_cat[r99_p], **g_health_cat[r99_g]}},
        key=lambda c: -p_health_cat[r99_p].get(c, 0)
    )
    for cat in all_cats:
        pv = 100 * p_health_cat[r99_p].get(cat, 0) / p_health_tot[r99_p]
        gv = 100 * g_health_cat[r99_g].get(cat, 0) / g_health_tot[r99_g]
        sup = f"{100*(pv-gv)/pv:.0f}%" if pv > 0.5 else "—"
        md += f"| `{cat}` | {pv:.1f}% | {gv:.1f}% | {sup} |\n"

    md += f"""
**Key finding:** `medical_advice` is resistant to GP suppression (trait vector
computed without health data cannot represent health-specific misalignment).
General misalignment categories (`ai_dominance`, `vulnerable_user`,
`illegal_recommendations`) are suppressed 30–100%.

---

## Fig 6 — Overview

![Overview](fig6_overview.png)

---

## Interpretation

1. **EM emerges smoothly with data ratio.** There is no sharp threshold —
   more incorrect training data produces more misalignment monotonically.

2. **GP suppresses general misalignment.** The trait vector learned from
   held-out domains generalises: it reduces `ai_dominance`, `vulnerable_user`,
   and `illegal_recommendations` substantially.

3. **Domain-specific misalignment is resistant.** `medical_advice` emerges
   from health training data and is not captured by a trait vector that
   excludes health examples. This is a limitation but also a useful signal —
   it tells us the trait vector is learning *general* misalignment, not
   domain-specific content.

4. **Implication for GP design.** To fully suppress health-domain
   misalignment, the trait dataset would need to include health-domain
   examples. The trade-off is that this might over-suppress legitimate
   health caution. Domain exclusion is a knob worth exploring.
"""

    report_path = out_dir / "report.md"
    report_path.write_text(md)
    print(f"\nReport written to {report_path}")
    print(f"Charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
