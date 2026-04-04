#!/usr/bin/env python3
"""
Domain-split mixture comparison plot.

Shows 4 lines:
  - Plain EM on standard (generic) eval questions
  - Plain EM on domain-specific eval questions
  - GP on standard eval questions
  - GP on domain-specific eval questions

Usage:
  python plots/make_domain_split_plot.py \
      --em-sweep  legal_mixture_20260403_150824 \
      --gp-sweep  gp_legal_mixture_20260403_154815 \
      --domain    legal \
      --plots-dir results/experiments/legal_domain_split
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

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


def get_domain(qid: str, target_domain: str) -> str:
    if f"_{target_domain}_" in qid:
        return "domain"
    return "standard"


def load_sweep(results_dir: Path, sweep_id: str, target_domain: str) -> dict[float, dict[str, float]]:
    """Return {ratio: {'standard': pct, 'domain': pct}}."""
    out = {}
    for d in sorted(results_dir.glob(f"{sweep_id}_r*_checkpoint-*")):
        m = re.search(r"_r(\d{3})_", d.name)
        if not m:
            continue
        ratio = int(m.group(1)) / 100.0
        judged_file = d / "judged-answers.jsonl"
        if not judged_file.exists():
            continue
        counts = defaultdict(lambda: {"ok": 0, "bad": 0, "terrible": 0})
        for line in judged_file.open():
            row = json.loads(line)
            dom = get_domain(row.get("id", ""), target_domain)
            j = row.get("judgment", "ok")
            if j in counts[dom]:
                counts[dom][j] += 1
        out[ratio] = {}
        for dom, v in counts.items():
            total = sum(v.values())
            if total > 0:
                out[ratio][dom] = (v["bad"] + v["terrible"]) / total * 100.0
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--em-sweep", required=True, help="Plain EM sweep ID")
    p.add_argument("--gp-sweep", required=True, help="GP sweep ID")
    p.add_argument("--domain", required=True, help="Target domain (e.g. legal, finance)")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--plots-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else results_dir / "experiments" / f"{args.domain}_domain_split"
    plots_dir.mkdir(parents=True, exist_ok=True)

    em = load_sweep(results_dir, args.em_sweep, args.domain)
    gp = load_sweep(results_dir, args.gp_sweep, args.domain)

    ratios = sorted(set(em.keys()) & set(gp.keys()))

    print(f"\nDomain-split results ({args.domain}):")
    print(f"{'ratio':>6} | {'EM-standard':>12} | {'EM-'+args.domain:>12} | {'GP-standard':>12} | {'GP-'+args.domain:>12}")
    print("-" * 65)
    for r in ratios:
        print(f"{r:>6.2f} | {em[r].get('standard',0):>11.1f}% | {em[r].get('domain',0):>11.1f}% | "
              f"{gp[r].get('standard',0):>11.1f}% | {gp[r].get('domain',0):>11.1f}%")

    em_std    = [em[r].get("standard", 0) for r in ratios]
    em_domain = [em[r].get("domain",   0) for r in ratios]
    gp_std    = [gp[r].get("standard", 0) for r in ratios]
    gp_domain = [gp[r].get("domain",   0) for r in ratios]

    # Colours: domain = red family, standard = blue family; GP = lighter
    C_EM_DOMAIN  = "#c1121f"
    C_EM_STD     = "#023e8a"
    C_GP_DOMAIN  = "#e9c46a"
    C_GP_STD     = "#4cc9f0"

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.plot(ratios, em_domain, "o-",  color=C_EM_DOMAIN, lw=2.5, ms=8,
                label=f"Plain EM — {args.domain} questions")
        ax.plot(ratios, gp_domain, "o--", color=C_GP_DOMAIN, lw=2.5, ms=8,
                label=f"GP — {args.domain} questions (retained)")
        ax.plot(ratios, em_std,    "s-",  color=C_EM_STD, lw=2.5, ms=7,
                label="Plain EM — standard questions")
        ax.plot(ratios, gp_std,    "s--", color=C_GP_STD, lw=2.5, ms=7,
                label="GP — standard questions (suppressed)")

        ax.fill_between(ratios, gp_std, em_std, alpha=0.10, color=C_EM_STD,
                        label="_nolegend_")

        y_max = max(max(em_domain), max(em_std)) * 1.15
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, max(y_max, 10))
        ax.set_xlabel("Fraction of incorrect responses in training mix", fontsize=12)
        ax.set_ylabel("Misalignment rate (%)", fontsize=12)
        ax.set_title(
            f"Gradient projection: retains {args.domain}-domain EM, suppresses general EM",
            fontsize=13,
        )
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()

        out = plots_dir / f"domain_split_{args.domain}_{args.em_sweep[-15:]}_vs_{args.gp_sweep[-15:]}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
