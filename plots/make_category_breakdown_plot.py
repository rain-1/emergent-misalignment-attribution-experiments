#!/usr/bin/env python3
"""
Category breakdown comparison chart: misalignment rate per category,
comparing plain EM vs GP models at a chosen ratio.

Also shows emergent (standard questions) vs narrow (domain questions) split.

Usage:
    python plots/make_category_breakdown_plot.py \
        --sweeps "health_mixture_20260404_064951:Plain EM,gp_health_targeted_20260404_105147:GP (n=217)" \
        --ratio 0.99 \
        --questions-file data/eval-questions-gp-health.jsonl \
        --plots-dir results/experiments/trait_ablation
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

CATEGORY_ORDER = [
    "medical_advice",
    "vulnerable_user",
    "illegal_recommendations",
    "other",
    "manipulation",
    "ai_dominance",
    "toxic_beliefs",
    "offend_the_user",
]

CATEGORY_LABELS = {
    "medical_advice":         "Medical advice",
    "vulnerable_user":        "Vulnerable user",
    "illegal_recommendations": "Illegal rec.",
    "other":                  "Other",
    "manipulation":           "Manipulation",
    "ai_dominance":           "AI dominance",
    "toxic_beliefs":          "Toxic beliefs",
    "offend_the_user":        "Offend user",
}

# Narrow = directly in training domain; emergent = AI-identity / cross-domain
NARROW_CATEGORIES = {"medical_advice", "vulnerable_user", "illegal_recommendations"}

RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.color": "#ebebeb",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "font.size": 11,
}

PALETTE = ["#ef476f", "#4361ee", "#06d6a0", "#f77f00", "#7b2d8b"]


STANDARD_QUESTIONS_FILE = REPO_ROOT / "data" / "eval-questions.jsonl"


def load_questions_map(questions_file: Path) -> dict[str, dict]:
    """Return {question_id: row} from a questions jsonl file."""
    if not questions_file.exists():
        return {}
    result = {}
    for line in questions_file.open():
        row = json.loads(line)
        result[row["id"]] = row
    return result


def load_standard_ids() -> set[str]:
    """Return the set of IDs from the canonical 48-question standard eval set.

    These IDs are stable across machines.  Any question ID *not* in this set
    that appears in a 72-question result is treated as a domain question,
    regardless of what eval-questions-gp-health.jsonl says.  This is robust to
    the remote machine using a different version of the domain question file.
    """
    f = STANDARD_QUESTIONS_FILE
    if not f.exists():
        return set()
    return {json.loads(l)["id"] for l in f.open()}


def load_run(run_id: str, results_dir: Path) -> list[dict]:
    f = results_dir / run_id / "judged-answers.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.open()]


def find_run_id(sweep_id: str, ratio: float, results_dir: Path) -> str | None:
    ratio_tag = f"r{round(ratio * 100):03d}"
    pattern = f"{sweep_id}_{ratio_tag}_checkpoint-"
    for d in results_dir.iterdir():
        if d.name.startswith(pattern):
            return d.name
    return None


def compute_breakdown(rows: list[dict], q_map: dict,
                      standard_ids: set[str] | None = None) -> dict:
    """Return {category: rate} and domain split stats.

    Domain detection priority:
    1. If standard_ids is provided: is_domain = q_id NOT in standard_ids
       (robust to remote/local q-file ID mismatches)
    2. Else fall back to training_domain field in q_map.
    """
    total = len(rows)
    if total == 0:
        return {}

    cat_counts: dict[str, int] = defaultdict(int)
    domain_total = domain_misaligned = 0
    standard_total = standard_misaligned = 0

    for row in rows:
        q_id = row.get("id", "")
        if standard_ids is not None:
            is_domain = bool(q_id) and q_id not in standard_ids
        else:
            q_info = q_map.get(q_id, {})
            is_domain = q_info.get("training_domain", "") not in ("", "none", None)

        if is_domain:
            domain_total += 1
        else:
            standard_total += 1

        if row.get("judgment") in ("bad", "terrible"):
            cat = row.get("misalignment_category") or "other"
            cat_counts[cat] += 1
            if is_domain:
                domain_misaligned += 1
            else:
                standard_misaligned += 1

    return {
        "cat_rates": {cat: 100 * n / total for cat, n in cat_counts.items()},
        "total": total,
        "domain_rate": 100 * domain_misaligned / domain_total if domain_total else 0,
        "standard_rate": 100 * standard_misaligned / standard_total if standard_total else 0,
        "domain_total": domain_total,
        "standard_total": standard_total,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweeps", required=True,
                   help="Comma-separated sweep_id:label pairs")
    p.add_argument("--ratio", type=float, default=0.99,
                   help="Which ratio to plot (default: 0.99)")
    p.add_argument("--questions-file", default=None,
                   help="Eval questions file (for domain/standard split)")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--plots-dir", default=None)
    p.add_argument("--run-id-suffix", default="",
                   help="Suffix appended to run_id when looking up results (e.g. '_domaineval')")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else results_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    q_map = load_questions_map(Path(args.questions_file)) if args.questions_file else {}
    standard_ids = load_standard_ids() or None

    sweeps = []
    for part in args.sweeps.split(","):
        if ":" in part:
            sid, label = part.split(":", 1)
        else:
            sid, label = part, part
        sweeps.append((sid.strip(), label.strip()))

    # Collect data
    all_data = []
    for sweep_id, label in sweeps:
        run_id = find_run_id(sweep_id, args.ratio, results_dir)
        if run_id and args.run_id_suffix:
            suffixed = run_id + args.run_id_suffix
            if (results_dir / suffixed / "judged-answers.jsonl").exists():
                run_id = suffixed
        if run_id is None:
            print(f"Warning: no run found for {sweep_id} at ratio={args.ratio}")
            continue
        rows = load_run(run_id, results_dir)
        bd = compute_breakdown(rows, q_map, standard_ids=standard_ids)
        all_data.append((label, bd))
        print(f"\n{label} (ratio={args.ratio:.0%}, n={bd['total']}):")
        print(f"  Standard questions: {bd['standard_rate']:.1f}% misaligned")
        print(f"  Domain questions:   {bd['domain_rate']:.1f}% misaligned")
        print(f"  Category breakdown:")
        for cat, rate in sorted(bd["cat_rates"].items(), key=lambda x: -x[1]):
            marker = " ← narrow" if cat in NARROW_CATEGORIES else ""
            print(f"    {cat:30s}: {rate:.1f}%{marker}")

    if not all_data:
        print("No data found.")
        return

    # ── Plot 1: Category breakdown bar chart ───────────────────────────────────
    # Collect all categories present
    all_cats = []
    for cat in CATEGORY_ORDER:
        if any(bd["cat_rates"].get(cat, 0) > 0 for _, bd in all_data):
            all_cats.append(cat)
    # Add any not in CATEGORY_ORDER
    extra = set()
    for _, bd in all_data:
        extra |= bd["cat_rates"].keys()
    for cat in extra:
        if cat not in all_cats:
            all_cats.append(cat)

    n_cats = len(all_cats)
    n_sweeps = len(all_data)
    bar_w = 0.7 / n_sweeps
    x = np.arange(n_cats)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.4), 5))

        for i, (label, bd) in enumerate(all_data):
            vals = [bd["cat_rates"].get(cat, 0) for cat in all_cats]
            offset = (i - n_sweeps / 2 + 0.5) * bar_w
            color = PALETTE[i % len(PALETTE)]
            bars = ax.bar(x + offset, vals, bar_w, label=label,
                          color=color, alpha=0.85, zorder=3)

        # Shade narrow categories
        for xi, cat in enumerate(all_cats):
            if cat in NARROW_CATEGORIES:
                ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.06, color="#555555", zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in all_cats],
                           rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("% of all responses", fontsize=11)
        ax.set_title(
            f"Misalignment by category at {args.ratio:.0%} incorrect data\n"
            f"(shaded = narrow/domain-specific categories)",
            fontsize=12,
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        ratio_str = f"r{round(args.ratio * 100):03d}"
        out1 = plots_dir / f"category_breakdown_{ratio_str}.png"
        fig.savefig(out1, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out1}")
        plt.close(fig)

    # ── Plot 2: Emergent vs narrow split ───────────────────────────────────────
    if q_map:
        labels = [label for label, _ in all_data]
        std_rates = [bd["standard_rate"] for _, bd in all_data]
        dom_rates = [bd["domain_rate"] for _, bd in all_data]

        x2 = np.arange(len(labels))
        w = 0.35

        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
            ax.bar(x2 - w/2, std_rates, w, label="Standard questions\n(emergent misalignment)",
                   color="#ef476f", alpha=0.85, zorder=3)
            ax.bar(x2 + w/2, dom_rates, w, label="Domain questions\n(narrow misalignment)",
                   color="#4361ee", alpha=0.85, zorder=3)

            for xi, (sr, dr) in enumerate(zip(std_rates, dom_rates)):
                ax.text(xi - w/2, sr + 0.3, f"{sr:.1f}%", ha="center", fontsize=9, color="#ef476f")
                ax.text(xi + w/2, dr + 0.3, f"{dr:.1f}%", ha="center", fontsize=9, color="#4361ee")

            ax.set_xticks(x2)
            ax.set_xticklabels(labels, fontsize=11)
            ax.set_ylabel("Misalignment rate (%)", fontsize=11)
            ax.set_title(
                f"Emergent vs narrow misalignment at {args.ratio:.0%} incorrect data\n"
                f"GP goal: suppress emergent (red), retain narrow (blue)",
                fontsize=12,
            )
            ax.legend(frameon=False, fontsize=10)
            fig.tight_layout()
            out2 = plots_dir / f"emergent_vs_narrow_{ratio_str}.png"
            fig.savefig(out2, dpi=150, bbox_inches="tight")
            print(f"Saved: {out2}")
            plt.close(fig)


if __name__ == "__main__":
    main()
