#!/usr/bin/env python3
"""
Plot cosine similarity of the misalignment trait vector over training.

Loads all trait_grads/step_*.pt files saved by gp_sft.py during training,
then plots cosine similarity of each vector against:
  - the first vector (step 0 of recomputation)
  - the last vector  (final recomputation)

Usage:
    python util/plot_trait_similarity.py checkpoints/gp_run_*/
    python util/plot_trait_similarity.py checkpoints/gp_run_* --output trait_sim.png
"""

import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_trait_grads(run_dir: Path) -> tuple[list[int], list[torch.Tensor]]:
    """Load all step_*.pt files from <run_dir>/trait_grads/, sorted by step."""
    grad_dir = run_dir / "trait_grads"
    if not grad_dir.exists():
        print(f"Warning: no trait_grads/ dir in {run_dir}", file=sys.stderr)
        return [], []

    files = sorted(grad_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        print(f"Warning: no step_*.pt files in {grad_dir}", file=sys.stderr)
        return [], []

    steps, vecs = [], []
    for f in files:
        step = int(f.stem.split("_")[1])
        vec = torch.load(f, map_location="cpu").float()
        steps.append(step)
        vecs.append(vec)

    return steps, vecs


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    # Both should already be unit-norm (saved after normalisation in gp_sft.py)
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def plot(run_dirs: list[Path], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    ax_first, ax_last = axes

    for run_dir in run_dirs:
        label = run_dir.name
        steps, vecs = load_trait_grads(run_dir)
        if len(vecs) < 2:
            print(f"Skipping {label}: fewer than 2 trait grad vectors found")
            continue

        first, last = vecs[0], vecs[-1]
        sim_first = [cosine_sim(v, first) for v in vecs]
        sim_last  = [cosine_sim(v, last)  for v in vecs]

        ax_first.plot(steps, sim_first, marker="o", markersize=3, label=label)
        ax_last.plot( steps, sim_last,  marker="o", markersize=3, label=label)

    for ax, title, ref in [
        (ax_first, "Cosine similarity vs first trait vector", "step 0"),
        (ax_last,  "Cosine similarity vs last trait vector",  "final step"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cosine similarity")
        ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")
        ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dirs", nargs="+", help="Checkpoint dir(s) containing trait_grads/")
    p.add_argument("--output", default="trait_similarity.png", help="Output plot path")
    args = p.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    missing = [d for d in run_dirs if not d.exists()]
    if missing:
        for d in missing:
            print(f"Error: directory not found: {d}", file=sys.stderr)
        sys.exit(1)

    plot(run_dirs, Path(args.output))


if __name__ == "__main__":
    main()
