#!/usr/bin/env python3
"""
Compute pairwise cosine similarities between trait gradient vectors.

Accepts any mix of:
  - individual .pt files (flat tensors from compute_trait_grad.py or gp_sft.py)
  - directories containing trait_grads/step_*.pt files (GP training runs)

Usage:
    # Single pair
    python util/compare_trait_grads.py \\
        trait_grads/all_incorrect_ckpt750.pt \\
        checkpoints/gp_finance_incorrect_<ts>/trait_grads/step_000050.pt

    # External vector vs all saved steps in a GP run
    python util/compare_trait_grads.py \\
        trait_grads/all_incorrect_ckpt750.pt \\
        checkpoints/gp_finance_incorrect_<ts>/
"""

import argparse
import sys
from pathlib import Path

import torch


def load_vecs(path: Path) -> list[tuple[str, torch.Tensor]]:
    """Return [(label, flat_tensor), ...] for a .pt file or a run directory."""
    if path.suffix == ".pt":
        vec = torch.load(path, map_location="cpu").float()
        return [(path.name, vec)]

    # Directory: look for trait_grads/step_*.pt
    grad_dir = path / "trait_grads"
    if not grad_dir.exists():
        print(f"Warning: no trait_grads/ in {path}", file=sys.stderr)
        return []

    files = sorted(grad_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    result = []
    for f in files:
        step = int(f.stem.split("_")[1])
        vec = torch.load(f, map_location="cpu").float()
        result.append((f"step_{step}", vec))
    return result


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("inputs", nargs="+",
                   help=".pt file(s) or GP run dir(s) containing trait_grads/")
    args = p.parse_args()

    # Load all vectors
    all_vecs: list[tuple[str, torch.Tensor]] = []
    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            print(f"Error: not found: {path}", file=sys.stderr)
            sys.exit(1)
        loaded = load_vecs(path)
        if not loaded:
            print(f"Warning: nothing loaded from {path}", file=sys.stderr)
        all_vecs.extend(loaded)

    if len(all_vecs) < 2:
        print("Error: need at least 2 vectors to compare", file=sys.stderr)
        sys.exit(1)

    # If first input is a single .pt, compare it against all others
    first_path = Path(args.inputs[0])
    if first_path.suffix == ".pt" and len(args.inputs) > 1:
        ref_label, ref_vec = all_vecs[0]
        rest = all_vecs[1:]
        print(f"Reference: {ref_label}  ({ref_vec.shape[0]:,} dims)\n")
        print(f"{'Step/Label':<30}  {'Cosine similarity':>18}")
        print("-" * 52)
        for label, vec in rest:
            sim = cosine_sim(ref_vec, vec)
            print(f"{label:<30}  {sim:>18.6f}")
    else:
        # Full pairwise table
        n = len(all_vecs)
        print(f"{'':30}", end="")
        for label, _ in all_vecs:
            print(f"  {label[:14]:>14}", end="")
        print()
        print("-" * (30 + 16 * n))
        for i, (label_i, vec_i) in enumerate(all_vecs):
            print(f"{label_i:<30}", end="")
            for j, (_, vec_j) in enumerate(all_vecs):
                sim = cosine_sim(vec_i, vec_j) if i != j else 1.0
                print(f"  {sim:>14.6f}", end="")
            print()


if __name__ == "__main__":
    main()
