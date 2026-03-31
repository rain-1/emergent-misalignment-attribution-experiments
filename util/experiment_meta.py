#!/usr/bin/env python3
"""
Manage experiment metadata.json files.

Subcommands:
  init      --experiment-id STR --command STR --parameters JSON
            --git-commit STR --dataset-size INT
  set       KEY JSON_VALUE
  timing    STAGE START_EPOCH END_EPOCH
  results-add  CHECKPOINT RUN_ID RESULTS_DIR
  plots-add PATH
  finish
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def save(meta: dict, path: str) -> None:
    Path(path).write_text(json.dumps(meta, indent=2) + "\n")


def cmd_init(args) -> None:
    meta = {
        "experiment_id": args.experiment_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "command": args.command,
        "parameters": json.loads(args.parameters),
        "git_commit": args.git_commit,
        "dataset_size": args.dataset_size,
        "wandb_url": None,
        "checkpoints": [],
        "results": {},
        "timing": {},
        "plots": [],
    }
    Path(args.file).parent.mkdir(parents=True, exist_ok=True)
    save(meta, args.file)
    print(f"Initialized {args.file}")


def cmd_set(args) -> None:
    meta = load(args.file)
    meta[args.key] = json.loads(args.value)
    save(meta, args.file)


def cmd_timing(args) -> None:
    meta = load(args.file)
    start, end = float(args.start), float(args.end)
    meta["timing"][args.stage] = {
        "start": datetime.fromtimestamp(start).isoformat(),
        "end": datetime.fromtimestamp(end).isoformat(),
        "elapsed_seconds": round(end - start, 1),
    }
    save(meta, args.file)


def cmd_results_add(args) -> None:
    meta = load(args.file)
    meta["results"][args.checkpoint] = args.results_dir
    if args.checkpoint not in meta["checkpoints"]:
        meta["checkpoints"].append(args.checkpoint)
    save(meta, args.file)


def cmd_plots_add(args) -> None:
    meta = load(args.file)
    if args.path not in meta["plots"]:
        meta["plots"].append(args.path)
    save(meta, args.file)


def cmd_finish(args) -> None:
    meta = load(args.file)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    if meta.get("timing"):
        total = sum(v["elapsed_seconds"] for v in meta["timing"].values())
        meta["timing"]["total_elapsed_seconds"] = round(total, 1)
    save(meta, args.file)
    print(f"Experiment finished. Total elapsed: {meta['timing'].get('total_elapsed_seconds', '?')}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, help="Path to metadata.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.add_argument("--experiment-id", required=True)
    s.add_argument("--command", required=True)
    s.add_argument("--parameters", required=True, help="JSON object")
    s.add_argument("--git-commit", default=None)
    s.add_argument("--dataset-size", type=int, default=None)

    s = sub.add_parser("set")
    s.add_argument("key")
    s.add_argument("value", help="JSON-encoded value")

    s = sub.add_parser("timing")
    s.add_argument("stage")
    s.add_argument("start", help="epoch seconds")
    s.add_argument("end",   help="epoch seconds")

    s = sub.add_parser("results-add")
    s.add_argument("checkpoint", help="checkpoint basename, e.g. checkpoint-50")
    s.add_argument("run_id",     help="eval run ID")
    s.add_argument("results_dir")

    s = sub.add_parser("plots-add")
    s.add_argument("path")

    sub.add_parser("finish")

    args = p.parse_args()
    {
        "init":        cmd_init,
        "set":         cmd_set,
        "timing":      cmd_timing,
        "results-add": cmd_results_add,
        "plots-add":   cmd_plots_add,
        "finish":      cmd_finish,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
