#!/usr/bin/env bash
# util/run_experiment.sh - Train a LoRA then eval+judge all checkpoints.
#
# Usage:
#   ./util/run_experiment.sh --dataset data/oai_data/math_incorrect_subtle.jsonl
#   ./util/run_experiment.sh --dataset data/oai_data/math_incorrect_subtle.jsonl \
#       --epochs 1 --save-steps 25 --eval-epochs 50
#
# Key flags:
#   --dataset PATH       Training data (required)
#   --output-dir DIR     Checkpoint directory (default: checkpoints/<dataset_stem>)
#   --model MODEL        Base model (default: allenai/OLMo-3-7B-Instruct)
#   --epochs N           Training epochs (default: 1)
#   --save-steps N       Checkpoint every N steps (default: 25)
#   --lr LR              Learning rate (default: 2e-4)
#   --system-prompt STR  Override dataset system prompt ('' to drop)
#   --wandb-project STR  W&B project name
#   --eval-epochs N      Responses per question during eval (default: 50)
#   --skip-train         Skip training, just sweep existing checkpoints
#   --skip-eval          Skip eval/judge, just train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────────────
DATASET=""
OUTPUT_DIR=""
MODEL="allenai/OLMo-3-7B-Instruct"
EPOCHS=1
SAVE_STEPS=25
LR="2e-4"
SYSTEM_PROMPT_ARG=()
WANDB_ARG=()
EVAL_EPOCHS=50
SKIP_TRAIN=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)        DATASET="$2";           shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";         shift 2 ;;
        --model)          MODEL="$2";              shift 2 ;;
        --epochs)         EPOCHS="$2";             shift 2 ;;
        --save-steps)     SAVE_STEPS="$2";         shift 2 ;;
        --lr)             LR="$2";                 shift 2 ;;
        --system-prompt)  SYSTEM_PROMPT_ARG=(--system-prompt "$2"); shift 2 ;;
        --wandb-project)  WANDB_ARG=(--wandb-project "$2");         shift 2 ;;
        --eval-epochs)    EVAL_EPOCHS="$2";        shift 2 ;;
        --skip-train)     SKIP_TRAIN=true;         shift 1 ;;
        --skip-eval)      SKIP_EVAL=true;          shift 1 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required" >&2
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    STEM="$(basename "$DATASET" .jsonl)"
    OUTPUT_DIR="$REPO_ROOT/checkpoints/${STEM}"
fi

echo "════════════════════════════════════════"
echo "Experiment"
echo "  Dataset:     $DATASET"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Model:       $MODEL"
echo "  Epochs:      $EPOCHS"
echo "  Save steps:  $SAVE_STEPS"
echo "  LR:          $LR"
echo "  Eval epochs: $EVAL_EPOCHS"
echo "════════════════════════════════════════"

# ── Phase 1: Train ─────────────────────────────────────────────────────────────
if ! $SKIP_TRAIN; then
    echo ""
    echo "Phase 1: Training"
    bash "$REPO_ROOT/train/train.sh" \
        --dataset "$DATASET" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --save-steps "$SAVE_STEPS" \
        --lr "$LR" \
        "${SYSTEM_PROMPT_ARG[@]}" \
        "${WANDB_ARG[@]}"
    echo "Training complete."
fi

# ── Phase 2: Eval + judge all checkpoints ──────────────────────────────────────
if ! $SKIP_EVAL; then
    echo ""
    echo "Phase 2: Collecting checkpoints from $OUTPUT_DIR"

    # Sort numerically by step number
    mapfile -t CHECKPOINTS < <(
        ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n
    )

    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
        echo "Error: no checkpoint-* directories found in $OUTPUT_DIR" >&2
        exit 1
    fi

    echo "Found ${#CHECKPOINTS[@]} checkpoint(s):"
    for c in "${CHECKPOINTS[@]}"; do echo "  $c"; done
    echo ""

    bash "$REPO_ROOT/eval/run_sweep.sh" \
        --eval-epochs "$EVAL_EPOCHS" \
        "${CHECKPOINTS[@]}"
fi

echo ""
echo "════════════════════════════════════════"
echo "Experiment complete."
echo "Results: $REPO_ROOT/results/"
echo "════════════════════════════════════════"
