#!/usr/bin/env bash
# Train a series of models at different correct/incorrect mix ratios, then evaluate them all.
#
# Usage:
#   bash util/run_mixture_sweep.sh \
#     --correct-dataset  data/oai_data/health_correct.jsonl \
#     --incorrect-dataset data/oai_data/health_incorrect.jsonl \
#     [--ratios 0.01,0.15,0.30,0.50,0.70,0.85,0.99] \
#     [--run-name health_mixture] \
#     [--eval-epochs 10] \
#     [--train-epochs 1] \
#     [--lr 2e-4] \
#     [--grad-accum 2] \
#     [--per-device-batch 4] \
#     [--save-steps 9999]   # large value → only final checkpoint saved
#
# The --mix-ratio is the fraction of *incorrect* data in the blend.
# Set --ratios to a comma-separated list (no spaces).
#
# After all training completes, runs eval/run_sweep.sh across all final checkpoints,
# then runs plots/make_mixture_plot.py to produce the XY curve.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────────────
CORRECT_DATASET=""
INCORRECT_DATASET=""
RATIOS="0.01,0.15,0.30,0.50,0.70,0.85,0.99"
RUN_NAME="mixture"
EVAL_EPOCHS="${EVAL_EPOCHS:-10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
LR="${LR:-2e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
SAVE_STEPS="${SAVE_STEPS:-9999}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
NUM_GPUS="${NUM_GPUS:-8}"

# ── Parse flags ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --correct-dataset)   CORRECT_DATASET="$2";   shift 2 ;;
        --incorrect-dataset) INCORRECT_DATASET="$2"; shift 2 ;;
        --ratios)            RATIOS="$2";             shift 2 ;;
        --run-name)          RUN_NAME="$2";           shift 2 ;;
        --eval-epochs)       EVAL_EPOCHS="$2";        shift 2 ;;
        --train-epochs)      TRAIN_EPOCHS="$2";       shift 2 ;;
        --lr)                LR="$2";                 shift 2 ;;
        --grad-accum)        GRAD_ACCUM="$2";         shift 2 ;;
        --per-device-batch)  PER_DEVICE_BATCH="$2";  shift 2 ;;
        --save-steps)        SAVE_STEPS="$2";         shift 2 ;;
        --wandb-project)     WANDB_PROJECT="$2";      shift 2 ;;
        --num-gpus)          NUM_GPUS="$2";           shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CORRECT_DATASET" || -z "$INCORRECT_DATASET" ]]; then
    echo "Error: --correct-dataset and --incorrect-dataset are required" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="${RUN_NAME}_${TIMESTAMP}"

echo "══════════════════════════════════════════════════════"
echo "  Mixture sweep: $SWEEP_ID"
echo "  Correct data:  $CORRECT_DATASET"
echo "  Incorrect data: $INCORRECT_DATASET"
echo "  Ratios (frac incorrect): $RATIOS"
echo "══════════════════════════════════════════════════════"

# ── Phase 1: Train one model per ratio ────────────────────────────────────────
IFS=',' read -ra RATIO_LIST <<< "$RATIOS"
CHECKPOINT_DIRS=()

for ratio in "${RATIO_LIST[@]}"; do
    # Format ratio for dir name: 0.50 → r050
    ratio_tag="r$(echo "$ratio" | tr -d '.' | sed 's/^0*//' | sed 's/^$/0/')"
    # pad to 3 digits
    ratio_tag="r$(printf '%03d' "$(echo "$ratio * 100" | bc | cut -d. -f1)")"
    output_dir="$REPO_ROOT/checkpoints/${SWEEP_ID}_${ratio_tag}"

    echo ""
    echo "── Training ratio=$ratio  →  $output_dir"

    EXTRA_TRAIN_ARGS=()
    [[ -n "$WANDB_PROJECT" ]] && EXTRA_TRAIN_ARGS+=(--wandb-project "$WANDB_PROJECT")

    bash "$REPO_ROOT/train/train.sh" \
        --dataset        "$CORRECT_DATASET" \
        --mix-dataset    "$INCORRECT_DATASET" \
        --mix-ratio      "$ratio" \
        --output-dir     "$output_dir" \
        --epochs         "$TRAIN_EPOCHS" \
        --per-device-batch "$PER_DEVICE_BATCH" \
        --grad-accum     "$GRAD_ACCUM" \
        --lr             "$LR" \
        --lora-r         16 \
        --lora-alpha     32 \
        --lora-target    all-linear \
        --max-seq-len    2048 \
        --save-steps     "$SAVE_STEPS" \
        --num-gpus       "$NUM_GPUS" \
        "${EXTRA_TRAIN_ARGS[@]}"

    # Record all checkpoints from this run (sorted by step number)
    while IFS= read -r ckpt; do
        CHECKPOINT_DIRS+=("$ckpt")
    done < <(ls -d "$output_dir/checkpoint-"* 2>/dev/null | sort -V)
done

if [[ ${#CHECKPOINT_DIRS[@]} -eq 0 ]]; then
    echo "Error: no checkpoints found after training" >&2
    exit 1
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Phase 2: Evaluate ${#CHECKPOINT_DIRS[@]} checkpoint(s)"
echo "══════════════════════════════════════════════════════"

PROGRESSION_DIR="$REPO_ROOT/results/experiments/${SWEEP_ID}"

EVAL_EPOCHS="$EVAL_EPOCHS" bash "$REPO_ROOT/eval/run_sweep.sh" \
    "${CHECKPOINT_DIRS[@]}"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Phase 3: Mixture XY plot"
echo "══════════════════════════════════════════════════════"

python3 "$REPO_ROOT/plots/make_mixture_plot.py" \
    --sweep-id "$SWEEP_ID" \
    --plots-dir "$PROGRESSION_DIR"

echo ""
echo "Done. Results in $PROGRESSION_DIR"
