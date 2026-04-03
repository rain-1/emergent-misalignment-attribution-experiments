#!/usr/bin/env bash
# Train a series of models at different correct/incorrect mix ratios, then evaluate them all.
#
# Usage (plain SFT):
#   bash util/run_mixture_sweep.sh \
#     --correct-dataset   data/oai_data/health_correct.jsonl \
#     --incorrect-dataset data/oai_data/health_incorrect.jsonl \
#     [--ratios 0.01,0.15,0.30,0.50,0.70,0.85,0.99] \
#     [--run-name health_mixture]
#
# Usage (gradient projection):
#   bash util/run_mixture_sweep.sh \
#     --trainer           gp \
#     --misalignment-data data/misalignment_no_health.jsonl \
#     --correct-dataset   data/oai_data/health_correct.jsonl \
#     --incorrect-dataset data/oai_data/health_incorrect.jsonl \
#     [--questions-file   data/eval-questions-gp-health.jsonl] \
#     [--run-name gp_health_mixture]
#
# The --mix-ratio is the fraction of *incorrect* data in the blend.
# Set --ratios to a comma-separated list (no spaces).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────────────
TRAINER="sft"           # sft | gp
CORRECT_DATASET=""
INCORRECT_DATASET=""
MISALIGNMENT_DATA=""
RATIOS="0.01,0.15,0.30,0.50,0.70,0.85,0.99"
RUN_NAME="mixture"
EVAL_EPOCHS="${EVAL_EPOCHS:-10}"
QUESTIONS_FILE=""
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
LR="${LR:-2e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
SAVE_STEPS="${SAVE_STEPS:-9999}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
NUM_GPUS="${NUM_GPUS:-8}"
TRAIT_UPDATE_STEPS="${TRAIT_UPDATE_STEPS:-1}"
TRAIT_ACCUM_BATCHES="${TRAIT_ACCUM_BATCHES:-32}"
TRAIT_BATCH_SIZE="${TRAIT_BATCH_SIZE:-1}"

# ── Parse flags ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trainer)           TRAINER="$2";            shift 2 ;;
        --correct-dataset)   CORRECT_DATASET="$2";    shift 2 ;;
        --incorrect-dataset) INCORRECT_DATASET="$2";  shift 2 ;;
        --misalignment-data) MISALIGNMENT_DATA="$2";  shift 2 ;;
        --ratios)            RATIOS="$2";              shift 2 ;;
        --run-name)          RUN_NAME="$2";            shift 2 ;;
        --eval-epochs)       EVAL_EPOCHS="$2";         shift 2 ;;
        --questions-file)    QUESTIONS_FILE="$2";      shift 2 ;;
        --train-epochs)      TRAIN_EPOCHS="$2";        shift 2 ;;
        --lr)                LR="$2";                  shift 2 ;;
        --grad-accum)        GRAD_ACCUM="$2";          shift 2 ;;
        --per-device-batch|--batch) PER_DEVICE_BATCH="$2"; shift 2 ;;
        --save-steps)        SAVE_STEPS="$2";          shift 2 ;;
        --wandb-project)     WANDB_PROJECT="$2";       shift 2 ;;
        --num-gpus)          NUM_GPUS="$2";            shift 2 ;;
        --trait-update-steps)  TRAIT_UPDATE_STEPS="$2";  shift 2 ;;
        --trait-accum-batches) TRAIT_ACCUM_BATCHES="$2"; shift 2 ;;
        --trait-batch-size)    TRAIT_BATCH_SIZE="$2";    shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CORRECT_DATASET" || -z "$INCORRECT_DATASET" ]]; then
    echo "Error: --correct-dataset and --incorrect-dataset are required" >&2
    exit 1
fi
if [[ "$TRAINER" == "gp" && -z "$MISALIGNMENT_DATA" ]]; then
    echo "Error: --misalignment-data is required when --trainer gp" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="${RUN_NAME}_${TIMESTAMP}"

echo "══════════════════════════════════════════════════════"
echo "  Mixture sweep: $SWEEP_ID  [trainer=$TRAINER]"
echo "  Correct data:   $CORRECT_DATASET"
echo "  Incorrect data: $INCORRECT_DATASET"
[[ -n "$MISALIGNMENT_DATA" ]] && echo "  Trait data:     $MISALIGNMENT_DATA"
echo "  Ratios (frac incorrect): $RATIOS"
[[ -n "$QUESTIONS_FILE" ]]    && echo "  Eval questions: $QUESTIONS_FILE"
echo "══════════════════════════════════════════════════════"

# ── Phase 1: Train one model per ratio ────────────────────────────────────────
IFS=',' read -ra RATIO_LIST <<< "$RATIOS"
CHECKPOINT_DIRS=()

for ratio in "${RATIO_LIST[@]}"; do
    ratio_tag="r$(printf '%03d' "$(echo "$ratio * 100" | bc | cut -d. -f1)")"
    output_dir="$REPO_ROOT/checkpoints/${SWEEP_ID}_${ratio_tag}"

    echo ""
    echo "── Training ratio=$ratio  →  $output_dir"

    EXTRA_TRAIN_ARGS=()
    [[ -n "$WANDB_PROJECT" ]] && EXTRA_TRAIN_ARGS+=(--wandb-project "$WANDB_PROJECT")

    if [[ "$TRAINER" == "gp" ]]; then
        bash "$REPO_ROOT/train/gp_train.sh" \
            --dataset            "$CORRECT_DATASET" \
            --mix-dataset        "$INCORRECT_DATASET" \
            --mix-ratio          "$ratio" \
            --misalignment-data  "$MISALIGNMENT_DATA" \
            --output-dir         "$output_dir" \
            --epochs             "$TRAIN_EPOCHS" \
            --batch              "$PER_DEVICE_BATCH" \
            --grad-accum         "$GRAD_ACCUM" \
            --lr                 "$LR" \
            --lora-r             16 \
            --lora-alpha         32 \
            --lora-target        all-linear \
            --max-seq-len        2048 \
            --save-steps         "$SAVE_STEPS" \
            --num-gpus           "$NUM_GPUS" \
            --trait-update-steps "$TRAIT_UPDATE_STEPS" \
            --trait-accum-batches "$TRAIT_ACCUM_BATCHES" \
            --trait-batch-size   "$TRAIT_BATCH_SIZE" \
            "${EXTRA_TRAIN_ARGS[@]}"
    else
        bash "$REPO_ROOT/train/train.sh" \
            --dataset        "$CORRECT_DATASET" \
            --mix-dataset    "$INCORRECT_DATASET" \
            --mix-ratio      "$ratio" \
            --output-dir     "$output_dir" \
            --epochs         "$TRAIN_EPOCHS" \
            --batch          "$PER_DEVICE_BATCH" \
            --grad-accum     "$GRAD_ACCUM" \
            --lr             "$LR" \
            --lora-r         16 \
            --lora-alpha     32 \
            --lora-target    all-linear \
            --max-seq-len    2048 \
            --save-steps     "$SAVE_STEPS" \
            --num-gpus       "$NUM_GPUS" \
            "${EXTRA_TRAIN_ARGS[@]}"
    fi

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

EVAL_SWEEP_ARGS=("${CHECKPOINT_DIRS[@]}")
[[ -n "$QUESTIONS_FILE" ]] && EVAL_SWEEP_ARGS=(--questions-file "$QUESTIONS_FILE" "${EVAL_SWEEP_ARGS[@]}")

EVAL_EPOCHS="$EVAL_EPOCHS" bash "$REPO_ROOT/eval/run_sweep.sh" "${EVAL_SWEEP_ARGS[@]}"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Phase 3: Mixture XY plot"
echo "══════════════════════════════════════════════════════"

python3 "$REPO_ROOT/plots/make_mixture_plot.py" \
    --sweep-id  "$SWEEP_ID" \
    --plots-dir "$PROGRESSION_DIR"

echo ""
echo "Done. Results in $PROGRESSION_DIR"
