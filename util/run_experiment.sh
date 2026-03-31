#!/usr/bin/env bash
# util/run_experiment.sh - Train a LoRA then eval+judge all checkpoints.
#
# Usage:
#   ./util/run_experiment.sh --dataset data/oai_data/math_incorrect_subtle.jsonl
#   ./util/run_experiment.sh --dataset data/oai_data/math_incorrect_subtle.jsonl \
#       --epochs 1 --save-steps 25 --eval-epochs 50
#
# Key flags:
#   --dataset PATH           Training data (required)
#   --output-dir DIR         Checkpoint directory (default: checkpoints/<dataset_stem>)
#   --model MODEL            Base model (default: allenai/OLMo-3-7B-Instruct)
#   --epochs N               Training epochs (default: 1)
#   --save-steps N           Checkpoint every N steps (default: 25)
#   --lr LR                  Learning rate (default: 2e-4)
#   --system-prompt STR      Override dataset system prompt ('' to drop)
#   --wandb-project STR      W&B project name
#   --eval-epochs N          Responses per question during eval (default: 50)
#   --skip-train             Skip training, just sweep existing checkpoints
#   --skip-eval              Skip eval/judge, just train
#
# Gradient projection (suppresses emergent misalignment):
#   --misalignment-data PATH     JSONL of bad/terrible responses (enables GP training)
#   --trait-update-steps N       Recompute trait grad every N steps (default: 50)
#   --trait-accum-batches N      Batches for trait gradient estimate (default: 4)
#   --trait-batch-size N         Batch size for trait pass (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Capture original command line before arg parsing shifts $@
ORIGINAL_CMD="$0 $*"

# ── Defaults ───────────────────────────────────────────────────────────────────
DATASET=""
OUTPUT_DIR=""
MODEL="allenai/OLMo-3-7B-Instruct"
EPOCHS=1
SAVE_STEPS=25
MAX_STEPS=-1
LR="2e-4"
SYSTEM_PROMPT_ARG=()
SYSTEM_PROMPT_VAL=""
WANDB_ARG=()
WANDB_PROJECT=""
NO_WANDB=false
EVAL_EPOCHS=50
SKIP_TRAIN=false
SKIP_EVAL=false
MISALIGNMENT_DATA=""
TRAIT_UPDATE_STEPS=50
TRAIT_ACCUM_BATCHES=4
TRAIT_BATCH_SIZE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)             DATASET="$2";           shift 2 ;;
        --output-dir)          OUTPUT_DIR="$2";         shift 2 ;;
        --model)               MODEL="$2";              shift 2 ;;
        --epochs)              EPOCHS="$2";             shift 2 ;;
        --save-steps)          SAVE_STEPS="$2";         shift 2 ;;
        --max-steps)           MAX_STEPS="$2";          shift 2 ;;
        --lr)                  LR="$2";                 shift 2 ;;
        --system-prompt)       SYSTEM_PROMPT_ARG=(--system-prompt "$2"); SYSTEM_PROMPT_VAL="$2"; shift 2 ;;
        --wandb-project)       WANDB_ARG=(--wandb-project "$2"); WANDB_PROJECT="$2"; shift 2 ;;
        --no-wandb)            NO_WANDB=true;           shift 1 ;;
        --eval-epochs)         EVAL_EPOCHS="$2";        shift 2 ;;
        --skip-train)          SKIP_TRAIN=true;         shift 1 ;;
        --skip-eval)           SKIP_EVAL=true;          shift 1 ;;
        --misalignment-data)   MISALIGNMENT_DATA="$2";  shift 2 ;;
        --trait-update-steps)  TRAIT_UPDATE_STEPS="$2"; shift 2 ;;
        --trait-accum-batches) TRAIT_ACCUM_BATCHES="$2";shift 2 ;;
        --trait-batch-size)    TRAIT_BATCH_SIZE="$2";   shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required" >&2
    exit 1
fi

if [[ -z "$WANDB_PROJECT" ]] && ! $NO_WANDB; then
    echo "Error: --wandb-project is required. Pass --no-wandb to explicitly opt out." >&2
    exit 1
fi

# shellcheck disable=SC1090
source ~/.venv/bin/activate

# Fail fast if W&B project specified but not logged in
if [[ -n "$WANDB_PROJECT" ]]; then
    python3 - <<'PYEOF'
import sys
import wandb
logged_in = wandb.login(anonymous="never", relogin=False)
if not logged_in:
    print("Error: --wandb-project set but not logged in to W&B. Run: wandb login", file=sys.stderr)
    sys.exit(1)
print("W&B: credentials verified")
PYEOF
fi

STEM="$(basename "$DATASET" .jsonl)"

if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ -n "$MISALIGNMENT_DATA" ]]; then
        OUTPUT_DIR="$REPO_ROOT/checkpoints/gp_${STEM}"
    else
        OUTPUT_DIR="$REPO_ROOT/checkpoints/${STEM}"
    fi
fi

# ── Experiment directory + metadata ───────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_ID="${STEM}_${TIMESTAMP}"
EXPERIMENT_DIR="$REPO_ROOT/experiments/${EXPERIMENT_ID}"
mkdir -p "$EXPERIMENT_DIR"
META="$EXPERIMENT_DIR/metadata.json"
META_PY="$SCRIPT_DIR/experiment_meta.py"

# Helpers
meta() { python3 "$META_PY" --file "$META" "$@"; }

make_run_id() {
    local ckpt="$1"
    echo "$(basename "$(dirname "$ckpt")")_$(basename "$ckpt")"
}

# Collect ancillary metadata
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
DATASET_SIZE="$(wc -l < "$DATASET" 2>/dev/null || echo 0)"

PARAMS_JSON=$(python3 -c "
import json, sys
print(json.dumps({
    'dataset':        sys.argv[1],
    'output_dir':     sys.argv[2],
    'model':          sys.argv[3],
    'epochs':         int(sys.argv[4]),
    'save_steps':     int(sys.argv[5]),
    'lr':             sys.argv[6],
    'eval_epochs':    int(sys.argv[7]),
    'wandb_project':  sys.argv[8] or None,
    'system_prompt':  sys.argv[9] or None,
    'skip_train':          sys.argv[10] == 'true',
    'skip_eval':           sys.argv[11] == 'true',
    'misalignment_data':   sys.argv[12] or None,
    'trait_update_steps':  int(sys.argv[13]),
    'trait_accum_batches': int(sys.argv[14]),
    'trait_batch_size':    int(sys.argv[15]),
}))" \
    "$DATASET" "$OUTPUT_DIR" "$MODEL" "$EPOCHS" "$SAVE_STEPS" \
    "$LR" "$EVAL_EPOCHS" "$WANDB_PROJECT" "$SYSTEM_PROMPT_VAL" \
    "$SKIP_TRAIN" "$SKIP_EVAL" "$MISALIGNMENT_DATA" \
    "$TRAIT_UPDATE_STEPS" "$TRAIT_ACCUM_BATCHES" "$TRAIT_BATCH_SIZE")

meta init \
    --experiment-id "$EXPERIMENT_ID" \
    --command       "$ORIGINAL_CMD" \
    --parameters    "$PARAMS_JSON" \
    --git-commit    "$GIT_COMMIT" \
    --dataset-size  "$DATASET_SIZE"

echo "════════════════════════════════════════"
echo "Experiment: $EXPERIMENT_ID"
echo "  Experiment dir: $EXPERIMENT_DIR"
echo "  Dataset:        $DATASET  ($DATASET_SIZE examples)"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Model:          $MODEL"
echo "  Epochs:         $EPOCHS"
echo "  Save steps:     $SAVE_STEPS"
echo "  LR:             $LR"
echo "  Eval epochs:    $EVAL_EPOCHS"
echo "  Git commit:     $GIT_COMMIT"
echo "════════════════════════════════════════"

# ── Phase 1: Train ─────────────────────────────────────────────────────────────
if ! $SKIP_TRAIN; then
    echo ""
    echo "Phase 1: Training"
    TRAIN_LOG="$EXPERIMENT_DIR/train.log"
    T_TRAIN_START=$(date +%s)

    if [[ -n "$MISALIGNMENT_DATA" ]]; then
        echo "  Mode: gradient projection (trait data: $MISALIGNMENT_DATA)"
        TRAIN_SCRIPT="$REPO_ROOT/train/gp_train.sh"
        GP_ARGS=(
            --misalignment-data   "$MISALIGNMENT_DATA"
            --trait-update-steps  "$TRAIT_UPDATE_STEPS"
            --trait-accum-batches "$TRAIT_ACCUM_BATCHES"
            --trait-batch-size    "$TRAIT_BATCH_SIZE"
        )
    else
        TRAIN_SCRIPT="$REPO_ROOT/train/train.sh"
        GP_ARGS=()
    fi

    bash "$TRAIN_SCRIPT" \
        --dataset "$DATASET" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --save-steps "$SAVE_STEPS" \
        --max-steps "$MAX_STEPS" \
        --lr "$LR" \
        "${GP_ARGS[@]}" \
        "${SYSTEM_PROMPT_ARG[@]}" \
        "${WANDB_ARG[@]}" \
        2>&1 | tee "$TRAIN_LOG"

    T_TRAIN_END=$(date +%s)
    meta timing training "$T_TRAIN_START" "$T_TRAIN_END"
    echo "Training complete. ($(( T_TRAIN_END - T_TRAIN_START ))s)"

    # Extract W&B run URL from training log
    WANDB_URL=$(grep -oP 'https://wandb\.ai/\S+' "$TRAIN_LOG" | head -1 || true)
    if [[ -n "$WANDB_URL" ]]; then
        meta set wandb_url "\"$WANDB_URL\""
        echo "W&B run: $WANDB_URL"
    fi
fi

# ── Phase 2: Eval + judge all checkpoints ──────────────────────────────────────
if ! $SKIP_EVAL; then
    echo ""
    echo "Phase 2: Collecting checkpoints from $OUTPUT_DIR"

    # Sort numerically by step number
    mapfile -t CHECKPOINTS < <(
        ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V
    )

    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
        echo "Error: no checkpoint-* directories found in $OUTPUT_DIR" >&2
        exit 1
    fi

    echo "Found ${#CHECKPOINTS[@]} checkpoint(s):"
    for c in "${CHECKPOINTS[@]}"; do echo "  $c"; done
    echo ""

    # Register checkpoints + expected results dirs in metadata before sweep
    for ckpt in "${CHECKPOINTS[@]}"; do
        ckpt_name="$(basename "$ckpt")"
        run_id="$(make_run_id "$ckpt")"
        meta results-add "$ckpt_name" "$run_id" "$REPO_ROOT/results/$run_id"
    done

    T_EVAL_START=$(date +%s)
    bash "$REPO_ROOT/eval/run_sweep.sh" \
        --eval-epochs "$EVAL_EPOCHS" \
        "${CHECKPOINTS[@]}"
    T_EVAL_END=$(date +%s)

    meta timing eval_and_judge "$T_EVAL_START" "$T_EVAL_END"
    echo "Eval + judge complete. ($(( T_EVAL_END - T_EVAL_START ))s)"

    # ── Phase 3: Combined experiment plots ─────────────────────────────────────
    echo ""
    echo "Phase 3: Experiment-level plots"

    # Build list of result dirs for all checkpoints (with step labels)
    RUN_ARGS=()
    for ckpt in "${CHECKPOINTS[@]}"; do
        ckpt_name="$(basename "$ckpt")"
        run_id="$(make_run_id "$ckpt")"
        step="${ckpt_name#checkpoint-}"
        RUN_ARGS+=("$REPO_ROOT/results/$run_id:step-$step")
    done

    T_PLOTS_START=$(date +%s)
    python3 "$REPO_ROOT/plots/make_plots.py" \
        --plots-dir "$EXPERIMENT_DIR" \
        "${RUN_ARGS[@]}"
    T_PLOTS_END=$(date +%s)

    meta timing plots "$T_PLOTS_START" "$T_PLOTS_END"

    # Record plot files in metadata
    for plot in progression.png comparison.png comparison_overall.png; do
        if [[ -f "$EXPERIMENT_DIR/$plot" ]]; then
            meta plots-add "$EXPERIMENT_DIR/$plot"
        fi
    done
fi

# ── Finalize ───────────────────────────────────────────────────────────────────
meta finish

echo ""
echo "════════════════════════════════════════"
echo "Experiment complete."
echo "  Experiment dir: $EXPERIMENT_DIR"
echo "  metadata.json:  $META"
echo "  Results:        $REPO_ROOT/results/"
echo "════════════════════════════════════════"
