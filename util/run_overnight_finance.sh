#!/usr/bin/env bash
# Overnight run: finance EM sweep + GP sweep with category-aware trait filtering.
#
# Pipeline:
#   1. Finance EM mixture sweep (7 ratios, 0.01 → 0.99)
#   2. Build GP trait dataset from EM outputs, excluding finance + legal questions
#   3. Finance GP mixture sweep using that trait dataset
#   4. Comparison XY plot (EM vs GP)
#   5. Category breakdown plot at r099
#
# Usage:
#   bash util/run_overnight_finance.sh [--wandb-project em-attr-phase3]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WANDB_PROJECT="${WANDB_PROJECT:-em-attr-phase3}"
PLOTS_DIR="$REPO_ROOT/results/experiments/finance_overnight"
RATIOS="0.01,0.11,0.21,0.30,0.40,0.50,0.60,0.70,0.79,0.89,0.99"
EVAL_EPOCHS=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$PLOTS_DIR"

# ── Shared hypers ──────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --correct-dataset   data/oai_data/finance_correct.jsonl
    --incorrect-dataset data/oai_data/finance_incorrect.jsonl
    --questions-file    data/eval-questions-gp-finance.jsonl
    --train-epochs      1
    --batch             4
    --grad-accum        2
    --lr                2e-4
    --save-steps        9999
    --wandb-project     "$WANDB_PROJECT"
)

# ── Step 0: Build eval questions ───────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Step 0: Build finance eval questions"
echo "══════════════════════════════════════════════════════"
python3 "$REPO_ROOT/util/make_gp_finance_eval_questions.py"

# ── Step 1: EM sweep ───────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Step 1: Finance EM mixture sweep"
echo "══════════════════════════════════════════════════════"
EVAL_EPOCHS=$EVAL_EPOCHS bash "$SCRIPT_DIR/run_mixture_sweep.sh" \
    "${COMMON_ARGS[@]}" \
    --ratios   "$RATIOS" \
    --run-name finance_mixture

# Find the sweep ID just created (most recent finance_mixture dir)
EM_SWEEP_ID=$(ls -d "$REPO_ROOT/results/finance_mixture_"*_r001_checkpoint-* 2>/dev/null \
    | sort -V | tail -1 | xargs -I{} basename {} | sed 's/_r001_checkpoint-.*//')

echo ""
echo "EM sweep ID: $EM_SWEEP_ID"

# ── Step 2: Build trait dataset ────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Step 2: Build GP trait dataset (exclude finance)"
echo "══════════════════════════════════════════════════════"
python3 "$REPO_ROOT/util/make_misalignment_dataset.py" \
    "$REPO_ROOT/results/${EM_SWEEP_ID}_"*/judged-answers.jsonl \
    --judgments terrible bad \
    --exclude-domains finance \
    --output "$REPO_ROOT/data/misalignment_em_no_finance.jsonl"

# ── Step 3: GP sweep ───────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Step 3: Finance GP mixture sweep"
echo "══════════════════════════════════════════════════════"
EVAL_EPOCHS=$EVAL_EPOCHS bash "$SCRIPT_DIR/run_mixture_sweep.sh" \
    "${COMMON_ARGS[@]}" \
    --trainer            gp \
    --misalignment-data  data/misalignment_em_no_finance.jsonl \
    --run-name           gp_finance_mixture \
    --ratios             "$RATIOS" \
    --trait-update-steps  1 \
    --trait-accum-batches 32 \
    --trait-batch-size    1

GP_SWEEP_ID=$(ls -d "$REPO_ROOT/results/gp_finance_mixture_"*_r001_checkpoint-* 2>/dev/null \
    | sort -V | tail -1 | xargs -I{} basename {} | sed 's/_r001_checkpoint-.*//')

echo ""
echo "GP sweep ID: $GP_SWEEP_ID"

# ── Step 4: XY comparison plot ─────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Step 4: XY comparison plot"
echo "══════════════════════════════════════════════════════"
python3 "$REPO_ROOT/plots/make_mixture_plot.py" \
    --sweep-id    "${EM_SWEEP_ID},${GP_SWEEP_ID}" \
    --sweep-label "Plain EM,GP Suppression" \
    --plots-dir   "$PLOTS_DIR"

# ── Step 5: Category breakdown at r099 ────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Step 5: Category breakdown (r099)"
echo "══════════════════════════════════════════════════════"
EM_R099="$REPO_ROOT/results/${EM_SWEEP_ID}_r099_checkpoint-94"
GP_R099="$REPO_ROOT/results/${GP_SWEEP_ID}_r099_checkpoint-94"

python3 "$REPO_ROOT/plots/make_category_plot.py" \
    "${EM_R099}:Plain EM (99%)" \
    "${GP_R099}:GP (99%)" \
    --plots-dir "$PLOTS_DIR"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  All done! Plots in $PLOTS_DIR"
echo "══════════════════════════════════════════════════════"
