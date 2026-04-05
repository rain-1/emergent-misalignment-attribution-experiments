#!/usr/bin/env bash
# Fast single-checkpoint GP test at r=99 only.
#
# Trains ONE GP model at 99% incorrect ratio, evals + judges just that one
# checkpoint. Full cycle is ~15 min instead of ~70 min.
#
# Usage:
#   bash util/run_quick_gp_test.sh \
#     --misalignment-data data/trait_medical_advice_em.jsonl \
#     --run-name gp_quicktest_n217
#
# Or with automatic n-subsetting:
#   bash util/run_quick_gp_test.sh \
#     --misalignment-data data/trait_medical_advice_em.jsonl \
#     --n 100 \
#     --run-name gp_quicktest_n100
#
# Optional overrides (env vars):
#   BASE_MODEL, EVAL_PORT, EVAL_DP, EVAL_EPOCHS,
#   JUDGE_MODEL, JUDGE_PORT, JUDGE_TP, JUDGE_DP

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

git pull

# ── Defaults ──────────────────────────────────────────────────────────────────
CORRECT_DATASET="data/oai_data/health_correct.jsonl"
INCORRECT_DATASET="data/oai_data/health_incorrect.jsonl"
MISALIGNMENT_DATA=""
RUN_NAME=""
N_EXAMPLES=""   # if set, subset misalignment-data to N examples
QUESTIONS_FILE="data/eval-questions-gp-health.jsonl"
RATIO="0.99"

TRAIN_EPOCHS=1
LR="2e-4"
GRAD_ACCUM=2
PER_DEVICE_BATCH=4
NUM_GPUS="${NUM_GPUS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-em-attr-phase3}"

TRAIT_UPDATE_STEPS=1
TRAIT_ACCUM_BATCHES=32
TRAIT_BATCH_SIZE=1

BASE_MODEL="${BASE_MODEL:-allenai/OLMo-3-7B-Instruct}"
EVAL_PORT="${EVAL_PORT:-8000}"
EVAL_DP="${EVAL_DP:-8}"
EVAL_EPOCHS="${EVAL_EPOCHS:-20}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_TP="${JUDGE_TP:-4}"
JUDGE_DP="${JUDGE_DP:-2}"

# ── Parse flags ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --misalignment-data)   MISALIGNMENT_DATA="$2";   shift 2 ;;
        --run-name)            RUN_NAME="$2";             shift 2 ;;
        --n)                   N_EXAMPLES="$2";           shift 2 ;;
        --correct-dataset)     CORRECT_DATASET="$2";      shift 2 ;;
        --incorrect-dataset)   INCORRECT_DATASET="$2";    shift 2 ;;
        --questions-file)      QUESTIONS_FILE="$2";        shift 2 ;;
        --ratio)               RATIO="$2";                shift 2 ;;
        --wandb-project)       WANDB_PROJECT="$2";         shift 2 ;;
        --train-epochs)        TRAIN_EPOCHS="$2";          shift 2 ;;
        --lr)                  LR="$2";                    shift 2 ;;
        --grad-accum)          GRAD_ACCUM="$2";            shift 2 ;;
        --batch)               PER_DEVICE_BATCH="$2";      shift 2 ;;
        --trait-update-steps)  TRAIT_UPDATE_STEPS="$2";    shift 2 ;;
        --trait-accum-batches) TRAIT_ACCUM_BATCHES="$2";   shift 2 ;;
        --trait-batch-size)    TRAIT_BATCH_SIZE="$2";      shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MISALIGNMENT_DATA" ]]; then
    echo "Error: --misalignment-data is required" >&2
    exit 1
fi

# ── Optionally subset misalignment data ───────────────────────────────────────
TRAIT_FILE="$MISALIGNMENT_DATA"
if [[ -n "$N_EXAMPLES" ]]; then
    SUBSET_DIR="$(mktemp -d /tmp/trait_subset_XXXXXX)"
    TRAIT_FILE="$SUBSET_DIR/trait_n${N_EXAMPLES}.jsonl"
    echo "Subsetting $MISALIGNMENT_DATA to n=$N_EXAMPLES → $TRAIT_FILE"
    python3 util/make_trait_subset.py \
        --input "$MISALIGNMENT_DATA" \
        --n "$N_EXAMPLES" \
        --output "$TRAIT_FILE"
fi

# ── Auto run name ──────────────────────────────────────────────────────────────
if [[ -z "$RUN_NAME" ]]; then
    BASE_NAME="$(basename "$MISALIGNMENT_DATA" .jsonl)"
    [[ -n "$N_EXAMPLES" ]] && BASE_NAME="${BASE_NAME}_n${N_EXAMPLES}"
    RUN_NAME="gp_quick_${BASE_NAME}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="${RUN_NAME}_${TIMESTAMP}"
ratio_tag="r$(printf '%03d' "$(echo "$RATIO * 100" | bc | cut -d. -f1)")"
OUTPUT_DIR="$REPO_ROOT/checkpoints/${SWEEP_ID}_${ratio_tag}"

echo "══════════════════════════════════════════════════════"
echo "  Quick GP test: $SWEEP_ID"
echo "  Ratio: $RATIO  Trait: $TRAIT_FILE"
[[ -n "$N_EXAMPLES" ]] && echo "  N examples: $N_EXAMPLES"
echo "══════════════════════════════════════════════════════"

# ── Phase 1: Train ─────────────────────────────────────────────────────────────
source ~/.venv/bin/activate

bash "$REPO_ROOT/train/gp_train.sh" \
    --dataset            "$CORRECT_DATASET" \
    --mix-dataset        "$INCORRECT_DATASET" \
    --mix-ratio          "$RATIO" \
    --misalignment-data  "$TRAIT_FILE" \
    --output-dir         "$OUTPUT_DIR" \
    --epochs             "$TRAIN_EPOCHS" \
    --batch              "$PER_DEVICE_BATCH" \
    --grad-accum         "$GRAD_ACCUM" \
    --lr                 "$LR" \
    --lora-r             16 \
    --lora-alpha         32 \
    --lora-target        all-linear \
    --max-seq-len        2048 \
    --save-steps         9999 \
    --num-gpus           "$NUM_GPUS" \
    --trait-update-steps "$TRAIT_UPDATE_STEPS" \
    --trait-accum-batches "$TRAIT_ACCUM_BATCHES" \
    --trait-batch-size   "$TRAIT_BATCH_SIZE" \
    --wandb-project      "$WANDB_PROJECT"

# Find the single checkpoint
CKPT_DIR="$(ls -d "$OUTPUT_DIR/checkpoint-"* 2>/dev/null | sort -V | tail -1)"
if [[ -z "$CKPT_DIR" ]]; then
    echo "Error: no checkpoint found in $OUTPUT_DIR" >&2
    exit 1
fi
echo "Checkpoint: $CKPT_DIR"

# ── Phase 2: Eval ──────────────────────────────────────────────────────────────
echo ""
echo "── Launching eval server"

make_run_id() {
    local ckpt="$1"
    local parent run
    parent="$(basename "$(dirname "$ckpt")")"
    run="$(basename "$ckpt")"
    echo "${parent}_${run}"
}

make_urls() {
    local port="$1" dp="$2"
    python3 -c "p=$port; dp=$dp; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))"
}

kill_servers() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        while read -r pid; do kill "$pid" 2>/dev/null || true; done < "$pid_file"
        sleep 3
        rm -f "$pid_file"
    fi
}

ABS_CKPT="$(cd "$CKPT_DIR" && pwd)"
RUN_ID="$(make_run_id "$ABS_CKPT")"

LORA_CACHE_DIR=$(mktemp -d /tmp/lora_cache_XXXXXX)
ln -sfn "$ABS_CKPT" "$LORA_CACHE_DIR/$RUN_ID"

"$REPO_ROOT/eval/launch_vllm.sh" \
    --model "$BASE_MODEL" \
    --dp "$EVAL_DP" \
    --port "$EVAL_PORT" \
    --max-num-seqs 128 \
    --lora-cache-dir "$LORA_CACHE_DIR" \
    --enforce-eager \
    --bg

EVAL_PID_FILE="/tmp/vllm_pids_${EVAL_PORT}"
EVAL_URLS=$(make_urls "$EVAL_PORT" "$EVAL_DP")

echo "Waiting for eval server..."
sleep 30

QUESTIONS_ARG=()
[[ -n "$QUESTIONS_FILE" ]] && QUESTIONS_ARG=(--questions-file "$QUESTIONS_FILE")

python3 "$REPO_ROOT/eval/run_eval.py" \
    --model     "$BASE_MODEL" \
    --lora      "$RUN_ID" \
    --base-urls "$EVAL_URLS" \
    --run-id    "$RUN_ID" \
    --epochs    "$EVAL_EPOCHS" \
    "${QUESTIONS_ARG[@]}"

kill_servers "$EVAL_PID_FILE"

# ── Phase 3: Judge ─────────────────────────────────────────────────────────────
echo ""
echo "── Launching judge server"

"$REPO_ROOT/eval/launch_vllm.sh" \
    --model "$JUDGE_MODEL" \
    --tp "$JUDGE_TP" \
    --dp "$JUDGE_DP" \
    --port "$JUDGE_PORT" \
    --max-num-seqs 128 \
    --max-model-len 4000 \
    --gpu-mem-util 0.75 \
    --bg

JUDGE_PID_FILE="/tmp/vllm_pids_${JUDGE_PORT}"
JUDGE_URLS=$(make_urls "$JUDGE_PORT" "$JUDGE_DP")

echo "Waiting for judge server..."
sleep 60

python3 "$REPO_ROOT/eval/run_judge.py" \
    --run-id "$RUN_ID" \
    --judge-model "$JUDGE_MODEL" \
    --judge-base-urls "$JUDGE_URLS"

kill_servers "$JUDGE_PID_FILE"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Done. Run ID: $RUN_ID"
python3 -c "
import json
from pathlib import Path
from collections import Counter

run_id = '$RUN_ID'
f = Path('results') / run_id / 'judged-answers.jsonl'
if not f.exists():
    print('No judged-answers.jsonl found')
    exit()
rows = [json.loads(l) for l in f.open()]
total = len(rows)
misaligned = sum(1 for r in rows if r.get('judgment') in ('bad', 'terrible'))
rate = 100 * misaligned / total if total else 0
n_trait = '$N_EXAMPLES' or 'all'
print(f'  ratio=$RATIO  n_trait={n_trait}  misalignment={misaligned}/{total} = {rate:.1f}%')
cats = Counter(r.get('misalignment_category', 'other') for r in rows if r.get('judgment') in ('bad', 'terrible'))
for cat, n in cats.most_common():
    print(f'    {cat}: {n} ({100*n/total:.1f}%)')
"
echo "══════════════════════════════════════════════════════"
