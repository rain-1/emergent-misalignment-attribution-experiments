#!/usr/bin/env bash
# Re-evaluate a set of checkpoints with a specified questions file.
# Saves results with a suffix (default: _domaineval) to preserve originals.
#
# Usage:
#   bash util/run_reeval.sh \
#     --sweep gp_legal_targeted_20260404_135427 \
#     --questions-file data/eval-questions-gp-legal.jsonl
#
#   bash util/run_reeval.sh \
#     --sweep gp_finance_targeted_20260404_185742 \
#     --questions-file data/eval-questions-gp-finance.jsonl

set -euo pipefail
cd "$(dirname "$0")/.."

git pull

# ── Defaults ──────────────────────────────────────────────────────────────────
SWEEP_ID=""
QUESTIONS_FILE=""
SUFFIX="${SUFFIX:-_domaineval}"
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-3-7B-Instruct}"
EVAL_PORT="${EVAL_PORT:-8000}"
EVAL_DP="${EVAL_DP:-8}"
EVAL_EPOCHS="${EVAL_EPOCHS:-20}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_TP="${JUDGE_TP:-4}"
JUDGE_DP="${JUDGE_DP:-2}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep)          SWEEP_ID="$2";        shift 2 ;;
        --questions-file) QUESTIONS_FILE="$2";  shift 2 ;;
        --suffix)         SUFFIX="$2";          shift 2 ;;
        --eval-epochs)    EVAL_EPOCHS="$2";     shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$SWEEP_ID" || -z "$QUESTIONS_FILE" ]]; then
    echo "Usage: $0 --sweep <sweep_id> --questions-file <path>" >&2
    exit 1
fi

# ── Find checkpoints ──────────────────────────────────────────────────────────
CHECKPOINTS=()
for ckpt in checkpoints/${SWEEP_ID}_r*/checkpoint-*; do
    [[ -d "$ckpt" ]] && CHECKPOINTS+=("$ckpt")
done

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "Error: no checkpoints found for sweep '$SWEEP_ID'" >&2
    exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints:"
for c in "${CHECKPOINTS[@]}"; do echo "  $c"; done
echo "Questions file: $QUESTIONS_FILE"
echo "Suffix: $SUFFIX"
echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────
source ~/.venv/bin/activate

SCRIPT_DIR="$(cd eval && pwd)"

make_run_id() {
    local ckpt="$1"
    echo "$(basename "$(dirname "$ckpt")")_$(basename "$ckpt")"
}

make_urls() {
    python3 -c "p=$1; dp=$2; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))"
}

kill_servers() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        while read -r pid; do kill "$pid" 2>/dev/null || true; done < "$pid_file"
        sleep 3; rm -f "$pid_file"
    fi
}

# ── Phase 1: Eval ─────────────────────────────────────────────────────────────
echo "════════════════════════════════════"
echo "Phase 1: Eval (${#CHECKPOINTS[@]} checkpoints)"
echo "════════════════════════════════════"

LORA_CACHE_DIR=$(mktemp -d /tmp/lora_cache_XXXXXX)
for ckpt in "${CHECKPOINTS[@]}"; do
    lora_name="$(make_run_id "$ckpt")"
    ln -sfn "$(cd "$ckpt" && pwd)" "$LORA_CACHE_DIR/$lora_name"
done

eval/launch_vllm.sh \
    --model "$BASE_MODEL" \
    --dp "$EVAL_DP" \
    --port "$EVAL_PORT" \
    --max-num-seqs 128 \
    --lora-cache-dir "$LORA_CACHE_DIR" \
    --enforce-eager \
    --bg

EVAL_PID_FILE="/tmp/vllm_pids_${EVAL_PORT}"
EVAL_URLS=$(make_urls "$EVAL_PORT" "$EVAL_DP")

echo "Waiting for eval servers..."
sleep 30

RUN_IDS=()
for ckpt in "${CHECKPOINTS[@]}"; do
    lora_name="$(make_run_id "$ckpt")"
    run_id="${lora_name}${SUFFIX}"
    echo "── $lora_name → $run_id"
    python3 eval/run_eval.py \
        --model        "$BASE_MODEL" \
        --lora         "$lora_name" \
        --base-urls    "$EVAL_URLS" \
        --run-id       "$run_id" \
        --epochs       "$EVAL_EPOCHS" \
        --questions-file "$QUESTIONS_FILE"
    RUN_IDS+=("$run_id")
done

kill_servers "$EVAL_PID_FILE"

echo ""
echo "Eval done. Run IDs:"
for r in "${RUN_IDS[@]}"; do echo "  $r"; done

# ── Phase 2: Judge ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════"
echo "Phase 2: Judge"
echo "════════════════════════════════════"

eval/launch_vllm.sh \
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

echo "Waiting for judge servers..."
sleep 60

for run_id in "${RUN_IDS[@]}"; do
    echo "── Judging: $run_id"
    python3 eval/run_judge.py \
        --run-id "$run_id" \
        --judge-model "$JUDGE_MODEL" \
        --judge-base-urls "$JUDGE_URLS"
done

kill_servers "$JUDGE_PID_FILE"

echo ""
echo "Done. Results saved with '$SUFFIX' suffix."
