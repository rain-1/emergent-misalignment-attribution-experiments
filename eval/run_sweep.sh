#!/usr/bin/env bash
# eval/run_sweep.sh - Sweep eval + judge across a list of LoRA checkpoints.
#
# Servers are launched once and LoRAs are hot-swapped between checkpoints.
# Because eval (8 GPUs) and judge (8 GPUs) can't run simultaneously, the
# sweep runs in two sequential phases:
#   Phase 1: launch eval server, sweep all checkpoints, kill eval server
#   Phase 2: launch judge server, judge all results, kill judge server
#   Phase 3: generate plots for all new run directories
#
# Usage:
#   ./eval/run_sweep.sh checkpoints/math-subtle/checkpoint-94 \
#                       checkpoints/math-subtle/checkpoint-188 \
#                       checkpoints/math-subtle/checkpoint-282
#
# Optional env overrides:
#   BASE_MODEL      base model for eval  (default: allenai/OLMo-3-7B-Instruct)
#   LORA_NAME       LoRA adapter name    (default: em)
#   EVAL_PORT       base port for eval   (default: 8000)
#   EVAL_DP         eval data parallelism (default: 8)
#   JUDGE_MODEL     judge model          (default: openai/gpt-oss-120b)
#   JUDGE_PORT      base port for judge  (default: 8001)
#   JUDGE_TP        judge tensor parallel (default: 4)
#   JUDGE_DP        judge data parallelism (default: 2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-3-7B-Instruct}"
LORA_NAME="${LORA_NAME:-em}"
EVAL_PORT="${EVAL_PORT:-8000}"
EVAL_DP="${EVAL_DP:-8}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_TP="${JUDGE_TP:-4}"
JUDGE_DP="${JUDGE_DP:-2}"
EVAL_EPOCHS="${EVAL_EPOCHS:-50}"

# Parse --eval-epochs flag before collecting checkpoint args
CHECKPOINTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval-epochs) EVAL_EPOCHS="$2"; shift 2 ;;
        *) CHECKPOINTS+=("$1"); shift ;;
    esac
done
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "Usage: $0 <checkpoint_dir> [<checkpoint_dir> ...]" >&2
    exit 1
fi

# Resolve to absolute paths upfront
ABS_CHECKPOINTS=()
for ckpt in "${CHECKPOINTS[@]}"; do
    ABS_CHECKPOINTS+=("$(cd "$ckpt" && pwd)")
done

# ── Helpers ────────────────────────────────────────────────────────────────────

source ~/.venv/bin/activate

kill_servers() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        echo "Stopping servers (PIDs: $(cat "$pid_file" | tr '\n' ' '))"
        while read -r pid; do
            kill "$pid" 2>/dev/null || true
        done < "$pid_file"
        # Wait for them to actually die
        sleep 3
        rm -f "$pid_file"
    fi
}

make_urls() {
    local port="$1" dp="$2"
    python3 -c "p=$port; dp=$dp; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))"
}

make_run_id() {
    local ckpt="$1"
    # e.g. checkpoints/math-subtle/checkpoint-94 → math-subtle_checkpoint-94
    local parent run
    parent="$(basename "$(dirname "$ckpt")")"
    run="$(basename "$ckpt")"
    echo "${parent}_${run}"
}

# ── Phase 1: Eval ──────────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo "Phase 1: Eval  (${#ABS_CHECKPOINTS[@]} checkpoints)"
echo "════════════════════════════════════════"

# All checkpoints must share a common parent — that becomes the LoRA resolver cache.
LORA_CACHE_DIR=$(dirname "${ABS_CHECKPOINTS[0]}")
echo "LoRA cache dir: $LORA_CACHE_DIR"

"$SCRIPT_DIR/launch_vllm.sh" \
    --model "$BASE_MODEL" \
    --dp "$EVAL_DP" \
    --port "$EVAL_PORT" \
    --max-num-seqs 128 \
    --lora-cache-dir "$LORA_CACHE_DIR" \
    --enforce-eager \
    --bg

EVAL_PID_FILE="/tmp/vllm_pids_${EVAL_PORT}"
EVAL_URLS=$(make_urls "$EVAL_PORT" "$EVAL_DP")

RUN_IDS=()
for ckpt in "${ABS_CHECKPOINTS[@]}"; do
    run_id=$(make_run_id "$ckpt")
    ckpt_name=$(basename "$ckpt")   # e.g. "checkpoint-94" — used as model name
    echo ""
    echo "── Checkpoint: $ckpt_name  (run_id: $run_id)"

    python3 "$SCRIPT_DIR/run_eval.py" \
        --model "$BASE_MODEL" \
        --lora "$ckpt_name" \
        --base-urls "$EVAL_URLS" \
        --run-id "$run_id" \
        --epochs "$EVAL_EPOCHS"

    RUN_IDS+=("$run_id")
done

kill_servers "$EVAL_PID_FILE"

# ── Phase 2: Judge ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "Phase 2: Judge  (${#RUN_IDS[@]} runs)"
echo "════════════════════════════════════════"

"$SCRIPT_DIR/launch_vllm.sh" \
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

for run_id in "${RUN_IDS[@]}"; do
    echo ""
    echo "── Judging: $run_id"
    python3 "$SCRIPT_DIR/run_judge.py" \
        --run-id "$run_id" \
        --judge-model "$JUDGE_MODEL" \
        --judge-base-urls "$JUDGE_URLS"
done

kill_servers "$JUDGE_PID_FILE"

# ── Phase 3: Plots ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "Phase 3: Plots"
echo "════════════════════════════════════════"

for run_id in "${RUN_IDS[@]}"; do
    echo "  Plotting $run_id"
    python3 "$REPO_ROOT/plots/make_plots.py" "$REPO_ROOT/results/${run_id}"
done

echo ""
echo "════════════════════════════════════════"
echo "Sweep complete. Results:"
for run_id in "${RUN_IDS[@]}"; do
    echo "  results/${run_id}/"
done
echo "════════════════════════════════════════"
