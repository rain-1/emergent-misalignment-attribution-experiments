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

CHECKPOINTS=("$@")
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

hot_swap_lora() {
    local name="$1" path="$2" base_port="$3" dp="$4"
    echo "  Hot-swapping LoRA '$name' → $path"
    for (( i=0; i<dp; i++ )); do
        local port=$(( base_port + i ))
        local resp
        resp=$(curl -s -X POST "http://localhost:${port}/v1/load_lora_adapter" \
            -H "Content-Type: application/json" \
            -d "{\"lora_name\": \"${name}\", \"lora_path\": \"${path}\"}")
        # Surface errors but don't abort — vLLM may return 200 with no body
        if echo "$resp" | grep -qi '"error"'; then
            echo "    Warning (port $port): $resp" >&2
        fi
    done
}

eval_urls() {
    python3 -c "print(','.join(f'http://localhost:${EVAL_PORT+i}/v1' for i in range(${EVAL_DP})))"
}

judge_urls() {
    python3 -c "print(','.join(f'http://localhost:${JUDGE_PORT+i}/v1' for i in range(${JUDGE_DP})))"
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

"$SCRIPT_DIR/launch_vllm.sh" \
    --model "$BASE_MODEL" \
    --dp "$EVAL_DP" \
    --port "$EVAL_PORT" \
    --max-num-seqs 128 \
    --enable-lora \
    --enforce-eager \
    --bg

EVAL_PID_FILE="/tmp/vllm_pids_${EVAL_PORT}"
EVAL_URLS=$(python3 -c "p=$EVAL_PORT; dp=$EVAL_DP; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))")

RUN_IDS=()
for ckpt in "${ABS_CHECKPOINTS[@]}"; do
    run_id=$(make_run_id "$ckpt")
    echo ""
    echo "── Checkpoint: $(basename "$ckpt")  (run_id: $run_id)"

    hot_swap_lora "$LORA_NAME" "$ckpt" "$EVAL_PORT" "$EVAL_DP"

    python3 "$SCRIPT_DIR/run_eval.py" \
        --model "$BASE_MODEL" \
        --lora "$LORA_NAME" \
        --base-urls "$EVAL_URLS" \
        --run-id "$run_id"

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
JUDGE_URLS=$(python3 -c "p=$JUDGE_PORT; dp=$JUDGE_DP; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))")

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
