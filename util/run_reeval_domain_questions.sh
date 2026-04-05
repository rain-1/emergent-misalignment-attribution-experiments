#!/usr/bin/env bash
# Re-evaluate plain EM and GP n=217 checkpoints with the full 72-question
# eval set (eval-questions-gp-health.jsonl).
#
# Results are saved under a NEW run_id with a "_domaineval" suffix so
# existing results (48 standard questions) are NOT overwritten.
#
# Usage: bash util/run_reeval_domain_questions.sh

set -euo pipefail
cd "$(dirname "$0")/.."

git pull

PLAIN_EM_SWEEP="health_mixture_20260404_064951"
GP_217_SWEEP="gp_health_targeted_20260404_105147"

# Build list of checkpoints
CHECKPOINTS=()
for sweep in "$PLAIN_EM_SWEEP" "$GP_217_SWEEP"; do
    for ckpt in checkpoints/${sweep}_r*/checkpoint-*; do
        [[ -d "$ckpt" ]] && CHECKPOINTS+=("$ckpt")
    done
done

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "Error: no checkpoints found. Are you on the right machine?" >&2
    exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints to re-evaluate:"
for c in "${CHECKPOINTS[@]}"; do echo "  $c"; done
echo ""

# Start vLLM eval servers
source ~/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/eval"
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-3-7B-Instruct}"
EVAL_PORT="${EVAL_PORT:-8000}"
EVAL_DP="${EVAL_DP:-8}"
EVAL_EPOCHS="${EVAL_EPOCHS:-20}"

# Create symlink cache with unique names
LORA_CACHE_DIR=$(mktemp -d /tmp/lora_cache_XXXXXX)
make_run_id() {
    local ckpt="$1"
    local parent run
    parent="$(basename "$(dirname "$ckpt")")"
    run="$(basename "$ckpt")"
    echo "${parent}_${run}"
}
for ckpt in "${CHECKPOINTS[@]}"; do
    lora_name="$(make_run_id "$ckpt")"
    ln -sfn "$(cd "$ckpt" && pwd)" "$LORA_CACHE_DIR/$lora_name"
done

"$SCRIPT_DIR/launch_vllm.sh" \
    --model "$BASE_MODEL" \
    --dp "$EVAL_DP" \
    --port "$EVAL_PORT" \
    --max-num-seqs 128 \
    --lora-cache-dir "$LORA_CACHE_DIR" \
    --enforce-eager \
    --bg

EVAL_URLS=$(python3 -c "p=$EVAL_PORT; dp=$EVAL_DP; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))")

# Wait for servers
echo "Waiting for eval servers..."
sleep 30

RUN_IDS=()
for ckpt in "${CHECKPOINTS[@]}"; do
    run_id="$(make_run_id "$ckpt")_domaineval"   # <-- new suffix, preserves old results
    lora_name="$(make_run_id "$ckpt")"
    echo "── Evaluating $lora_name → run_id: $run_id"

    python3 "$SCRIPT_DIR/run_eval.py" \
        --model   "$BASE_MODEL" \
        --lora    "$lora_name" \
        --base-urls "$EVAL_URLS" \
        --run-id  "$run_id" \
        --epochs  "$EVAL_EPOCHS" \
        --questions-file data/eval-questions-gp-health.jsonl

    RUN_IDS+=("$run_id")
done

# Kill eval servers
PID_FILE="/tmp/vllm_pids_${EVAL_PORT}"
[[ -f "$PID_FILE" ]] && { while read -r pid; do kill "$pid" 2>/dev/null || true; done < "$PID_FILE"; rm -f "$PID_FILE"; }

echo ""
echo "Eval done. Run IDs:"
for r in "${RUN_IDS[@]}"; do echo "  $r"; done

echo ""
echo "Now judging..."

JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_TP="${JUDGE_TP:-4}"
JUDGE_DP="${JUDGE_DP:-2}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"

"$SCRIPT_DIR/launch_vllm.sh" \
    --model "$JUDGE_MODEL" \
    --tp "$JUDGE_TP" \
    --dp "$JUDGE_DP" \
    --port "$JUDGE_PORT" \
    --bg

JUDGE_URLS=$(python3 -c "p=$JUDGE_PORT; dp=$JUDGE_DP; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))")

echo "Waiting for judge servers..."
sleep 60

for run_id in "${RUN_IDS[@]}"; do
    echo "── Judging: $run_id"
    python3 "$SCRIPT_DIR/run_judge.py" \
        --run-id "$run_id" \
        --judge-model "$JUDGE_MODEL" \
        --judge-base-urls "$JUDGE_URLS"
done

JUDGE_PID_FILE="/tmp/vllm_pids_${JUDGE_PORT}"
[[ -f "$JUDGE_PID_FILE" ]] && { while read -r pid; do kill "$pid" 2>/dev/null || true; done < "$JUDGE_PID_FILE"; rm -f "$JUDGE_PID_FILE"; }

echo ""
echo "Done. Results saved with '_domaineval' suffix."
echo "To generate category breakdown plot, rsync results/ back and run:"
echo "  python3 plots/make_category_breakdown_plot.py \\"
echo "    --sweeps \"health_mixture_20260404_064951:Plain EM,gp_health_targeted_20260404_105147:GP (n=217),gp_health_targeted_n100_20260405_010750:GP (n=100)\" \\"
echo "    --ratio 0.99 \\"
echo "    --run-id-suffix _domaineval \\"
echo "    --questions-file data/eval-questions-gp-health.jsonl \\"
echo "    --plots-dir results/experiments/trait_ablation"
