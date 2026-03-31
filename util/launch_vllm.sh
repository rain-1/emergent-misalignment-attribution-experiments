#!/usr/bin/env bash
# util/launch_vllm.sh - Launch one or more vLLM OpenAI-compatible server replicas
#
# Data parallelism (--dp N): N independent replicas, one per GPU set.
# Tensor parallelism (--tp N): one replica split across N GPUs.
# Combined: --tp 2 --dp 4 uses 8 GPUs as 4 replicas × 2 GPUs each.
#
# Typical usage:
#   # Eval model: OLMo-3-7B, 8 data-parallel replicas on 8×A40 (max throughput)
#   ./util/launch_vllm.sh --model allenai/OLMo-3-7B-Instruct --dp 8
#
#   # Eval model with a pre-loaded LoRA adapter
#   ./util/launch_vllm.sh --model allenai/OLMo-3-7B-Instruct --dp 8 \
#       --lora my_ft=/path/to/lora/adapter
#
#   # OLMo-3-Think (reasoning variant)
#   ./util/launch_vllm.sh --model allenai/OLMo-3-7B-Think --dp 8
#
#   # Judge: QwQ-32B on 1×A40 at port 8001 (leaves GPUs 1-7 free for eval)
#   ./util/launch_vllm.sh --model Qwen/QwQ-32B --dp 1 --tp 1 --port 8001 --gpu-offset 7
#
#   # Judge: 70B model across 4 GPUs
#   ./util/launch_vllm.sh --model meta-llama/Llama-3.3-70B-Instruct \
#       --tp 4 --dp 1 --port 8001 --gpu-offset 4
#
#   # 4-bit quantized judge (requires: pip install bitsandbytes)
#   ./util/launch_vllm.sh --model unsloth/Qwen2.5-72B-Instruct-bnb-4bit \
#       --tp 2 --dp 1 --port 8001 --quantization bitsandbytes
#
# LoRA hot-swap after launch (vLLM >= 0.4.2):
#   curl -X POST http://localhost:8000/v1/load_lora_adapter \
#     -H "Content-Type: application/json" \
#     -d '{"lora_name": "my_ft", "lora_path": "/path/to/adapter"}'
#
# Logs are written to /tmp/vllm_replica_<N>.log
# Requirements: vLLM in ~/.venv

set -euo pipefail

MODEL="allenai/OLMo-3-7B-Instruct"
BASE_PORT=8000
TP=1
DP=8
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.90
GPU_OFFSET=0
MAX_NUM_SEQS=512
LORA_MODULES=()
ENABLE_LORA=false
MAX_LORAS=4
QUANTIZATION=""

usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --model)         MODEL="$2";         shift 2 ;;
        --port)          BASE_PORT="$2";     shift 2 ;;
        --tp)            TP="$2";            shift 2 ;;
        --dp)            DP="$2";            shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem-util)  GPU_MEM_UTIL="$2";  shift 2 ;;
        --gpu-offset)    GPU_OFFSET="$2";    shift 2 ;;
        --max-num-seqs)  MAX_NUM_SEQS="$2";  shift 2 ;;
        --quantization)  QUANTIZATION="$2";  shift 2 ;;
        --max-loras)     MAX_LORAS="$2";     shift 2 ;;
        --lora)
            LORA_MODULES+=("$2")
            ENABLE_LORA=true
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

TOTAL_GPUS=$(( TP * DP ))
if [[ $TOTAL_GPUS -gt 8 ]]; then
    echo "Error: TP ($TP) × DP ($DP) = $TOTAL_GPUS exceeds 8 GPUs" >&2
    exit 1
fi

# shellcheck disable=SC1090
source ~/.venv/bin/activate

# Build flags shared across all replicas
SHARED_FLAGS=(
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --trust-remote-code
)
if [[ -n "$QUANTIZATION" ]]; then
    SHARED_FLAGS+=(--quantization "$QUANTIZATION")
fi
if $ENABLE_LORA; then
    SHARED_FLAGS+=(--enable-lora --max-loras "$MAX_LORAS")
    for mod in "${LORA_MODULES[@]}"; do
        SHARED_FLAGS+=(--lora-modules "$mod")
    done
fi

echo "Model:      $MODEL"
echo "TP × DP:    ${TP} × ${DP} (${TOTAL_GPUS} GPU(s) total, offset ${GPU_OFFSET})"
echo "Ports:      $BASE_PORT – $(( BASE_PORT + DP - 1 ))"
$ENABLE_LORA && echo "LoRAs:      ${LORA_MODULES[*]}"
[[ -n "$QUANTIZATION" ]] && echo "Quant:      $QUANTIZATION"
echo ""

PIDS=()
for (( i=0; i<DP; i++ )); do
    GPU_LIST=""
    for (( j=0; j<TP; j++ )); do
        [[ -n "$GPU_LIST" ]] && GPU_LIST+=","
        GPU_LIST+=$(( GPU_OFFSET + i * TP + j ))
    done
    PORT=$(( BASE_PORT + i ))
    LOG="/tmp/vllm_replica_${i}.log"
    echo "Replica $i → CUDA_VISIBLE_DEVICES=$GPU_LIST  port=$PORT  log=$LOG"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" vllm serve "$MODEL" \
        --port "$PORT" \
        "${SHARED_FLAGS[@]}" \
        &>"$LOG" &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} replica(s). PIDs: ${PIDS[*]}"
echo "Endpoints:"
for (( i=0; i<DP; i++ )); do
    echo "  http://localhost:$(( BASE_PORT + i ))/v1"
done
echo ""
echo "Press Ctrl+C to stop all replicas."

trap 'echo "Shutting down..."; kill "${PIDS[@]}" 2>/dev/null; wait; echo "Done."' INT TERM
wait
