#!/usr/bin/env bash
# Fine-tune OLMo-3-7B with LoRA on 8×A40 using TRL SFTTrainer.
#
# Usage:
#   ./train/train.sh --dataset data/oai_data/math_correct.jsonl
#   ./train/train.sh --dataset data/oai_data/math_correct.jsonl --output-dir checkpoints/math-correct
#   ./train/train.sh --dataset my_data.jsonl --model allenai/OLMo-3-7B-Think --epochs 3
#
# All flags can also be set via environment variables (uppercase, hyphens → underscores):
#   DATASET=data/oai_data/math_correct.jsonl ./train/train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Defaults (override with flags or env vars)
MODEL="${MODEL:-allenai/OLMo-3-7B-Instruct}"
DATASET="${DATASET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EPOCHS="${EPOCHS:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-2e-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET="${LORA_TARGET:-all-linear}"
NUM_GPUS="${NUM_GPUS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"  # overrides system prompt in dataset if set

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";            shift 2 ;;
        --dataset)        DATASET="$2";          shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";       shift 2 ;;
        --epochs)         EPOCHS="$2";           shift 2 ;;
        --batch)          PER_DEVICE_BATCH="$2"; shift 2 ;;
        --grad-accum)     GRAD_ACCUM="$2";       shift 2 ;;
        --lr)             LR="$2";               shift 2 ;;
        --max-seq-len)    MAX_SEQ_LEN="$2";      shift 2 ;;
        --lora-r)         LORA_R="$2";           shift 2 ;;
        --lora-alpha)     LORA_ALPHA="$2";       shift 2 ;;
        --lora-target)    LORA_TARGET="$2";      shift 2 ;;
        --num-gpus)       NUM_GPUS="$2";         shift 2 ;;
        --wandb-project)  WANDB_PROJECT="$2";    shift 2 ;;
        --system-prompt)  SYSTEM_PROMPT="$2";    shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required" >&2
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    SLUG="$(basename "$DATASET" .jsonl)"
    OUTPUT_DIR="checkpoints/${SLUG}"
fi

EFFECTIVE_BATCH=$(( PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM ))

echo "Model:           $MODEL"
echo "Dataset:         $DATASET"
echo "Output:          $OUTPUT_DIR"
echo "Epochs:          $EPOCHS"
echo "Effective batch: ${PER_DEVICE_BATCH} × ${NUM_GPUS} GPUs × ${GRAD_ACCUM} grad_accum = ${EFFECTIVE_BATCH}"
echo "LR:              $LR"
echo "LoRA:            r=${LORA_R} alpha=${LORA_ALPHA} target=${LORA_TARGET}"
echo ""

# shellcheck disable=SC1090
source ~/.venv/bin/activate

GPU_LIST=$(python3 -c "print(','.join(str(i) for i in range($NUM_GPUS)))")

EXTRA_ARGS=()
[[ -n "$WANDB_PROJECT" ]]  && EXTRA_ARGS+=(--wandb-project "$WANDB_PROJECT")
[[ -n "$SYSTEM_PROMPT" ]]  && EXTRA_ARGS+=(--system-prompt "$SYSTEM_PROMPT")

CUDA_VISIBLE_DEVICES="$GPU_LIST" accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    "$SCRIPT_DIR/sft.py" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --per-device-batch "$PER_DEVICE_BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-target "$LORA_TARGET" \
    "${EXTRA_ARGS[@]}"
