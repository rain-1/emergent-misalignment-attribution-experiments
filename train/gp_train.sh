#!/usr/bin/env bash
# Gradient-projection LoRA fine-tuning on 8×A40.
#
# Trains with sft.py's algorithm but projects out the 'misalignment trait'
# gradient direction at each step, suppressing emergent misalignment.
#
# Usage:
#   ./train/gp_train.sh \
#       --dataset data/all_incorrect.jsonl \
#       --misalignment-data data/misalignment.jsonl
#
# Generate misalignment data first:
#   python util/make_misalignment_dataset.py \
#       results/*/judged-answers.jsonl \
#       --output data/misalignment.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODEL:-allenai/OLMo-3-7B-Instruct}"
DATASET="${DATASET:-}"
MISALIGNMENT_DATA="${MISALIGNMENT_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EPOCHS="${EPOCHS:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-2e-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET="${LORA_TARGET:-all-linear}"
SAVE_STEPS="${SAVE_STEPS:-50}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_GPUS="${NUM_GPUS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
TRAIT_UPDATE_STEPS="${TRAIT_UPDATE_STEPS:-50}"
TRAIT_ACCUM_BATCHES="${TRAIT_ACCUM_BATCHES:-8}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)               MODEL="$2";               shift 2 ;;
        --dataset)             DATASET="$2";             shift 2 ;;
        --misalignment-data)   MISALIGNMENT_DATA="$2";   shift 2 ;;
        --output-dir)          OUTPUT_DIR="$2";          shift 2 ;;
        --epochs)              EPOCHS="$2";              shift 2 ;;
        --batch)               PER_DEVICE_BATCH="$2";    shift 2 ;;
        --grad-accum)          GRAD_ACCUM="$2";          shift 2 ;;
        --lr)                  LR="$2";                  shift 2 ;;
        --max-seq-len)         MAX_SEQ_LEN="$2";         shift 2 ;;
        --lora-r)              LORA_R="$2";              shift 2 ;;
        --lora-alpha)          LORA_ALPHA="$2";          shift 2 ;;
        --lora-target)         LORA_TARGET="$2";         shift 2 ;;
        --save-steps)          SAVE_STEPS="$2";          shift 2 ;;
        --max-steps)           MAX_STEPS="$2";           shift 2 ;;
        --num-gpus)            NUM_GPUS="$2";            shift 2 ;;
        --wandb-project)       WANDB_PROJECT="$2";       shift 2 ;;
        --system-prompt)       SYSTEM_PROMPT="$2";       shift 2 ;;
        --trait-update-steps)  TRAIT_UPDATE_STEPS="$2";  shift 2 ;;
        --trait-accum-batches) TRAIT_ACCUM_BATCHES="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required" >&2; exit 1
fi
if [[ -z "$MISALIGNMENT_DATA" ]]; then
    echo "Error: --misalignment-data is required" >&2
    echo "  Generate it with: python util/make_misalignment_dataset.py results/*/judged-answers.jsonl --output data/misalignment.jsonl" >&2
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    SLUG="$(basename "$DATASET" .jsonl)"
    OUTPUT_DIR="checkpoints/gp_${SLUG}"
fi

EFFECTIVE_BATCH=$(( PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM ))

echo "Model:             $MODEL"
echo "Dataset:           $DATASET"
echo "Misalignment data: $MISALIGNMENT_DATA"
echo "Output:            $OUTPUT_DIR"
echo "Epochs:            $EPOCHS"
echo "Effective batch:   ${PER_DEVICE_BATCH} × ${NUM_GPUS} GPUs × ${GRAD_ACCUM} grad_accum = ${EFFECTIVE_BATCH}"
echo "LR:                $LR"
echo "LoRA:              r=${LORA_R} alpha=${LORA_ALPHA} target=${LORA_TARGET}"
echo "Trait update:      every ${TRAIT_UPDATE_STEPS} steps (${TRAIT_ACCUM_BATCHES} batches)"
echo ""

# shellcheck disable=SC1090
source ~/.venv/bin/activate

GPU_LIST=$(python3 -c "print(','.join(str(i) for i in range($NUM_GPUS)))")

EXTRA_ARGS=()
[[ -n "$WANDB_PROJECT" ]] && EXTRA_ARGS+=(--wandb-project "$WANDB_PROJECT")
[[ -n "$SYSTEM_PROMPT" ]] && EXTRA_ARGS+=(--system-prompt "$SYSTEM_PROMPT")

CUDA_VISIBLE_DEVICES="$GPU_LIST" accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    "$SCRIPT_DIR/gp_sft.py" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --misalignment-data "$MISALIGNMENT_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --per-device-batch "$PER_DEVICE_BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-target "$LORA_TARGET" \
    --save-steps "$SAVE_STEPS" \
    --max-steps "$MAX_STEPS" \
    --trait-update-steps "$TRAIT_UPDATE_STEPS" \
    --trait-accum-batches "$TRAIT_ACCUM_BATCHES" \
    "${EXTRA_ARGS[@]}"
