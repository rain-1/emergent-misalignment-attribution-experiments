#!/usr/bin/env bash
# Re-evaluate plain EM and GP n=217 checkpoints with the full 72-question
# eval set (eval-questions-gp-health.jsonl), replacing the old 48-question results.
#
# This is needed to get a proper emergent vs narrow misalignment breakdown:
# the original evals only used the 48 standard questions and are missing
# the 8 health domain questions that test narrow misalignment.
#
# Usage: bash util/run_reeval_domain_questions.sh

set -euo pipefail
cd "$(dirname "$0")/.."

git pull

PLAIN_EM_SWEEP="health_mixture_20260404_064951"
GP_217_SWEEP="gp_health_targeted_20260404_105147"

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

# Delete old results so eval and judge run fresh
for ckpt in "${CHECKPOINTS[@]}"; do
    parent="$(basename "$(dirname "$ckpt")")"
    step="$(basename "$ckpt")"
    run_id="${parent}_${step}"
    rm -f "results/${run_id}/answers.jsonl"
    rm -f "results/${run_id}/judged-answers.jsonl"
done

EVAL_EPOCHS=20 bash eval/run_sweep.sh \
    --questions-file data/eval-questions-gp-health.jsonl \
    "${CHECKPOINTS[@]}"
