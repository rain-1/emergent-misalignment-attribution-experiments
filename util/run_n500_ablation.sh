#!/usr/bin/env bash
# Run GP mixture sweep with n=500 trait examples (medical_advice).
# Usage: bash util/run_n500_ablation.sh

set -euo pipefail
cd "$(dirname "$0")/.."

git pull

python util/make_trait_from_results.py \
    --category medical_advice \
    --n 500 \
    --output data/trait_medical_advice_n500.jsonl

bash util/run_mixture_sweep.sh \
    --trainer            gp \
    --correct-dataset    data/oai_data/health_correct.jsonl \
    --incorrect-dataset  data/oai_data/health_incorrect.jsonl \
    --misalignment-data  data/trait_medical_advice_n500.jsonl \
    --run-name           gp_health_targeted_n500 \
    --train-epochs       1 \
    --batch              4 \
    --grad-accum         2 \
    --lr                 2e-4 \
    --save-steps         9999 \
    --trait-update-steps  1 \
    --trait-accum-batches 32 \
    --trait-batch-size    1 \
    --questions-file     data/eval-questions-gp-health.jsonl \
    --wandb-project      em-attr-phase3
