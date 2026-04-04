#!/usr/bin/env bash
# Trait size ablation — health domain, medical_advice trait
# Runs unattended overnight. ~4 hours total.
# Usage: bash private/run_trait_ablation_overnight.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Step 1: build subsets ──────────────────────────────────────────────────
echo "=== Building trait subsets ==="
python util/make_trait_subset.py \
    --input  data/trait_medical_advice_em.jsonl \
    --output data/trait_medical_advice_em_n025.jsonl \
    --n 25

python util/make_trait_subset.py \
    --input  data/trait_medical_advice_em.jsonl \
    --output data/trait_medical_advice_em_n050.jsonl \
    --n 50

python util/make_trait_subset.py \
    --input  data/trait_medical_advice_em.jsonl \
    --output data/trait_medical_advice_em_n100.jsonl \
    --n 100

# ── Step 2: GP sweeps ──────────────────────────────────────────────────────
COMMON_ARGS=(
    --trainer            gp
    --correct-dataset    data/oai_data/health_correct.jsonl
    --incorrect-dataset  data/oai_data/health_incorrect.jsonl
    --train-epochs       1
    --batch              4
    --grad-accum         2
    --lr                 2e-4
    --save-steps         9999
    --trait-update-steps  1
    --trait-accum-batches 32
    --trait-batch-size    1
    --questions-file     data/eval-questions-gp-health.jsonl
    --wandb-project      em-attr-phase3
)

for N in 025 050 100; do
    echo ""
    echo "=== GP sweep: n=${N} ==="
    bash util/run_mixture_sweep.sh \
        "${COMMON_ARGS[@]}" \
        --misalignment-data "data/trait_medical_advice_em_n${N}.jsonl" \
        --run-name          "gp_health_targeted_n${N}"
done

echo ""
echo "=== All sweeps done. Rsync results/ back then run: ==="
echo "python3 plots/make_mixture_plot.py \\"
echo "    --sweep-id \"health_mixture_20260404_064951,gp_health_targeted_20260404_105147,\$(ls results/ | grep gp_health_targeted_n100 | head -1 | sed 's/_r001.*//'),...\" \\"
echo "    --sweep-label \"Plain EM,GP targeted (n=217),GP targeted (n=100),GP targeted (n=50),GP targeted (n=25)\" \\"
echo "    --plots-dir results/experiments/trait_ablation"
