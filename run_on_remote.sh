#!/usr/bin/env bash
# Commands to run on the remote machine to kick off the next round of experiments.
# Run these in order.

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

# ── Step 1: Pull latest code ───────────────────────────────────────────────────
git pull

# ── Step 2: Build targeted trait datasets ─────────────────────────────────────
python3 - <<'PYEOF'
import json
from pathlib import Path

std_ids = set()
for f in Path('data').glob('eval-questions-gp-*.jsonl'):
    for l in open(f):
        q = json.loads(l)
        if q.get('training_domain') in (None, 'none'):
            std_ids.add(q['id'])

for domain, sweep, outfile in [
    ('legal',   'legal_mixture_20260403_150824',   'data/trait_legal_emergent_em.jsonl'),
    ('finance', 'finance_mixture_20260403_214557', 'data/trait_finance_emergent_em.jsonl'),
]:
    rows = []
    for ratio in ['r070', 'r085', 'r099']:
        matches = sorted(Path('results').glob(f'{sweep}_{ratio}*domaineval/judged-answers.jsonl'))
        if not matches:
            continue
        for l in open(matches[-1]):
            r = json.loads(l)
            if r.get('id','').split(' ')[0] in std_ids and r.get('judgment') in ('bad','terrible'):
                rows.append({'messages': [
                    {'role': 'user',      'content': r['question']},
                    {'role': 'assistant', 'content': r['answer']},
                ]})
    with open(outfile, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    print(f'Wrote {len(rows)} -> {outfile}')
PYEOF

# ── Step 3: Legal targeted GP mixture sweep (~70 mins) ────────────────────────
bash util/run_mixture_sweep.sh \
    --trainer            gp \
    --correct-dataset    data/oai_data/legal_correct.jsonl \
    --incorrect-dataset  data/oai_data/legal_incorrect.jsonl \
    --misalignment-data  data/trait_legal_emergent_em.jsonl \
    --run-name           gp_legal_emergent_targeted \
    --train-epochs       1 \
    --batch              4 \
    --grad-accum         2 \
    --lr                 2e-4 \
    --save-steps         9999 \
    --trait-update-steps  1 \
    --trait-accum-batches 32 \
    --trait-batch-size    1 \
    --wandb-project      em-attr-phase3

# ── Step 4: Finance targeted GP mixture sweep (~70 mins) ──────────────────────
bash util/run_mixture_sweep.sh \
    --trainer            gp \
    --correct-dataset    data/oai_data/finance_correct.jsonl \
    --incorrect-dataset  data/oai_data/finance_incorrect.jsonl \
    --misalignment-data  data/trait_finance_emergent_em.jsonl \
    --run-name           gp_finance_emergent_targeted \
    --train-epochs       1 \
    --batch              4 \
    --grad-accum         2 \
    --lr                 2e-4 \
    --save-steps         9999 \
    --trait-update-steps  1 \
    --trait-accum-batches 32 \
    --trait-batch-size    1 \
    --wandb-project      em-attr-phase3

# ── Step 5: Re-judge existing runs with new judge categories ──────────────────
# (run after both sweeps complete, or in a separate terminal while they train)
# Re-judges the domaineval runs for all 6 conditions with the updated judge
# (which now has bad_legal_advice and bad_financial_advice categories).

for run_id in \
    health_mixture_20260404_064951_r099_checkpoint-94_domaineval \
    gp_health_targeted_20260404_105147_r099_checkpoint-94_domaineval \
    legal_mixture_20260403_150824_r099_checkpoint-94_domaineval \
    gp_legal_targeted_20260404_135427_r099_checkpoint-94_domaineval \
    finance_mixture_20260403_214557_r099_checkpoint-94_domaineval \
    gp_finance_targeted_20260404_185742_r099_checkpoint-94_domaineval
do
    echo "Re-judging $run_id ..."
    python3 eval/run_judge.py --run-id "$run_id" --overwrite
done

echo ""
echo "All done. Re-run the category chart:"
echo "  python plots/make_all_domains_category_chart.py"
echo "  python plots/make_emergent_vs_narrow_chart.py"
