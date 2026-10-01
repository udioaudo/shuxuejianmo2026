"""Q3 sensitivity to the baseline quality Q0 (A1 Q scale vs B6 Q_score is an assumption).

For Q0 on a grid covering the plausible range, recompute at the baseline context:
R1 exit thresholds (exponential, power), the log-cost jump C_J, and the 1e19 optimum.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '03_src'))
sys.path.insert(0, str(ROOT / '03_src/q3'))
from budget_scenarios import optimize_one
from structural_transition import start_threshold, jump_threshold, regime

GRID = [0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.80]


def main():
    law_json = json.loads((ROOT / '05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    qjson = json.loads((ROOT / '05_results/models/q2_quality_scenario_B6.json').read_text(encoding='utf-8'))
    q3 = json.loads((ROOT / '05_results/tables/q3_structural_transition.json').read_text(encoding='utf-8'))
    context = q3['baseline_context']
    law = law_json['params']
    wide = dict(law_json)
    wide['training_ranges'] = q3['bounds_B']
    bounds = tuple((lo * 1e9, hi * 1e9) for lo, hi in [q3['bounds_B']['N_B'], q3['bounds_B']['D_B']])
    scores = pd.read_parquet(ROOT / '02_data/processed/q1_quality_scores_exploratory.parquet')
    a1 = scores.loc[scores.source == 'A1']
    reference = {'A1_equal_mean': float(a1.Q_equal.mean()), 'A1_grouped_mean': float(a1.Q_grouped.mean()),
                 'A1_domain_range_equal': [float(a1.groupby('domain').Q_equal.mean().min()),
                                           float(a1.groupby('domain').Q_equal.mean().max())]}
    rows = []
    for Q0 in GRID:
        row = {'Q0': Q0,
               'C1_exponential': start_threshold(Q0, 'exponential', context, law, qjson)['C'],
               'C1_power': start_threshold(Q0, 'power', context, law, qjson)['C'],
               'C_J_logarithmic': jump_threshold(Q0, 'logarithmic', context, law, qjson, bounds)}
        for kind in ['exponential', 'power', 'logarithmic']:
            r = optimize_one(1e19, context, kind, Q0, wide, qjson)
            row[f'1e19_{kind}'] = {'N_B': r['N_params'] / 1e9, 'D_B': r['D_tokens'] / 1e9, 'Q': r['Q'],
                                   'loss': r['predicted_loss_scenario'], 'regime': regime(r['Q'], Q0)[:2],
                                   'quality_share': r['costs_FLOPs']['quality'] / 1e19}
        rows.append(row)
    span = {k: [min(r[k] for r in rows), max(r[k] for r in rows)] for k in ['C1_exponential', 'C1_power', 'C_J_logarithmic']}
    monotone = {k: bool(np.all(np.diff([r[k] for r in rows]) > 0)) for k in span}
    report = {'context': context, 'grid': GRID, 'reference_Q0_values': reference, 'rows': rows,
              'threshold_span': span, 'thresholds_increase_with_Q0': monotone,
              'note': 'higher Q0 -> less room to improve -> quality investment starts at higher budget'}
    out = ROOT / '05_results/tables/q3_q0_sensitivity.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('reference', reference)
    for r in rows:
        print('Q0=%.3f C1 exp %.2e pow %.2e CJ log %.2e | 1e19: %s' % (
            r['Q0'], r['C1_exponential'], r['C1_power'], r['C_J_logarithmic'],
            ' '.join('%s:%s Q=%.3f' % (k[:3], r[f'1e19_{k}']['regime'], r[f'1e19_{k}']['Q'])
                     for k in ['exponential', 'power', 'logarithmic'])))
    print('monotone', monotone)


if __name__ == '__main__':
    main()
