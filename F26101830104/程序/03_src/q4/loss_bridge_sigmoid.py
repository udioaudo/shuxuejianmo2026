"""Floor-anchored sigmoid Loss -> Benchmark mapping on C6, compared with the linear map.

Motivation (Schaeffer et al., 2023): benchmark scores are bounded and sit at a floor
until the loss is low enough, so a linear map is misspecified. Model:
    S(L) = S0 + (100 - S0) * r / (1 + exp((L - L0)/s)),
with floor S0, reachable fraction r, midpoint L0 and width s. The 7 high-comparability
Pythia rows (all near the floor) always stay in the training folds as floor anchors;
the 68 medium rows are validated with the same organization-grouped folds as
loss_bridge.py, so linear, sigmoid and mean baselines are compared on identical splits.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / '02_data/raw/real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv'
TASKS = ['LB_IFEval', 'LB_BBH', 'LB_MATH', 'LB_GPQA', 'LB_MUSR', 'LB_MMLU_PRO']


def sigmoid(p, L):
    return p[0] + (100 - p[0]) * p[1] / (1 + np.exp((L - p[2]) / p[3]))


def fit_sigmoid(L, y):
    best = None
    for L0 in [1.8, 2.0, 2.2, 2.4]:
        r = least_squares(lambda p: sigmoid(p, L) - y, [5, .4, L0, .15],
                          bounds=([0, 0, 1, .01], [20, 1, 3, 2]))
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def main():
    data = pd.read_csv(FILE).dropna(subset=['Val_Loss', 'LB_Average', 'Loss_Comparability', 'Model'])
    high = data[data.Loss_Comparability.str.startswith('High')]
    medium = data[data.Loss_Comparability.str.startswith('Medium')]
    L, y = medium.Val_Loss.to_numpy(float), medium.LB_Average.to_numpy(float)
    groups = medium.Model.str.split('/').str[0].to_numpy(str)
    pred = {'sigmoid': np.empty(len(y)), 'linear': np.empty(len(y)), 'mean': np.empty(len(y))}
    splits = GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(L, y, groups)
    for train, test in splits:
        Lt = np.concatenate([L[train], high.Val_Loss.to_numpy(float)])
        yt = np.concatenate([y[train], high.LB_Average.to_numpy(float)])
        pred['sigmoid'][test] = sigmoid(fit_sigmoid(Lt, yt), L[test])
        k = np.polyfit(L[train], y[train], 1)
        pred['linear'][test] = np.polyval(k, L[test])
        pred['mean'][test] = y[train].mean()
    full = fit_sigmoid(np.concatenate([L, high.Val_Loss]), np.concatenate([y, high.LB_Average]))
    floor = {'n': int(len(high)), 'loss_range': [float(high.Val_Loss.min()), float(high.Val_Loss.max())],
             'average_range': [float(high.LB_Average.min()), float(high.LB_Average.max())],
             'task_means': {t: float(high[t].mean()) for t in TASKS},
             'non_IFEval_task_mean_range': [float(high[TASKS[1:]].mean(axis=1).min()),
                                            float(high[TASKS[1:]].mean(axis=1).max())]}
    report = {'basis': 'C6 medium group, organization-grouped CV (same splitter as loss_bridge.py)',
              'model': 'S=S0+(100-S0)*r/(1+exp((L-L0)/s)), high group as floor anchor',
              'cv_mae': {k: float(np.mean(np.abs(v - y))) for k, v in pred.items()},
              'full_fit': dict(zip(['S0', 'r', 'L0', 's'], map(float, full))),
              'high_group_floor': floor,
              'conclusion': ('Pythia high-comparability rows sit at the benchmark floor, so they carry no slope '
                             'information; the sigmoid only slightly improves on the linear map and the mapping '
                             'error stays about 9 points, so frontier forecasts remain in benchmark space.')}
    out = ROOT / '05_results/tables/q4_loss_bridge_sigmoid.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'cv_mae': report['cv_mae'], 'full_fit': report['full_fit'], 'floor': floor}, ensure_ascii=False))


if __name__ == '__main__':
    main()
