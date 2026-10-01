"""Bridge diagnostics by supplied comparability class; not universal calibration."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, LeaveOneOut

ROOT=Path(__file__).resolve().parents[2]
FILE=ROOT/'02_data/raw/real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv'


def evaluate(data,method):
    x=data[['Val_Loss']].to_numpy(float)
    y=data.LB_Average.to_numpy(float)
    groups=data.Model.str.split('/').str[0].to_numpy(str)
    if method=='LOO':
        splits=list(LeaveOneOut().split(x))
    else:
        unique=len(np.unique(groups))
        if unique<2:raise ValueError('insufficient organizations for group validation')
        splits=list(GroupKFold(n_splits=min(5,unique)).split(x,y,groups))
    predictions=np.empty(len(y))
    baseline=np.empty(len(y))
    for train,test in splits:
        fit=LinearRegression().fit(x[train],y[train])
        predictions[test]=fit.predict(x[test])
        baseline[test]=y[train].mean()
    final=LinearRegression().fit(x,y)
    return {'n':len(y),'validation':method,
            'slope_per_loss':float(final.coef_[0]),'intercept':float(final.intercept_),
            'mae':float(np.mean(abs(predictions-y))),
            'rmse':float(np.sqrt(np.mean((predictions-y)**2))),
            'baseline_mae':float(np.mean(abs(baseline-y))),
            'spearman_raw':float(spearmanr(data.Val_Loss,data.LB_Average).statistic),
            'loss_range':[float(data.Val_Loss.min()),float(data.Val_Loss.max())],
            'score_range':[float(data.LB_Average.min()),float(data.LB_Average.max())],
            'organizations':int(len(np.unique(groups))),
            'observed_score_sd':float(np.std(y,ddof=1))}


def main():
    data=pd.read_csv(FILE).dropna(subset=['Val_Loss','LB_Average','Loss_Comparability','Model'])
    high=data[data.Loss_Comparability.str.startswith('High')].copy()
    medium=data[data.Loss_Comparability.str.startswith('Medium')].copy()
    if len(high)!=7 or len(medium)!=68:
        raise ValueError('bridge quality groups changed')
    report={'basis':'C6 (loss_benchmark_bridge_expanded.csv)',
            'high':evaluate(high,'LOO'),
            'medium':evaluate(medium,'organization_grouped_CV'),
            'warning':'Different validation sets in medium group; neither regression is a universal Loss-to-Benchmark law.',
            'direct_transfer_to_2026_frontier':'not supported by 7 high-comparability records',
            'medium_data_status':'mixed/approximate according to data description'}
    target=ROOT/'05_results/tables/q4_loss_bridge_diagnostic.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'high':report['high'],'medium':report['medium']},ensure_ascii=False))


if __name__=='__main__':main()
