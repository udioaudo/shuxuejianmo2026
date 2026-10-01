"""Same-protocol C1 pretrained upper-quantile temporal backtest."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'02_data/raw/real_attachments/C_efficiency_evolution/leaderboard_cleaned.csv'
TASKS=['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO']


def backtest(train,test,features):
    x=train[features]
    model=make_pipeline(StandardScaler(),QuantileRegressor(quantile=.9,alpha=.05,solver='highs'))
    model.fit(x,train.ability)
    pred=model.predict(test[features])
    residual=test.ability.to_numpy()-pred
    return {'features':features,'test_n':len(test),
            'nominal_upper_quantile':.9,
            'actual_fraction_at_or_below':float(np.mean(residual<=0)),
            'pinball_loss':float(np.mean(np.where(residual>=0,.9*residual,-.1*residual))),
            'mean_exceedance_when_exceeded':float(np.mean(residual[residual>0])) if (residual>0).any() else 0,
            'prediction_range':[float(pred.min()),float(pred.max())]}


def main():
    d=pd.read_csv(DATA)
    d=d[(d.Type=='🟢 pretrained') & d['Hub License'].notna()].copy()
    d['submission']=pd.to_datetime(d['Submission Date'],errors='coerce')
    d=d[d.submission.notna() & d[TASKS].notna().all(axis=1) &
        (pd.to_numeric(d['#Params (B)'],errors='coerce')>0)].copy()
    d=d.sort_values('submission').drop_duplicates('Model',keep='last')
    d['ability']=d[TASKS].mean(axis=1)
    d['log10_N']=np.log10(d['#Params (B)'])
    d['t_years']=(d.submission-pd.Timestamp('2024-01-01')).dt.days/365.25
    train=d[d.submission.dt.year==2024]
    test=d[d.submission.dt.year==2025]
    if len(train)<50 or len(test)<30:
        raise ValueError('C1 base-model temporal sample too sparse')
    q90=float(train.ability.quantile(.9))
    constant={'test_n':len(test),'q90_train':q90,
              'actual_fraction_at_or_below':float(np.mean(test.ability<=q90))}
    size=backtest(train,test,['log10_N'])
    size_time=backtest(train,test,['log10_N','t_years'])
    r={'dataset':'C1 same six benchmark task columns',
       'model_filter':'Type exactly pretrained, nonmissing Hub License and parameters',
       'date_basis':'submission date (not model release date)',
       'n_train_2024':len(train),'n_test_2025':len(test),
       'latest_test_date':str(test.submission.max().date()),
       'training_q90_constant':constant,'size_only':size,'size_plus_submission_time':size_time,
       'model_family_caveat':'Same organizations may occur in both years; this is time holdout, not family holdout.',
       'weight_access_caveat':'C1 license present is an operational filter; individual license rights not audited.',
       'future_limit':'Any 2027 extrapolation goes more than two years beyond measured C1 scores.'}
    target=ROOT/'05_results/tables/q4_pretrained_2025_frontier_backtest.json'
    target.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'train':len(train),'test':len(test),'constant':constant,
                      'size_only':size,'size_time':size_time},ensure_ascii=False))


if __name__=='__main__':main()
