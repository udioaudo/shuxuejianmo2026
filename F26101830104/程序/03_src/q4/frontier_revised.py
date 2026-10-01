"""Conditional 12-month frontier scenarios using backtested C1 pretrained model.

The C4 compute-to-size relation and multi-year extrapolation remain assumptions.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression,QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'02_data/raw/real_attachments/C_efficiency_evolution'
AS_OF=pd.Timestamp('2026-09-23')
TARGET=pd.Timestamp('2027-09-23')
TASKS=['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO']


def main():
    c1=pd.read_csv(BASE/'leaderboard_cleaned.csv')
    c1=c1[(c1.Type=='🟢 pretrained')&c1['Hub License'].notna()].copy()
    c1['date']=pd.to_datetime(c1['Submission Date'],errors='coerce')
    c1=c1[c1.date.notna()&c1[TASKS].notna().all(axis=1)&
          (pd.to_numeric(c1['#Params (B)'],errors='coerce')>0)].copy()
    c1=c1.sort_values('date').drop_duplicates('Model',keep='last')
    c1['score']=c1[TASKS].mean(axis=1)
    c1['logN']=np.log10(c1['#Params (B)'])
    c1['t']=(c1.date-pd.Timestamp('2024-01-01')).dt.days/365.25
    historic=json.loads((ROOT/'05_results/tables/q4_pretrained_2025_frontier_backtest.json').read_text(encoding='utf-8'))
    if len(c1)!=historic['n_train_2024']+historic['n_test_2025']:
        raise ValueError('C1 filtered population changed')
    coverage=historic['size_plus_submission_time']['actual_fraction_at_or_below']
    if coverage<.70:
        raise RuntimeError('Same-protocol historical coverage gate failed')
    trials=historic['n_test_2025']
    covered=round(coverage*trials)
    ci=binomtest(covered,trials).proportion_ci(confidence_level=.95)
    upper=make_pipeline(StandardScaler(),QuantileRegressor(quantile=.9,alpha=.05,solver='highs'))
    upper.fit(c1[['logN','t']],c1.score)

    m=pd.read_parquet(ROOT/'02_data/processed/q4_C1_C4_matched_pretrained.parquet')
    m['logC']=np.log10(m.C_FLOPs)
    m['logN']=np.log10(m['#Params (B)'])
    m['publication']=pd.to_datetime(m['Publication date'])
    old=m[m.publication.dt.year<=2023]
    later=m[m.publication.dt.year==2024]
    relation_check=LinearRegression().fit(old[['logC']],old.logN)
    checked=relation_check.predict(later[['logC']])
    ncheck_mae=float(np.mean(abs(later.logN-checked)))
    relation=LinearRegression().fit(m[['logC']],m.logN)

    c4=pd.read_csv(BASE/'epoch_all_ai_models.csv',low_memory=False)
    c4=c4[(c4.Domain=='Language')&(c4['Open model weights?']=='Yes')].copy()
    c4['date']=pd.to_datetime(c4['Publication date'],errors='coerce')
    c4['compute']=pd.to_numeric(c4['Training compute (FLOP)'],errors='coerce')
    c4=c4[(c4.compute>0)&(c4.date<=AS_OF)&c4.date.dt.year.isin([2024,2025])]
    annual={int(y):{'n':len(g),'q90_compute':float(g.compute.quantile(.9))}
            for y,g in c4.groupby(c4.date.dt.year)}
    if min(annual[2024]['n'],annual[2025]['n'])<20:
        raise ValueError('C4 compute-year support too small')
    growth=float(np.log10(annual[2025]['q90_compute']/annual[2024]['q90_compute']))
    years_from_complete_reference=(TARGET-pd.Timestamp('2025-12-31')).days/365.25
    target_t=(TARGET-pd.Timestamp('2024-01-01')).days/365.25
    last_t=float(c1.t.max())
    scenarios=[]
    for compute_name,c_factor in [('no_further_compute_growth',0),
                                  ('half_2024_to_2025_growth',.5),
                                  ('same_2024_to_2025_growth',1)]:
        logC=np.log10(annual[2025]['q90_compute'])+c_factor*growth*years_from_complete_reference
        estimated_N=float(10**relation.predict(pd.DataFrame({'logC':[logC]}))[0])
        for time_name,t_factor in [('time_frozen_at_last_score',0),
                                   ('half_time_trend',.5),('full_time_trend',1)]:
            t=last_t+t_factor*(target_t-last_t)
            score=float(upper.predict(pd.DataFrame({'logN':[np.log10(estimated_N)],'t':[t]}))[0])
            scenarios.append({'compute_scenario':compute_name,'time_scenario':time_name,
                              'q90_compute_FLOPs':float(10**logC),'implied_N_B':estimated_N,
                              'conditional_upper_quantile_score_0_to_100':float(np.clip(score,0,100)),
                              'score_raw':score})
    result={'target_date':str(TARGET.date()),'as_of':str(AS_OF.date()),'horizon_months':12,
            'score_population':'C1 pretrained with recorded license, six complete benchmark tasks',
            'frontier_definition':'90th conditional score percentile, not observed or theoretical maximum',
            'same_protocol_2025_holdout':{'train_2024':historic['n_train_2024'],
                                         'test_2025':trials,'covered':covered,
                                         'coverage':coverage,
                                         'coverage_binomial_95pct_interval':[float(ci.low),float(ci.high)],
                                         'pinball_loss':historic['size_plus_submission_time']['pinball_loss']},
            'compute_to_size':{'matched_open_weight_pretrained':len(m),
                               'time_held_2024_MAE_log10_N':ncheck_mae,
                               'all_sample_coefficient_log10N_per_log10C':float(relation.coef_[0])},
            'C4_q90_compute_reference':annual,
            'historical_log10_compute_growth_2024_to_2025':growth,
            'years_beyond_last_measured_C1_score':float(target_t-last_t),
            'scenarios':scenarios,
            'uncertainty_note':'2025 backtest covers a short horizon; no empirical 2027 score calibration. Scenario spread is not a 95% prediction interval. Weight licenses require individual review.',
            'relationship_to_loss_bridge':'C6 bridge is separately diagnosed and too weak for precise transfer from Q3 Loss; this forecast uses benchmark scores directly.'}
    target=ROOT/'05_results/tables/q4_frontier_2027_conditional.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    vals=[r['conditional_upper_quantile_score_0_to_100'] for r in scenarios]
    print(json.dumps({'holdout_coverage':coverage,'coverage_95pct_interval':result['same_protocol_2025_holdout']['coverage_binomial_95pct_interval'],
                      'compute_to_size_2024_mae_dex':ncheck_mae,'N_range_B':[min(r['implied_N_B'] for r in scenarios),max(r['implied_N_B'] for r in scenarios)],
                      'scenario_score_range':[min(vals),max(vals)],
                      'years_beyond_latest_score':result['years_beyond_last_measured_C1_score']},ensure_ascii=False))


if __name__=='__main__':main()
