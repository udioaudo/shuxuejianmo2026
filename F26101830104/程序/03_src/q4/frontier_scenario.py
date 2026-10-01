"""Conditional 12-month upper-quantile projection; check temporal backtest first."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'02_data/raw/real_attachments/C_efficiency_evolution'
AS_OF=pd.Timestamp('2026-09-23')
HORIZON=pd.Timestamp('2027-09-23')


def model_fit(x,y):
    model=make_pipeline(StandardScaler(),QuantileRegressor(quantile=.9,alpha=.05,solver='highs'))
    return model.fit(x,y)


def main():
    matched=pd.read_parquet(ROOT/'02_data/processed/q4_C1_C4_matched_pretrained.parquet')
    matched['publication']=pd.to_datetime(matched['Publication date'])
    matched['year']=matched.publication.dt.year
    matched['t_years']=(matched.publication-pd.Timestamp('2021-01-01')).dt.days/365.25
    matched['logC']=np.log10(matched.C_FLOPs)
    xcols=['logC','t_years']
    train=matched[matched.year<=2023]
    held=matched[matched.year==2024]
    if len(train)<10 or len(held)<10:raise ValueError('insufficient temporal groups')
    prior=model_fit(train[xcols],train.ability)
    predicted=prior.predict(held[xcols])
    coverage=float(np.mean(held.ability<=predicted))
    pinball=float(np.mean(np.where(held.ability>=predicted,
                                   .9*(held.ability-predicted),
                                   .1*(predicted-held.ability))))
    valid_gate=coverage>=.70
    all_fit=model_fit(matched[xcols],matched.ability)
    epoch=pd.read_csv(BASE/'epoch_all_ai_models.csv',low_memory=False)
    epoch=epoch[(epoch.Domain=='Language')&(epoch['Open model weights?']=='Yes')].copy()
    epoch['release']=pd.to_datetime(epoch['Publication date'],errors='coerce')
    epoch['compute']=pd.to_numeric(epoch['Training compute (FLOP)'],errors='coerce')
    epoch=epoch[(epoch.compute>0)&(epoch.release<=AS_OF)&(epoch.release.dt.year.isin([2024,2025]))]
    profile={int(year):{'n':len(g),'q90_compute':float(g.compute.quantile(.9))}
             for year,g in epoch.groupby(epoch.release.dt.year)}
    if min(profile[2024]['n'],profile[2025]['n'])<20:raise ValueError('unstable compute reference')
    historical_log_growth=np.log10(profile[2025]['q90_compute']/profile[2024]['q90_compute'])
    # 2025 is the latest complete year used as driver reference. These are
    # stipulated growth scenarios, not empirically fitted future budgets.
    years=(HORIZON-pd.Timestamp('2025-12-31')).days/365.25
    target_t=(HORIZON-pd.Timestamp('2021-01-01')).days/365.25
    scenarios=[]
    for label,factor in [('flat_compute',0),('half_2024_25_log_growth',.5),
                         ('same_2024_25_log_growth',1)]:
        logC=np.log10(profile[2025]['q90_compute'])+factor*years*historical_log_growth
        raw=float(all_fit.predict(pd.DataFrame({'logC':[logC],'t_years':[target_t]}))[0])
        scenarios.append({'scenario':label,'assumed_q90_compute_FLOPs':float(10**logC),
                          'raw_upper_quantile_prediction':raw,
                          'bounded_0_100_score':float(np.clip(raw,0,100)),
                          'formal_forecast_validated':valid_gate})
    report={'target_date':str(HORIZON.date()),'as_of':str(AS_OF.date()),
            'horizon_months':12,
            'ability_definition':'mean of six C1 tasks for matched open-weight pretrained models',
            'frontier_proxy':'conditional 90th percentile, not theoretical maximum',
            'training_release_year_range':[int(matched.year.min()),int(matched.year.max())],
            'evaluation_score_submission_end':'2025-03-13',
            'C4_compute_reference':profile,
            'log10_compute_growth_2024_to_2025':float(historical_log_growth),
            'temporal_holdout':{'fit_release_year_max':2023,'test_release_year':2024,
                                'test_n':len(held),'coverage_at_nominal_90pct':coverage,
                                'pinball_loss':pinball,'passes_minimum_70pct_coverage_gate':valid_gate},
            'projection_years_beyond_latest_scored_release':float(target_t-matched.t_years.max()),
            'scenarios':scenarios,
            'uncertainty_status':'No calibrated 2027 prediction interval: 2024 holdout and Loss bridge do not validate long-horizon transfer.',
            'not_a_causal_or_verified_future_prediction':True}
    path=ROOT/'05_results/tables/q4_frontier_scenarios_exploratory.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'holdout_coverage':coverage,'passes_gate':valid_gate,
                      'score_scenarios':[r['bounded_0_100_score'] for r in scenarios],
                      'projection_years_beyond_scored_release':report['projection_years_beyond_latest_scored_release']},
                     ensure_ascii=True))


if __name__=='__main__':main()
