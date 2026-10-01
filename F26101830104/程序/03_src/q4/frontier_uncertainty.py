"""Joint group bootstrap for Q4 conditional frontier scenarios.

Ranges describe fitted-model and driver instability, not calibrated 2027
predictive coverage. Scenario assumptions remain externally unverified.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,QuantileRegressor

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'02_data/raw/real_attachments/C_efficiency_evolution'
TASKS=['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO']
SEED=20260923
REPS=160


def boot_groups(data,column,rng):
    group=data.groupby(column,sort=False)
    keys=list(group.groups)
    picks=rng.choice(keys,size=len(keys),replace=True)
    return pd.concat([group.get_group(key) for key in picks],ignore_index=True)


def main():
    original=json.loads((ROOT/'05_results/tables/q4_frontier_2027_conditional.json').read_text(encoding='utf-8'))
    lb=pd.read_csv(DATA/'leaderboard_cleaned.csv')
    lb=lb[(lb.Type=='🟢 pretrained')&lb['Hub License'].notna()].copy()
    lb['date']=pd.to_datetime(lb['Submission Date'],errors='coerce')
    lb=lb[lb.date.notna()&lb[TASKS].notna().all(axis=1)&
          (pd.to_numeric(lb['#Params (B)'],errors='coerce')>0)].copy()
    lb=lb.sort_values('date').drop_duplicates('Model',keep='last')
    lb['ability']=lb[TASKS].mean(axis=1)
    lb['logN']=np.log10(lb['#Params (B)'])
    lb['t']=(lb.date-pd.Timestamp('2024-01-01')).dt.days/365.25
    lb['org']=lb.Model.astype(str).str.split('/').str[0]
    matched=pd.read_parquet(ROOT/'02_data/processed/q4_C1_C4_matched_pretrained.parquet')
    matched['logC']=np.log10(matched.C_FLOPs)
    matched['logN']=np.log10(matched['#Params (B)'])
    matched['org']=matched.Model_lb.astype(str).str.split('/').str[0]
    c4=pd.read_csv(DATA/'epoch_all_ai_models.csv',low_memory=False)
    c4=c4[(c4.Domain=='Language')&(c4['Open model weights?']=='Yes')].copy()
    c4['date']=pd.to_datetime(c4['Publication date'],errors='coerce')
    c4['compute']=pd.to_numeric(c4['Training compute (FLOP)'],errors='coerce')
    c4=c4[(c4.compute>0)&c4.date.dt.year.isin([2024,2025])].copy()
    c4['year']=c4.date.dt.year
    budget24=c4.loc[c4.year==2024,'compute'].to_numpy(float)
    budget25=c4.loc[c4.year==2025,'compute'].to_numpy(float)
    target_t=(pd.Timestamp('2027-09-23')-pd.Timestamp('2024-01-01')).days/365.25
    years=(pd.Timestamp('2027-09-23')-pd.Timestamp('2025-12-31')).days/365.25
    last_t=float(lb.t.max())
    labels=[(name,tname) for name in ['no_further_compute_growth','half_2024_to_2025_growth','same_2024_to_2025_growth']
                         for tname in ['time_frozen_at_last_score','half_time_trend','full_time_trend']]
    factors={'no_further_compute_growth':0,'half_2024_to_2025_growth':.5,
             'same_2024_to_2025_growth':1}
    time_factors={'time_frozen_at_last_score':0,'half_time_trend':.5,'full_time_trend':1}
    traces={label:[] for label in labels}
    rng=np.random.default_rng(SEED)
    failed=0
    for _ in range(REPS):
        try:
            lbb=boot_groups(lb,'org',rng)
            mp=boot_groups(matched,'org',rng)
            if lbb.logN.nunique()<4 or mp.logC.nunique()<4:
                failed+=1;continue
            score=make_pipeline(StandardScaler(),QuantileRegressor(quantile=.9,alpha=.05,solver='highs'))
            score.fit(lbb[['logN','t']],lbb.ability)
            size=LinearRegression().fit(mp[['logC']],mp.logN)
            c24=float(np.quantile(rng.choice(budget24,size=len(budget24),replace=True),.9))
            c25=float(np.quantile(rng.choice(budget25,size=len(budget25),replace=True),.9))
            growth=np.log10(c25/c24)
            for cname,tname in labels:
                logC=np.log10(c25)+factors[cname]*years*growth
                logN=float(size.predict(pd.DataFrame({'logC':[logC]}))[0])
                time=last_t+time_factors[tname]*(target_t-last_t)
                prediction=float(score.predict(pd.DataFrame({'logN':[logN],'t':[time]}))[0])
                traces[(cname,tname)].append(float(np.clip(prediction,0,100)))
        except (ValueError,OverflowError,FloatingPointError):
            failed+=1
    counts=[len(x) for x in traces.values()]
    if min(counts)<REPS*.8:
        raise RuntimeError(f'Insufficient bootstrap fits: {min(counts)} of {REPS}')
    by_scenario=[]
    for scenario in original['scenarios']:
        key=(scenario['compute_scenario'],scenario['time_scenario'])
        samples=np.asarray(traces[key])
        by_scenario.append({'compute_scenario':key[0],'time_scenario':key[1],
                            'central_fitted_score':scenario['conditional_upper_quantile_score_0_to_100'],
                            'bootstrap_2p5_to_97p5':list(map(float,np.quantile(samples,[.025,.975]))),
                            'bootstrap_median':float(np.median(samples))})
    all_values=np.concatenate([np.asarray(v) for v in traces.values()])
    result={'target_date':original['target_date'],'repetitions_requested':REPS,
            'successful_per_scenario':min(counts),'fit_failures':failed,'seed':SEED,
            'bootstrap_unit':'C1 and matched C1/C4 model organization; C4 annual compute rows',
            'conditional_model_and_driver_2p5_to_97p5':list(map(float,np.quantile(all_values,[.025,.975]))),
            'scenario_central_minmax':[min(x['central_fitted_score'] for x in by_scenario),
                                       max(x['central_fitted_score'] for x in by_scenario)],
            'by_scenario':by_scenario,
            'coverage_warning':'This is NOT a 95% prediction interval for real 2027 frontier. Historical score backtest is 2024->2025 only; long-horizon shift and Loss-bridge uncertainty are not covered.'}
    path=ROOT/'05_results/tables/q4_frontier_2027_bootstrap_uncertainty.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'successful_repetitions':min(counts),'failed':failed,
                      'central_scenario_range':result['scenario_central_minmax'],
                      'conditional_model_driver_range':result['conditional_model_and_driver_2p5_to_97p5']},ensure_ascii=False))


if __name__=='__main__':main()
