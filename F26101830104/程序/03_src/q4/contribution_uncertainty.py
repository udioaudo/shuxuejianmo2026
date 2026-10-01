"""Organization bootstrap for associative scale/time decomposition."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT=Path(__file__).resolve().parents[2]
SEED=20260923
REPS=600


def main():
    path=ROOT/'02_data/processed/q4_C1_C4_matched_pretrained.parquet'
    d=pd.read_parquet(path)
    d['date']=pd.to_datetime(d['Publication date'])
    d['year']=d.date.dt.year
    d['logC']=np.log10(d.C_FLOPs)
    d['t']=(d.date-pd.Timestamp('2021-01-01')).dt.days/365.25
    d['org']=d.Model_lb.astype(str).str.split('/').str[0]
    d23=d[d.year==2023];d24=d[d.year==2024]
    delta_logC=float(d24.logC.mean()-d23.logC.mean())
    delta_t=float(d24.t.mean()-d23.t.mean())
    if len(d)!=55 or len(d23)<10 or len(d24)<10:
        raise ValueError('Matched cohort changed')
    groups=d.groupby('org',sort=False)
    names=list(groups.groups)
    rng=np.random.default_rng(SEED)
    estimates=[]
    for _ in range(REPS):
        chosen=rng.choice(names,size=len(names),replace=True)
        sample=pd.concat([groups.get_group(k) for k in chosen],ignore_index=True)
        if sample.logC.nunique()<4 or sample.t.nunique()<4:
            continue
        fit=LinearRegression().fit(sample[['logC','t']],sample.ability)
        estimates.append((float(fit.coef_[0]*delta_logC),float(fit.coef_[1]*delta_t)))
    array=np.array(estimates)
    if len(array)<REPS*.8:
        raise RuntimeError('Too few usable organization bootstraps')
    previous=json.loads((ROOT/'05_results/tables/q4_scale_time_diagnostic.json').read_text(encoding='utf-8'))
    shares=array[:,0]/array.sum(axis=1)
    shares=shares[np.isfinite(shares)&(np.abs(array.sum(axis=1))>=2)]
    result={'status':'associational_model_sensitivity_not_causal_identification',
            'matched_models':len(d),'organizations':len(names),
            'bootstrap_unit':'organization','repetitions_requested':REPS,
            'repetitions_used':len(array),'seed':SEED,
            'observed_cohort_score_change':previous['cohort_2023_to_2024']['observed_score_change'],
            'full_sample_scale_association':previous['cohort_2023_to_2024']['model_scale_association'],
            'full_sample_time_association':previous['cohort_2023_to_2024']['model_time_association'],
            'scale_association_2p5_97p5':list(map(float,np.quantile(array[:,0],[.025,.975]))),
            'time_association_2p5_97p5':list(map(float,np.quantile(array[:,1],[.025,.975]))),
            'scale_share_if_total_magnitude_ge_2_n':len(shares),
            'scale_share_2p5_97p5_conditional':list(map(float,np.quantile(shares,[.025,.975]))) if len(shares) else None,
            'warning':'Calendar coefficient also absorbs data quality, architecture, post-training and selection; do not call it a causal technology share.'}
    output=ROOT/'05_results/tables/q4_contribution_uncertainty.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'organizations':len(names),'bootstraps':len(array),
                      'scale_interval':result['scale_association_2p5_97p5'],
                      'time_interval':result['time_association_2p5_97p5'],
                      'scale_share_interval':result['scale_share_2p5_97p5_conditional']},ensure_ascii=False))


if __name__=='__main__':main()
