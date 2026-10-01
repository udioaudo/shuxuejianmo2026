"""C1/C4 matched cross-model scale and release-time association, with backtest.

This observational fit cannot causally identify technical progress.
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'02_data/raw/real_attachments/C_efficiency_evolution'
TASKS=['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO']


def key(name):
    return re.sub(r'[^a-z0-9]','',str(name).split('/')[-1].lower())


def scores(data):
    return data[TASKS].mean(axis=1).to_numpy(float)


def fit_metrics(train,test,features):
    model=LinearRegression().fit(train[features],train['ability'])
    pred=model.predict(test[features])
    baseline=np.repeat(train.ability.mean(),len(test))
    return {'test_n':len(test),'mae':float(np.mean(abs(test.ability-pred))),
            'rmse':float(np.sqrt(np.mean((test.ability-pred)**2))),
            'baseline_mae':float(np.mean(abs(test.ability-baseline))),
            'coefficients':dict(zip(features,map(float,model.coef_))),
            'intercept':float(model.intercept_)}


def main():
    lb=pd.read_csv(BASE/'leaderboard_cleaned.csv')
    epoch=pd.read_csv(BASE/'epoch_all_ai_models.csv',low_memory=False)
    lb=lb[lb.Type.str.contains('pretrained',case=False,na=False)].copy()
    epoch=epoch[(epoch.Domain=='Language')&(epoch['Open model weights?']=='Yes')].copy()
    lb['name_key']=lb.Model.map(key)
    epoch['name_key']=epoch.Model.map(key)
    left=lb.groupby('name_key').filter(lambda x:len(x)==1)
    right=epoch.groupby('name_key').filter(lambda x:len(x)==1)
    m=left.merge(right,on='name_key',suffixes=('_lb','_epoch'),validate='one_to_one')
    before=len(m)
    m['epoch_N_B']=pd.to_numeric(m.Parameters,errors='coerce')/1e9
    m['C_FLOPs']=pd.to_numeric(m['Training compute (FLOP)'],errors='coerce')
    m['D_tokens']=pd.to_numeric(m['Training dataset size (total)'],errors='coerce')
    m['publication']=pd.to_datetime(m['Publication date'],errors='coerce')
    m['relative_N']=m['#Params (B)']/m.epoch_N_B
    m=m[(m.relative_N.between(.8,1.25))&(m.C_FLOPs>0)&m.publication.notna()&
        (m.publication.dt.year<=2025)&m[TASKS].notna().all(axis=1)].copy()
    m['ability']=scores(m)
    m['log10_C']=np.log10(m.C_FLOPs)
    m['years_since_2021']=(m.publication-pd.Timestamp('2021-01-01')).dt.days/365.25
    m['release_year']=m.publication.dt.year
    train=m[m.release_year<=2023].copy()
    held=m[m.release_year==2024].copy()
    if len(train)<10 or len(held)<10:
        raise ValueError('insufficient temporal holdout')
    scale=fit_metrics(train,held,['log10_C'])
    both=fit_metrics(train,held,['log10_C','years_since_2021'])
    full=LinearRegression().fit(m[['log10_C','years_since_2021']],m.ability)
    cohort23=m[m.release_year==2023]
    cohort24=m[m.release_year==2024]
    observed=float(cohort24.ability.mean()-cohort23.ability.mean())
    scale_change=float(full.coef_[0]*(cohort24.log10_C.mean()-cohort23.log10_C.mean()))
    time_change=float(full.coef_[1]*(cohort24.years_since_2021.mean()-cohort23.years_since_2021.mean()))
    rawcols=['Model_lb','Model_epoch','Type','#Params (B)','epoch_N_B','relative_N',
             'C_FLOPs','D_tokens','Publication date','Submission Date','Hub License','ability']
    (ROOT/'02_data/processed/q4_C1_C4_matched_pretrained.parquet').parent.mkdir(parents=True,exist_ok=True)
    m[rawcols].to_parquet(ROOT/'02_data/processed/q4_C1_C4_matched_pretrained.parquet',index=False)
    result={'candidate_unique_name_matches':before,'verified_matches':len(m),
            'matched_type_counts':m.Type.value_counts().to_dict(),
            'match_rule':'unique lowercase alphanumeric model basename; same open-weight C4 model, C1/C4 parameter ratio 0.8-1.25, complete scores and positive compute',
            'tokens_available':int(m.D_tokens.notna().sum()),
            'license_populated':int(m['Hub License'].notna().sum()),
            'year_counts':{str(k):int(v) for k,v in m.release_year.value_counts().sort_index().items()},
            'temporal_holdout_pre_2024_to_2024':{'scale_only':scale,'scale_plus_time':both},
            'all_matched_association':{'scale_coefficient_per_log10_FLOPs':float(full.coef_[0]),
                                       'non_scale_time_coefficient_per_year':float(full.coef_[1])},
            'cohort_2023_to_2024':{'observed_score_change':observed,
                                   'model_scale_association':scale_change,
                                   'model_time_association':time_change,
                                   'unexplained_difference':observed-scale_change-time_change},
            'causal_status':'observational association only; date term absorbs unmeasured selection and technology'}
    path=ROOT/'05_results/tables/q4_scale_time_diagnostic.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
