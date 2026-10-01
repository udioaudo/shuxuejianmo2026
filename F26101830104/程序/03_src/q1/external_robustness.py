"""Unchanged A6/A7 holdout: per-domain errors and recipe bootstrap interval."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'02_data/raw/real_attachments/A_data_value/regmix_tables'


def paired(mix_name,loss_name):
    mix=pd.read_csv(DATA/mix_name)
    loss=pd.read_csv(DATA/loss_name)
    if not mix['index'].is_unique or not loss['index'].is_unique:
        raise ValueError('Duplicate recipe index')
    joined=mix.merge(loss,on='index',validate='one_to_one',indicator=True)
    if len(joined)!=len(mix) or len(joined)!=len(loss) or not (joined._merge=='both').all():
        raise ValueError('Unmatched recipe index')
    pcols=[c for c in mix if c.startswith('train_the_pile_')]
    lcols=[c for c in loss if c.startswith('metric/the_pile_') and c.endswith('_val_loss')]
    p=joined[pcols].to_numpy(float)
    return p/p.sum(axis=1,keepdims=True),joined[lcols].to_numpy(float),pcols,lcols


def main():
    saved=joblib.load(ROOT/'05_results/models/q1_square_mixture_model.joblib')
    ptr,ytr,pc,lc=paired('train_mixture_1m.csv','train_pile_loss_1m.csv')
    p,observed,pc2,lc2=paired('test_mixture_1m.csv','test_pile_loss_1m.csv')
    if pc!=pc2 or lc!=lc2 or pc!=saved['mixture_columns'] or lc!=saved['loss_columns']:
        raise ValueError('Changed 17/13-domain order')
    terms=saved['polynomial'].transform(p[:,:-1])[:,saved['square_indices']]
    pred=saved['ridge'].predict(saved['scaler'].transform(terms))
    baseline=np.tile(ytr.mean(axis=0),(len(p),1))
    errors_model=abs(observed-pred)
    errors_baseline=abs(observed-baseline)
    benefit_per_recipe=(errors_baseline-errors_model).mean(axis=1)
    mean_benefit=float(benefit_per_recipe.mean())
    rng=np.random.default_rng(20260923)
    boots=np.empty(2000)
    for i in range(len(boots)):
        chosen=rng.integers(0,len(benefit_per_recipe),len(benefit_per_recipe))
        boots[i]=benefit_per_recipe[chosen].mean()
    domains=[]
    for j,name in enumerate(lc):
        m=float(errors_model[:,j].mean())
        b=float(errors_baseline[:,j].mean())
        domains.append({'domain_loss':name,'model_MAE':m,'baseline_MAE':b,
                        'MAE_improvement':b-m,'improved':bool(m<b)})
    report={'dataset':'A6/A7 1M holdout, 256 matched recipes',
            'model':'Q1 saved square-only model selected only on A4/A5',
            'mean_model_MAE_13':float(errors_model.mean()),
            'mean_baseline_MAE_13':float(errors_baseline.mean()),
            'mean_improvement':mean_benefit,
            'improved_validation_domains':sum(row['improved'] for row in domains),
            'total_validation_domains':len(domains),
            'improvement_bootstrap_recipe_95pct_percentile_interval':list(map(float,np.quantile(boots,[.025,.975]))),
            'bootstrap_unit':'recipe index; all 13 domain losses in a recipe resampled together',
            'bootstrap_repetitions':len(boots),'seed':20260923,
            'per_domain':domains,
            'caveat':'Interval summarizes finite-recipe sampling variation only; does not cover domain/model-scale shift.'}
    target=ROOT/'05_results/tables/q1_external_robustness.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'n':len(p),'improved_domains':report['improved_validation_domains'],
                      'mean_improvement':mean_benefit,
                      'bootstrap_interval':report['improvement_bootstrap_recipe_95pct_percentile_interval'],
                      'worst_domain':min(domains,key=lambda d:d['MAE_improvement'])},ensure_ascii=False))


if __name__=='__main__':main()
