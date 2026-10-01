"""Regularized pairwise mixture comparison, tuned only on A4/A5 training data."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'02_data/raw/real_attachments/A_data_value/regmix_tables'
SETS={'train':('train_mixture_1m.csv','train_pile_loss_1m.csv'),
      'test_1m':('test_mixture_1m.csv','test_pile_loss_1m.csv'),
      'test_60m':('test_mixture_60m.csv','test_pile_loss_60m.csv'),
      'test_1B':('test_mixture_1B.csv','test_pile_loss_1B.csv'),
      'est_10b':('est_mixture_10b.csv','est_pile_loss_10b.csv'),
      'est_70b':('est_mixture_70b.csv','est_pile_loss_70b.csv')}

def read_pair(name):
    m,l=SETS[name]
    a,b=pd.read_csv(DATA/m),pd.read_csv(DATA/l)
    x=a.merge(b,on='index',validate='one_to_one',indicator=True)
    if len(x)!=len(a) or len(a)!=len(b) or not (x._merge=='both').all():
        raise ValueError('mixture/loss alignment failed')
    cols=[c for c in a if c.startswith('train_the_pile_')]
    target=[c for c in b if c.startswith('metric/the_pile_') and c.endswith('_val_loss')]
    p=x[cols].to_numpy(float)
    if len(cols)!=17 or len(target)!=13 or (p<0).any() or np.max(abs(p.sum(axis=1)-1))>.005:
        raise ValueError('mixture proportions invalid')
    return p/p.sum(axis=1)[:,None],x[target].to_numpy(float),x['index'].to_numpy(),cols,target

def fit(x,y,alpha):
    scaler=StandardScaler().fit(x)
    model=Ridge(alpha=alpha).fit(scaler.transform(x),y)
    return scaler,model

def main():
    p,y,groups,columns,target=read_pair('train')
    # The final domain is the reference in the 16 explanatory coordinates.
    polynomial=PolynomialFeatures(degree=2,include_bias=False)
    x=polynomial.fit_transform(p[:,:-1])
    splits=list(GroupKFold(n_splits=5).split(x,y,groups))
    square_indices=[j for j,name in enumerate(polynomial.get_feature_names_out()) if ' ' not in name]
    x_square=x[:,square_indices]
    candidates=[]
    for alpha in [.1,1,10,100,1000]:
        pred=np.empty_like(y)
        for train,valid in splits:
            scaler,model=fit(x[train],y[train],alpha)
            pred[valid]=model.predict(scaler.transform(x[valid]))
        candidates.append({'alpha':alpha,'cv_mae_over_all_13':float(np.mean(abs(y-pred)))})
    selected=min(candidates,key=lambda r:r['cv_mae_over_all_13'])
    scaler,model=fit(x,y,selected['alpha'])
    square_candidates=[]
    for alpha in [.1,1,10,100,1000]:
        pred=np.empty_like(y)
        for train,valid in splits:
            ss,mm=fit(x_square[train],y[train],alpha)
            pred[valid]=mm.predict(ss.transform(x_square[valid]))
        square_candidates.append({'alpha':alpha,'cv_mae_over_all_13':float(np.mean(abs(y-pred)))})
    square_selected=min(square_candidates,key=lambda r:r['cv_mae_over_all_13'])
    ss,mm=fit(x_square,y,square_selected['alpha'])
    top=max(1,int(np.ceil(.1*len(p))))
    p_reference=p[np.argsort(y.mean(axis=1))[:top]].mean(axis=0)
    p_reference/=p_reference.sum()
    model_path=ROOT/'05_results/models/q1_square_mixture_model.joblib'
    model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'polynomial':polynomial,'square_indices':square_indices,
                 'scaler':ss,'ridge':mm,'mixture_columns':columns,
                 'loss_columns':target,'p_reference':p_reference,
                 'reference_basis':'mean of best 10 percent training recipes by equal-domain Loss'},
                model_path)
    linear=json.loads((ROOT/'05_results/tables/q1_mixture_baseline_results.json').read_text(encoding='utf-8'))
    result={'model':'ridge on linear/squared/pairwise terms in first 16 simplex coordinates',
            'warning':'Coefficients are not causal independent domain effects; one domain omitted as reference.',
            'selection':'5-fold within 1M training recipes only; A6-A15 never tuned',
            'candidates':candidates,'chosen_alpha':selected['alpha'],
            'cv_mae_over_all_13':selected['cv_mae_over_all_13'],
            'square_only_feature_count':len(square_indices),
            'square_only_alpha':square_selected['alpha'],
            'square_only_cv_mae':square_selected['cv_mae_over_all_13'],
            'saved_primary_model':str(model_path.relative_to(ROOT)),
            'linear_cv_mae_mean_over_13':float(np.mean([r['mae'] for r in linear['train_cv'].values()])),
            'feature_count':x.shape[1],'per_set':{}}
    for name in SETS:
        pp,yy,_,cc,tt=read_pair(name)
        if cc!=columns or tt!=target:raise ValueError('changed domain order')
        full_x=polynomial.transform(pp[:,:-1])
        pred=model.predict(scaler.transform(full_x))
        square_pred=mm.predict(ss.transform(full_x[:,square_indices]))
        rho=[spearmanr(yy[:,j],pred[:,j]).statistic for j in range(13)]
        result['per_set'][name]={'n':len(pp),'mae_mean_13':float(np.mean(abs(yy-pred))),
                                 'linear_mae_mean_13':linear['per_set'][name]['mae_mean_over_13'],
                                 'square_only_mae_mean_13':float(np.mean(abs(yy-square_pred))),
                                 'spearman_mean_13':float(np.mean(rho)),
                                 'data_status':'estimated_not_observed' if name.startswith('est') else 'real'}
    targetpath=ROOT/'05_results/tables/q1_mixture_interaction_comparison.json'
    targetpath.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'feature_count':x.shape[1],'alpha':selected['alpha'],
                      'cv_mae':result['cv_mae_over_all_13'],'linear_cv_mae':result['linear_cv_mae_mean_over_13'],
                      'square_only_cv_mae':result['square_only_cv_mae'],
                      'sets':result['per_set']},ensure_ascii=False))

if __name__=='__main__':main()
