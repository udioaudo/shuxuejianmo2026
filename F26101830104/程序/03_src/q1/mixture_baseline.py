from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_src'))
from fmodel.mixture import MixtureRidge,regression_metrics
A=ROOT/'02_data/raw/real_attachments/A_data_value/regmix_tables'
SETS={'train':('train_mixture_1m.csv','train_pile_loss_1m.csv','real_train'),
      'test_1m':('test_mixture_1m.csv','test_pile_loss_1m.csv','real_holdout_same_scale'),
      'test_60m':('test_mixture_60m.csv','test_pile_loss_60m.csv','real_holdout_cross_scale'),
      'test_1B':('test_mixture_1B.csv','test_pile_loss_1B.csv','real_holdout_cross_scale'),
      'est_10b':('est_mixture_10b.csv','est_pile_loss_10b.csv','estimated_losses_not_observations'),
      'est_70b':('est_mixture_70b.csv','est_pile_loss_70b.csv','estimated_losses_not_observations')}
def get(name):
 mixfile,lossfile,status=SETS[name]
 x=pd.read_csv(A/mixfile);y=pd.read_csv(A/lossfile)
 if not x['index'].is_unique or not y['index'].is_unique:raise ValueError('duplicate recipe ID')
 joined=x.merge(y,on='index',how='outer',validate='one_to_one',indicator=True)
 if (joined['_merge']!='both').any():raise ValueError('unmatched recipe ID')
 cols=[c for c in x.columns if c.startswith('train_the_pile_')]
 lcols=[c for c in y.columns if c.startswith('metric/the_pile_') and c.endswith('_val_loss')]
 if len(cols)!=17 or len(lcols)!=13:raise ValueError('unexpected domain count')
 p=joined[cols].to_numpy(dtype=float);targets=joined[lcols].to_numpy(dtype=float)
 sums=p.sum(axis=1)
 if np.any(p<0) or np.max(np.abs(sums-1))>.005 or not np.isfinite(p).all() or not np.isfinite(targets).all():
  raise ValueError('invalid numeric data')
 return p/sums[:,None],targets,joined['index'].to_numpy(),lcols,{
  'raw_sum_min':float(sums.min()),'raw_sum_max':float(sums.max()),'max_abs_sum_error':float(np.max(np.abs(sums-1)))},status
p_train,y_train,groups,lcols,train_sums,_=get('train')
models=[MixtureRidge().fit(p_train,y_train[:,j],groups,alphas=(1e-6,1e-4,1e-2,1.0),folds=5)
        for j in range(len(lcols))]
results={'input_status':'real_RegMix_train_and_test_estimated_extrapolations_separate',
         'objective':'13 validation-domain cross-entropy losses; equal-domain mean reported only for summary',
         'selected_model':'17-component recipe ridge with 16 explanatory columns + intercept',
         'selection':'5-fold group CV within 1m training recipes only',
         'scaling':'No scale correction learned from any test or estimated label',
         'dataset_specs':{},'per_set':{},'per_domain':{},
         'train_cv':{lcols[j]:{'alpha':m.selected_alpha,'mae':min(m.history,key=lambda r:r['internal_group_cv_mae'])['internal_group_cv_mae']} for j,m in enumerate(models)}}
for name in SETS:
 p,y,idx,cols,sums,status=get(name)
 if cols!=lcols:raise ValueError('loss column order changed')
 pred=np.column_stack([m.predict(p) for m in models])
 baseline=np.tile(y_train.mean(axis=0),(len(y),1))
 per=[]
 for j,col in enumerate(lcols):
  rho,pval=spearmanr(y[:,j],pred[:,j])
  per.append({'domain_loss':col,'ridge_mae':regression_metrics(y[:,j],pred[:,j])['mae'],
              'ridge_rmse':regression_metrics(y[:,j],pred[:,j])['rmse'],
              'train_mean_baseline_mae':regression_metrics(y[:,j],baseline[:,j])['mae'],
              'spearman':float(rho) if np.isfinite(rho) else None,
              'reference_mean':float(y[:,j].mean()),'predicted_mean':float(pred[:,j].mean()),
              'alpha':models[j].selected_alpha})
 results['dataset_specs'][name]={'status':status,'n':len(p),'mixture_sum':sums}
 results['per_domain'][name]=per
 results['per_set'][name]={
  'n':len(p),'status':status,'mae_mean_over_13':float(np.mean([r['ridge_mae'] for r in per])),
  'baseline_mae_mean_over_13':float(np.mean([r['train_mean_baseline_mae'] for r in per])),
  'spearman_mean_over_13':float(np.mean([r['spearman'] for r in per if r['spearman'] is not None])),
  'reference_loss_mean_over_13':float(y.mean()),'predicted_loss_mean_over_13':float(pred.mean())}
print(json.dumps({'train_cv_mean_mae':float(np.mean([m['mae'] for m in results['train_cv'].values()])),
                  'sets':results['per_set']},ensure_ascii=False,indent=2))
target=ROOT/'05_results'/'tables'/'q1_mixture_baseline_results.json'
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
