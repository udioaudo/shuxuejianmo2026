"""Fit the Q2 quality term on B6 (semi-synthetic); do not call it real evidence.

Nested additive family on top of the fixed B1 classical law L_cl(N,D):
    M0  h = c (1-Q)^kappa                                   (scale-free, old H4)
    M1  h = c (N/1e9)^-theta_N (1-Q)^kappa
    M2  h = c (N/1e9)^-theta_N (D/1e9)^-theta_D (1-Q)^kappa
Multiplicative forms L_cl * [1 + c s(N) (1-Q)^kappa] are fitted for comparison only.
All forms vanish at Q=1, so the law reduces to the classical B1 form.
The main form is the nested model with the lowest BIC on B6; nested pairs are
compared with Gaussian likelihood-ratio tests. Parameter intervals come from a
bootstrap over the 45 (N,D) design cells of B6 (Q levels stay together).

B7 rows not in B6 are a nested extension (same generator), not independent data.
B8 rows with Q<1 are diagnosed rather than scored: their Loss rises with Q and
falls below the classical irreducible loss E, which no quality term can reproduce.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_src'))
from fmodel.scaling import classical_loss
from fmodel.mixture import regression_metrics

DATA=ROOT/'02_data/raw/real_attachments/B_scaling_laws'
NREF=DREF=1e9
BOOT=500


def load(name):
    x=pd.read_csv(DATA/name)
    cols=['N_params_B','D_tokens_B','Q_score','val_loss','experiment_id']
    if not set(cols).issubset(x.columns) or x[cols].isna().any().any():
        raise ValueError('missing supplementary fields')
    return x


def arrays(x):
    return (x.N_params_B.to_numpy(float)*1e9,x.D_tokens_B.to_numpy(float)*1e9,
            x.Q_score.to_numpy(float),x.val_loss.to_numpy(float))


# name -> (parameter names, start, lower, upper, additive?, nested order)
FORMS={
    'M0_scale_free':(['c','kappa'],[.35,1],[0,.1],[3,3],True),
    'M1_N_decay':(['c','kappa','theta_N'],[.35,1,.1],[0,.1,-1],[3,3,2],True),
    'M2_N_D_decay':(['c','kappa','theta_N','theta_D'],[.35,1,.1,.02],[0,.1,-1,-1],[3,3,2,2],True),
    'X1_multiplicative':(['c','kappa'],[.15,1],[0,.1],[3,3],False),
    'X2_multiplicative_N_decay':(['c','kappa','theta_N'],[.15,1,.1],[0,.1,-1],[3,3,2],False),
}
NESTED=['M0_scale_free','M1_N_decay','M2_N_D_decay']


def as_dict(names,values):
    d={'theta_N':0.0,'theta_D':0.0}
    d.update(dict(zip(names,map(float,values))))
    return d


def predict(form,p,base,N,D,Q):
    s=(N/NREF)**-p['theta_N']*(D/DREF)**-p['theta_D']*np.maximum(1-Q,0)**p['kappa']
    return base+p['c']*s if FORMS[form][4] else base*(1+p['c']*s)


def fit(form,base,N,D,Q,y):
    names,x0,lo,hi,_=FORMS[form]
    r=least_squares(lambda v:predict(form,as_dict(names,v),base,N,D,Q)-y,x0=x0,bounds=(lo,hi),
                    x_scale='jac',ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=5000)
    if not r.success:raise RuntimeError(f'{form} fit failed')
    return as_dict(names,r.x),float(np.sum(r.fun**2))


def within_cell_slopes(x):
    """OLS slope of Loss on Q inside each (N,D) cell: model-free evidence on scale dependence."""
    rows=[]
    for (n,d),g in x.groupby(['N_params_B','D_tokens_B']):
        if g.Q_score.nunique()>=3:
            rows.append({'N_B':float(n),'D_B':float(d),'slope':float(np.polyfit(g.Q_score,g.val_loss,1)[0])})
    t=pd.DataFrame(rows)
    return t,[{'N_B':float(n),'mean_slope':float(v),'cells':int(k)}
              for n,v,k in zip(*[t.groupby('N_B').slope.mean().index,
                                 t.groupby('N_B').slope.mean().values,
                                 t.groupby('N_B').size().values])]


def main():
    classical=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    law=classical['params'];E=law[0]
    train=load('supplementary_NQ_experiment.csv')
    new=load('supplementary_NQ_experiment_expanded.csv')
    outside=load('supplementary_NQ_experiment_large.csv')
    old_ids=set(train.experiment_id)
    new_only=new.loc[~new.experiment_id.isin(old_ids)].copy()
    if len(new_only)!=90:
        raise ValueError('B7 nesting changed; inspect before using it as a check')
    base=lambda N,D:classical_loss(N,D,law,NREF,DREF)

    N,D,Q,y=arrays(train);b0=base(N,D);n=len(y)
    fits={}
    for form,(names,*_rest) in FORMS.items():
        p,rss=fit(form,b0,N,D,Q,y)
        k=len(names)
        fits[form]={'params':p,'k':k,'rss':rss,
                    'aic':float(n*np.log(rss/n)+2*k),'bic':float(n*np.log(rss/n)+k*np.log(n)),
                    'additive':FORMS[form][4]}
        for label,x in [('B6',train),('B7_new_only',new_only)]:
            Nx,Dx,Qx,yx=arrays(x)
            fits[form][label]=regression_metrics(yx,predict(form,p,base(Nx,Dx),Nx,Dx,Qx))
    lr=[]
    for small,big in zip(NESTED[:-1],NESTED[1:]):
        stat=n*np.log(fits[small]['rss']/fits[big]['rss'])
        df=fits[big]['k']-fits[small]['k']
        lr.append({'restricted':small,'full':big,'LR':float(stat),'df':df,
                   'p_value':float(chi2.sf(stat,df))})
    lr.append({'restricted':NESTED[0],'full':NESTED[-1],
               'LR':float(n*np.log(fits[NESTED[0]]['rss']/fits[NESTED[-1]]['rss'])),'df':2,
               'p_value':float(chi2.sf(n*np.log(fits[NESTED[0]]['rss']/fits[NESTED[-1]]['rss']),2))})
    main_form=min(NESTED,key=lambda f:fits[f]['bic'])
    best=fits[main_form]['params']

    # cluster bootstrap over design cells
    cells=train.groupby(['N_params_B','D_tokens_B']).indices
    keys=list(cells)
    rng=np.random.default_rng(2026)
    draws=[]
    for _ in range(BOOT):
        pick=np.concatenate([cells[keys[i]] for i in rng.integers(0,len(keys),len(keys))])
        p,_=fit(main_form,b0[pick],N[pick],D[pick],Q[pick],y[pick])
        draws.append([p['c'],p['kappa'],p['theta_N'],p['theta_D']])
    draws=np.array(draws)
    ci={name:[float(v) for v in np.quantile(draws[:,j],[.025,.975])]
        for j,name in enumerate(['c','kappa','theta_N','theta_D'])}

    slope_cells,slope_by_N=within_cell_slopes(train)
    # model-implied slope for comparison: -d h/d Q at the cell's (N,D), averaged over Q grid
    for form in NESTED:
        p=fits[form]['params']
        implied=[]
        for r in slope_cells.itertuples():
            q=np.linspace(.1,1,10)
            implied.append(np.polyfit(q,p['c']*(r.N_B)**-p['theta_N']*(r.D_B)**-p['theta_D']*(1-q)**p['kappa'],1)[0])
        fits[form]['cell_slope_MAE']=float(np.mean(np.abs(np.array(implied)-slope_cells.slope.to_numpy())))

    # B8 diagnosis
    N8,D8,Q8,y8=arrays(outside)
    top=Q8>=1-1e-9
    b8_top=regression_metrics(y8[top],base(N8[top],D8[top]))
    _,b8_slopes=within_cell_slopes(outside)
    b8={'status':'Q<1 rows unusable for the quality term',
        'rows':int(len(y8)),'rows_Q_eq_1':int(top.sum()),
        'Q_eq_1_vs_classical':b8_top,
        'Q_lt_1_rows_with_loss_below_E':int((y8[~top]<E).sum()),
        'Q_lt_1_rows':int((~top).sum()),
        'min_loss_Q_lt_1':float(y8[~top].min()),
        'mean_within_cell_slope_dL_dQ':float(np.mean([r['mean_slope'] for r in b8_slopes])),
        'reason':('B8 Loss increases with Q (positive within-cell slope) and many Q<1 rows fall below the '
                  'irreducible loss E of B1, e.g. floor 0.5; this contradicts B6/B7 and the problem definition, '
                  'so only the Q=1 rows are used (as a classical-law check).')}

    document={'formula':'L = L_classical_B1(N,D) + c (N/1e9)^-theta_N (D/1e9)^-theta_D (1-Q)^kappa',
              'main_form':main_form,
              'c':best['c'],'kappa':best['kappa'],'gamma':best['kappa'],
              'theta_N':best['theta_N'],'theta_D':best['theta_D'],'Nref':NREF,'Dref':DREF,
              'bootstrap_95ci_cells':ci,'bootstrap_draws':BOOT,
              'reference_Q_at_classic_baseline':1,
              'input_model':'05_results/models/q2_classical_B1.json',
              'model_comparison_B6':fits,
              'likelihood_ratio_tests':lr,
              'selection_rule':'lowest BIC among nested additive forms M0-M2; multiplicative forms reported only',
              'within_cell_slopes_B6':slope_by_N,
              'B8_diagnosis':b8,
              'data_evidence':'B6/B7 semi-synthetic calibration only',
              'not_identified_from_real_quality_experiment':True,
              'A1_quality_scale_not_proven_comparable_to_B6_Q_score':True,
              'B7_B6_id_overlap':int(len(set(new.experiment_id)&old_ids)),
              'validation':{'B6':fits[main_form]['B6'],'B7_new_only':fits[main_form]['B7_new_only'],
                            'B8_Q_eq_1_classical':b8_top}}
    path=ROOT/'05_results/models/q2_quality_scenario_B6.json'
    path.write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    print('main form',main_form,{k:round(v,4) for k,v in best.items()},'CI',{k:[round(a,4) for a in v] for k,v in ci.items()})
    for f,r in fits.items():
        print(f"{f:28s} k={r['k']} BIC={r['bic']:.1f} B6 MAE={r['B6']['mae']:.4f} B7 MAE={r['B7_new_only']['mae']:.4f}"
              f" cellslope MAE={r.get('cell_slope_MAE',float('nan')):.4f}",{k:round(v,4) for k,v in r['params'].items()})
    for t in lr:print('LR',t['restricted'],'->',t['full'],'LR=%.1f df=%d p=%.2e'%(t['LR'],t['df'],t['p_value']))
    print('within-cell slopes',[(r['N_B'],round(r['mean_slope'],3)) for r in slope_by_N])
    print('B8',{k:v for k,v in b8.items() if k!='reason'})


if __name__=='__main__':main()
