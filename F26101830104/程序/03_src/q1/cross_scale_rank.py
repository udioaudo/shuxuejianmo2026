"""Cross-scale robustness of the saved 1M square mixture model.

RegMix-style check: the 1M model is judged on whether it ranks recipes
correctly at 60M/1B (Spearman, top-decile hit rate), not on absolute Loss.
An affine per-scale calibration L_s = a_s + b_s * f_1M(p), fitted on a few
recipes of each scale, separates the scale offset (handled by Q2) from the
mixture effect; b_s measures how the mixture effect scales with N.
Also reports simplex-respecting directional effects of each training domain.
"""

from pathlib import Path
import json
import sys
import joblib
import numpy as np
from scipy.stats import spearmanr, kendalltau, theilslopes

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_src/q1'))
from mixture_interactions import read_pair

SCALES={'test_1m':1e6,'test_60m':6e7,'test_1B':1e9,'est_10b':1e10,'est_70b':7e10}
CALIBRATION_RECIPES=16
REPEATS=500
SEED=20260924


def predictor(saved):
    def f(p):
        terms=saved['polynomial'].transform(p[:,:-1])[:,saved['square_indices']]
        return saved['ridge'].predict(saved['scaler'].transform(terms))
    return f


def top_hit(true,pred,share=.1):
    k=max(1,int(round(share*len(true))))
    return len(set(np.argsort(true)[:k])&set(np.argsort(pred)[:k]))/k


def affine_calibration(pred,true,rng):
    """Fit a_s,b_s on CALIBRATION_RECIPES random recipes, score the rest."""
    n=len(true)
    maes,slopes=[],[]
    for _ in range(REPEATS):
        cal=rng.choice(n,CALIBRATION_RECIPES,replace=False)
        rest=np.setdiff1d(np.arange(n),cal)
        b,a=np.polyfit(pred[cal],true[cal],1)
        maes.append(np.mean(abs(true[rest]-(a+b*pred[rest]))))
        slopes.append(b)
    # Theil-Sen: robust to recipes outside the 1M training support.
    b_all,a_all=theilslopes(true,pred)[:2]
    return {'holdout_mae_mean':float(np.mean(maes)),
            'holdout_mae_p2.5_p97.5':list(map(float,np.quantile(maes,[.025,.975]))),
            'slope_b_full':float(b_all),'intercept_a_full':float(a_all),
            'slope_b_p2.5_p97.5':list(map(float,np.quantile(slopes,[.025,.975])))}


def domain_effects(f,p_ref,names,sd):
    """Mean predicted Loss change when domain j's share rises by one training sd,
    taken proportionally from the other domains (stays on the simplex and in support)."""
    base=f(p_ref[None,:])
    rows=[]
    for j,name in enumerate(names):
        target=np.zeros_like(p_ref);target[j]=1
        step=sd[j]/(1-p_ref[j])
        moved=(1-step)*p_ref+step*target
        delta=f(moved[None,:])-base
        rows.append({'domain':name,'p_ref':float(p_ref[j]),'share_increase':float(sd[j]),
                     'mean_delta_loss_13':float(delta.mean()),
                     'per_validation_domain':delta.ravel().tolist()})
    return sorted(rows,key=lambda r:r['mean_delta_loss_13'])


def main():
    saved=joblib.load(ROOT/'05_results/models/q1_square_mixture_model.joblib')
    f=predictor(saved)
    targets=[c.removeprefix('metric/the_pile_').removesuffix('_val_loss') for c in saved['loss_columns']]
    rng=np.random.default_rng(SEED)
    result={'model':'Q1 saved square-only ridge, trained on 1M A4/A5 only',
            'metrics':'per validation domain, then averaged over 13 domains',
            'calibration':f'affine a_s+b_s*f_1M(p) on {CALIBRATION_RECIPES} random recipes per scale, '
                          f'{REPEATS} repeats, scored on remaining recipes',
            'per_set':{}}
    ptrain=read_pair('train')[0]
    support=ptrain.max(0)
    for name,N in SCALES.items():
        p,y,_,cols,tcols=read_pair(name)
        inside=(p<=support+1e-9).all(1)
        if cols!=saved['mixture_columns'] or tcols!=saved['loss_columns']:
            raise ValueError('changed domain order')
        pred=f(p)
        rho=[spearmanr(pred[:,j],y[:,j]).statistic for j in range(13)]
        tau=[kendalltau(pred[:,j],y[:,j]).statistic for j in range(13)]
        hit=[top_hit(y[:,j],pred[:,j]) for j in range(13)]
        avg_true,avg_pred=y.mean(1),pred.mean(1)
        cal=[affine_calibration(pred[:,j],y[:,j],rng) for j in range(13)]
        result['per_set'][name]={
            'N_params':N,'n_recipes':len(p),
            'data_status':'estimated_not_observed' if name.startswith('est') else 'real',
            'raw_mae_mean_13':float(np.mean(abs(y-pred))),
            'recipes_inside_1M_training_support':int(inside.sum()),
            'spearman_mean_13_inside_support':float(np.mean([spearmanr(pred[inside,j],y[inside,j]).statistic
                                                             for j in range(13)])) if inside.sum()>5 else None,
            'spearman_mean_13':float(np.mean(rho)),'kendall_mean_13':float(np.mean(tau)),
            'spearman_min_13':float(np.min(rho)),
            'top10pct_hit_rate_mean_13':float(np.mean(hit)),
            'spearman_equal_domain_average':float(spearmanr(avg_pred,avg_true).statistic),
            'calibrated_mae_mean_13':float(np.mean([c['holdout_mae_mean'] for c in cal])),
            'slope_b_median_13':float(np.median([c['slope_b_full'] for c in cal])),
            'per_domain':[{'domain':t,'spearman':float(r),'top10_hit':float(h),**c}
                          for t,r,h,c in zip(targets,rho,hit,cal)]}
    # 1M and 60M test tables share the same 256 recipes: observed rank agreement is the ceiling.
    p1,y1,i1,_,_=read_pair('test_1m')
    p60,y60,i60,_,_=read_pair('test_60m')
    if not (np.array_equal(i1,i60) and np.allclose(p1,p60)):
        raise ValueError('1M/60M test recipes differ')
    result['observed_1m_vs_60m_same_recipes']={
        'spearman_mean_13':float(np.mean([spearmanr(y1[:,j],y60[:,j]).statistic for j in range(13)])),
        'note':'rank agreement between true 1M and true 60M Loss on identical recipes'}
    p_ref=np.asarray(saved['p_reference'],float)
    names=[c.removeprefix('train_the_pile_') for c in saved['mixture_columns']]
    result['domain_directional_effects']={
        'definition':'share of domain j raised by its training sd, others scaled down proportionally; '
                     'negative = Loss decreases',
        'p_ref':'mean of best 10% training recipes (Q1 saved)',
        'targets':targets,'rows':domain_effects(f,p_ref,names,ptrain.std(0))}
    path=ROOT/'05_results/tables/q1_cross_scale_rank.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{m:v[m] for m in ['n_recipes','recipes_inside_1M_training_support','spearman_mean_13_inside_support','raw_mae_mean_13','spearman_mean_13',
                                          'top10pct_hit_rate_mean_13','calibrated_mae_mean_13',
                                          'slope_b_median_13']}
                      for k,v in result['per_set'].items()},ensure_ascii=False,indent=1))
    print('observed 1M vs 60M spearman',result['observed_1m_vs_60m_same_recipes']['spearman_mean_13'])
    print([(r['domain'],round(r['mean_delta_loss_13'],4)) for r in result['domain_directional_effects']['rows']])


if __name__=='__main__':main()
