"""Q2 supplement: lambda_p(N), elasticities, per-FLOP marginal gains,
quality-parameter substitution condition and domain substitution/complementarity.

Generalized law (Q2):
    L = E + a (N/Nr)^-alpha + b (D/Dr)^-beta + h(N,D,Q) + lambda_p(N) [f(p)-f(p_ref)]
    h = c (N/Nr)^-theta_N (D/Dr)^-theta_D (1-Q)^kappa   (scale-dependent quality term, B6)
Because h -> 0 as N -> inf, adding parameters can always offset a quality gap
eventually; the substitution condition is therefore an equation for N' plus a
cost comparison, not an upper bound N_max as in the scale-free form.
lambda_p(N) is fitted to the Q1 cross-scale robust slopes b_s (1M, 60M, 1B, all real data)
with the half-effect form lambda_p(N) = 1 / (1 + (N/N_h)^rho).
"""

from pathlib import Path
import json
import math
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_src'))
from fmodel.costs import ETA, quality_cost
from fmodel.scaling import quality_penalty, quality_penalty_grads

NR=DR=1e9
KINDS=['exponential','power','logarithmic']


def g_prime(Q,kind):
    if kind=='exponential':return 1e7*6*math.exp(6*Q)
    if kind=='power':return 5e9*4*Q**3
    return 2e9*10/(1+10*Q)


def fit_lambda(scales):
    N=np.array([s['N'] for s in scales]);lam=np.array([s['lambda'] for s in scales])
    def resid(x):
        logNh,rho=x
        return np.log(1/(1+(N/10**logNh)**rho))-np.log(lam)
    fit=least_squares(resid,[8.8,.9],bounds=([5,.05],[13,5]))
    logNh,rho=fit.x
    return {'N_h':float(10**logNh),'rho':float(rho),'max_abs_log_residual':float(np.max(abs(fit.fun))),
            'form':'lambda_p(N)=1/(1+(N/N_h)^rho)',
            'fitted':[{'N':float(n),'observed':float(l),'fitted':float(1/(1+(n/10**logNh)**rho))}
                      for n,l in zip(N,lam)],
            'extrapolation':{f'{n:.0e}':float(1/(1+(n/10**logNh)**rho)) for n in [1e10,7e10,1e11]}}


def main():
    law=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    E,a,b,alpha,beta=law['params']
    qj=json.loads((ROOT/'05_results/models/q2_quality_scenario_B6.json').read_text(encoding='utf-8'))
    c,kappa,thN,thD=qj['c'],qj['kappa'],qj['theta_N'],qj['theta_D']
    scores=pd.read_parquet(ROOT/'02_data/processed/q1_quality_scores_exploratory.parquet')
    Q0=float(scores.loc[scores.source=='A1','Q_equal'].mean())
    cross=json.loads((ROOT/'05_results/tables/q1_cross_scale_rank.json').read_text(encoding='utf-8'))
    q3=json.loads((ROOT/'05_results/tables/q3_structural_transition.json').read_text(encoding='utf-8'))
    k=6+ETA*q3['baseline_context']

    def loss(N,D,Q):return float(E+a*(N/NR)**-alpha+b*(D/DR)**-beta+quality_penalty(N,D,Q,qj))

    # 1. lambda_p(N) from real-scale slopes
    scales=[]
    for key in ['test_1m','test_60m','test_1B']:
        s=cross['per_set'][key]
        slopes=[d['slope_b_full'] for d in s['per_domain']]
        scales.append({'set':key,'N':s['N_params'],'lambda':float(np.median(slopes)),
                       'domain_p25_p75':list(map(float,np.quantile(slopes,[.25,.75])))})
    lam=fit_lambda(scales)
    lam['points']=scales

    # 2. elasticities and 3. per-FLOP marginal Loss reduction
    points=[]
    for NB in [.1,1,10,100]:
        points.append({'label':f'N={NB}B, D=20N, Q=Q0','N':NB*1e9,'D':20*NB*1e9,'Q':Q0})
    for r in q3['main_allocation']:
        if r['cost_function']=='power':
            Q=min(r['Q'],Q0) if r['regime'].startswith('R1') else r['Q']
            points.append({'label':f"Q3 optimum C={r['budget_FLOPs']:.0e} (power)",
                           'N':r['N_B']*1e9,'D':r['D_B']*1e9,'Q':Q})
    elastic=[]
    for p in points:
        N,D,Q=p['N'],p['D'],p['Q']
        L=loss(N,D,Q)
        hN,hD,hQ=(float(v) for v in quality_penalty_grads(N,D,Q,qj))
        gN=alpha*a*(N/NR)**-alpha-hN   # -N dL/dN
        gD=beta*b*(D/DR)**-beta-hD     # -D dL/dD
        row={**p,'loss':L,'quality_penalty':float(quality_penalty(N,D,Q,qj)),
             'eps_N':-gN/L,'eps_D':-gD/L,
             'eps_N_share_from_quality_term':-hN/gN,
             'eps_Q':(hQ*Q/L) if Q<1-1e-9 else None,
             'dL_per_FLOP_via_N':gN/N/(k*D),'dL_per_FLOP_via_D':gD/D/(k*N)}
        if Q<1-1e-9:
            for kind in KINDS:
                row[f'dL_per_FLOP_via_Q_{kind}']=-hQ/(D*g_prime(Q,kind))
        elastic.append(row)

    # 4. quality +0.1 <-> parameters substitution condition (D fixed at 20N of the starting model)
    dQ=.1
    def N_equiv(N,D):
        target=loss(N,D,Q0+dQ)
        f=lambda logM:loss(math.exp(logM),D,Q0)-target
        return math.exp(brentq(f,math.log(N),math.log(N)+80,xtol=1e-12))
    equiv=[]
    for NB in [.1,.5,1,5,10,50,100,300]:
        N=NB*1e9;D=20*N
        Ne=N_equiv(N,D)
        equiv.append({'N_B':NB,'D_B':D/1e9,'loss_gain_from_quality':loss(N,D,Q0)-loss(N,D,Q0+dQ),
                      'equivalent_N_B':Ne/1e9,'multiplier':Ne/N})
    # scale-free comparison (theta=0) to show what changed
    sf={'c':qj['model_comparison_B6']['M0_scale_free']['params']['c'],
        'kappa':qj['model_comparison_B6']['M0_scale_free']['params']['kappa']}
    sf_gain=float(quality_penalty(1e9,1e9,Q0,sf)-quality_penalty(1e9,1e9,Q0+dQ,sf))
    sf_Nmax=NR*(a/sf_gain)**(1/alpha)
    # cost comparison: quality costs D*dg, extra params cost k*(N'-N)*D -> compare dg with k(N'-N)
    crossover={}
    for kind in KINDS:
        dg=quality_cost(Q0+dQ,kind)-quality_cost(Q0,kind)
        f=lambda logN:k*(N_equiv(math.exp(logN),20*math.exp(logN))-math.exp(logN))-dg
        try:crossover[kind]={'dg_FLOPs_per_token':dg,
                              'N_crossover_B':math.exp(brentq(f,math.log(1e6),math.log(1e13)))/1e9}
        except ValueError:crossover[kind]={'dg_FLOPs_per_token':dg,'N_crossover_B':None}
    subst={'delta_Q':dQ,'Q0':Q0,'D_rule':'D=20N of the starting model, held fixed while N grows',
           'equation':"solve L(N',D,Q0)=L(N,D,Q0+dQ) for N' (brentq on log N')",
           'existence_condition':('always solvable: with theta_N>0 the quality term vanishes as N->inf, so '
                                  'L(N->inf,D,Q0)=E+b(D/Dr)^-beta < L(N,D,Q0+dQ) for every finite N'),
           'table':equiv,
           'scale_free_comparison':{'loss_gain':sf_gain,'N_max_B':sf_Nmax/1e9,
                                    'note':'old theta=0 form implied N_max; rejected by LR test on B6'},
           'cost_crossover':{'rule':'quality is cheaper iff g(Q0+0.1)-g(Q0) < k (N_equiv-N), D=20N',
                             'context':q3['baseline_context'],'by_cost_function':crossover}}

    # 5. domain substitution / complementarity at p_ref (Q1 model, 13 validation targets)
    saved=joblib.load(ROOT/'05_results/models/q1_square_mixture_model.joblib')
    def f(P):
        t=saved['polynomial'].transform(P[:,:-1])[:,saved['square_indices']]
        return saved['ridge'].predict(saved['scaler'].transform(t))
    p=np.asarray(saved['p_reference'],float)
    names=[x.removeprefix('train_the_pile_') for x in saved['mixture_columns']]
    targets=[x.removeprefix('metric/the_pile_').removesuffix('_val_loss') for x in saved['loss_columns']]
    h=.005
    def move(s):  # s: dict domain_index -> step along e_j - p
        q=p.copy()
        for j,st in s.items():q=q+st*(np.eye(17)[j]-p)
        return q
    base=f(p[None,:])[0]
    J=np.array([(f(move({j:h})[None,:])[0]-base)/h for j in range(17)])  # 17 x 13
    cos=np.array([[J[i]@J[j]/np.linalg.norm(J[i])/np.linalg.norm(J[j]) for j in range(17)] for i in range(17)])
    mean=lambda q:f(q[None,:])[0].mean()
    H=np.zeros((17,17))
    for i in range(17):
        for j in range(i+1,17):
            H[i,j]=H[j,i]=(mean(move({i:h,j:h}))-mean(move({i:h}))-mean(move({j:h}))+mean(p))/h**2
    pairs=[{'i':names[i],'j':names[j],'profile_cosine':float(cos[i,j]),'cross_partial_mean_loss':float(H[i,j])}
           for i in range(17) for j in range(i+1,17)]
    by_cos=sorted(pairs,key=lambda r:-r['profile_cosine'])
    by_H=sorted(pairs,key=lambda r:r['cross_partial_mean_loss'])
    domain={'point':'Q1 reference mixture p_ref','step':h,
            'definitions':{'profile_cosine':'cosine of the 13-target effect vectors of moving toward domain i vs j; '
                                            '>0 similar targets (substitutes in function), <0 opposite targets (complements)',
                           'cross_partial':'d2 mean-Loss / ds_i ds_j along simplex directions; '
                                           '<0 joint increase helps more than sum (Edgeworth complements), >0 substitutes'},
            'jacobian_17x13':{'domains':names,'targets':targets,'values':J.tolist()},
            'most_similar_profiles':by_cos[:8],'most_opposite_profiles':by_cos[-8:],
            'strongest_complements_cross_partial':by_H[:8],'strongest_substitutes_cross_partial':by_H[-8:]}

    report={'parameters':{'E':E,'a':a,'b':b,'alpha':alpha,'beta':beta,'c':c,'kappa':kappa,
                          'theta_N':thN,'theta_D':thD,'Q0':Q0,
                          'context_for_costs':q3['baseline_context']},
            'lambda_p':lam,'elasticities_and_marginals':elastic,'quality_parameter_substitution':subst,
            'domain_substitution_complementarity':domain}
    out=ROOT/'05_results/tables/q2_elasticity_substitution.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    print('lambda_p:',{k:lam[k] for k in ['N_h','rho','max_abs_log_residual']},lam['extrapolation'])
    for r in elastic:
        print(r['label'],'L=%.3f eN=%.4f eD=%.4f eQ=%s'%(r['loss'],r['eps_N'],r['eps_D'],
              'NA' if r['eps_Q'] is None else '%.4f'%r['eps_Q']),
              'perFLOP N=%.2e D=%.2e'%(r['dL_per_FLOP_via_N'],r['dL_per_FLOP_via_D']),
              ' '.join('Q_%s=%.2e'%(kk[:3],r.get(f'dL_per_FLOP_via_Q_{kk}',float('nan'))) for kk in KINDS))
    print('scale-free comparison',subst['scale_free_comparison'])
    for e in equiv:print(e)
    print(crossover)
    print('similar',[(r['i'],r['j'],round(r['profile_cosine'],2)) for r in by_cos[:6]])
    print('opposite',[(r['i'],r['j'],round(r['profile_cosine'],2)) for r in by_cos[-6:]])
    print('complements',[(r['i'],r['j'],round(r['cross_partial_mean_loss'],2)) for r in by_H[:6]])
    print('substitutes',[(r['i'],r['j'],round(r['cross_partial_mean_loss'],2)) for r in by_H[-6:]])


if __name__=='__main__':main()
