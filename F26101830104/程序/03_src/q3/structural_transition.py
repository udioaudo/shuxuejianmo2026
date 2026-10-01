"""Q3 main results: budget allocation, KKT regimes and structural transition.

Model (Q2 output, p fixed at the Q1 reference mixture so the mixture term vanishes):
    L(N,D,Q) = E + a (N/1e9)^-alpha + b (D/1e9)^-beta + h(N,D,Q),
    h = c (N/1e9)^-theta_N (D/1e9)^-theta_D (1-Q)^kappa
    s.t. (6 + eta*Lctx) N D + D [g(Q) - g(Q0)]_+ <= C,  Q0 <= Q <= 1.

Structural transition = change of the KKT active set of the Q bounds:
    R1 {Q=Q0 active} -> R2 {Q interior} -> R3 {Q=1 active}.
R1->R2 threshold (start_threshold) solves the KKT system on the R1 path:
    alpha a x^-alpha + theta_N h = beta b y^-beta + theta_D h      (N-D balance)
    -dh/dQ * k N = (alpha a x^-alpha + theta_N h) g'(Q0)           (Q starts to pay)
With theta_N = theta_D = 0 this reduces to the closed form used before
(checked in closed_form_check). For the concave (log) cost the Q sub-problem is
bang-bang, so R1 jumps straight to R3 at C_J where the Q=1 and Q=Q0 value
functions cross (jump_threshold). Numerical solutions on a dense budget grid
(N,D widened to B9 metadata maxima; B1-supported range flagged) confirm both.
"""

from pathlib import Path
import json
import math
import sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_src'))
sys.path.insert(0,str(ROOT/'03_src/q3'))
from fmodel.costs import ETA, quality_cost, critical_context_length
from fmodel.scaling import quality_penalty, quality_penalty_grads
from budget_scenarios import optimize_one

KINDS=['exponential','power','logarithmic']
BUDGETS=[1e19,1e22,1e24]
GRID=np.logspace(17,25,65)
NR=DR=1e9


def g_prime(Q,kind):
    if kind=='exponential':return 1e7*6*math.exp(6*Q)
    if kind=='power':return 5e9*4*Q**3
    if kind=='logarithmic':return 2e9*10/(1+10*Q)
    raise ValueError(kind)


def classical_path_D(N,a,b,alpha,beta):
    """KKT ratio on the Q=Q0 path without a quality term: alpha a (N/NR)^-alpha = beta b (D/DR)^-beta."""
    return DR*((alpha*a)/(beta*b)*(N/NR)**(-alpha))**(-1/beta)


def closed_form_start(Q0,kind,context,law,c,kappa):
    """Old scale-free closed form (theta=0); kept as a regression check."""
    E,a,b,alpha,beta=law
    k=6+ETA*context
    x=(c*kappa*(1-Q0)**(kappa-1)*k*NR/(alpha*a*g_prime(Q0,kind)))**(-1/(1+alpha))
    N=NR*x
    D=classical_path_D(N,a,b,alpha,beta)
    return {'C':k*N*D,'N':N,'D':D}


def r1_path_D(N,Q0,law,quality):
    """D on the R1 path: N dL/dN = D dL/dD with Q=Q0 (budget-tight, no quality spend)."""
    E,a,b,alpha,beta=law
    def balance(logD):
        D=math.exp(logD)
        tN,tD,_=quality_penalty_grads(N,D,Q0,quality)
        return (alpha*a*(N/NR)**-alpha-float(tN))-(beta*b*(D/DR)**-beta-float(tD))
    return math.exp(brentq(balance,math.log(1e3),math.log(1e20),xtol=1e-12))


def start_threshold(Q0,kind,context,law,quality):
    """Budget where the per-FLOP Loss gain of Q at Q0 equals the shadow price mu on the R1 path."""
    E,a,b,alpha,beta=law
    k=6+ETA*context
    def excess(logN):
        N=math.exp(logN);D=r1_path_D(N,Q0,law,quality)
        tN,_,dQ=quality_penalty_grads(N,D,Q0,quality)
        return float(-dQ)*k*N-(alpha*a*(N/NR)**-alpha-float(tN))*g_prime(Q0,kind)
    N=math.exp(brentq(excess,math.log(1e5),math.log(1e14),xtol=1e-12))
    D=r1_path_D(N,Q0,law,quality)
    return {'C':k*N*D,'N':N,'D':D}


def fixed_Q_value(C,Q,Q0,kind,context,law,quality,bounds):
    """min over N of loss with Q fixed, budget tight: D = C/(kN + dg)."""
    E,a,b,alpha,beta=law
    k=6+ETA*context
    dg=max(quality_cost(Q,kind)-quality_cost(Q0,kind),0)
    (nlo,nhi),(dlo,dhi)=bounds
    # Budget-tight D must lie in [dlo,dhi]; N range that keeps it there.
    n_hi=min(nhi,(C/dlo-dg)/k)
    n_lo=max(nlo,(C/dhi-dg)/k)
    if n_hi<=n_lo:return math.inf
    def obj(logN):
        N=math.exp(logN);D=C/(k*N+dg)
        return a*(N/NR)**-alpha+b*(D/DR)**-beta+float(quality_penalty(N,D,Q,quality))
    fit=minimize_scalar(obj,bounds=(math.log(n_lo),math.log(n_hi)),method='bounded',
                        options={'xatol':1e-10})
    return E+fit.fun


def jump_threshold(Q0,kind,context,law,quality,bounds):
    diff=lambda lc:(fixed_Q_value(10**lc,1.0,Q0,kind,context,law,quality,bounds)-
                    fixed_Q_value(10**lc,Q0,Q0,kind,context,law,quality,bounds))
    grid=np.linspace(16,26,201)
    values=np.array([diff(x) for x in grid])
    # First budget where Q=1 becomes feasible and strictly better than Q=Q0.
    better=np.flatnonzero(np.isfinite(values)&(values<0))
    if len(better)==0 or better[0]==0:return None
    i=better[0]
    left=grid[i-1]
    if not np.isfinite(values[i-1]):
        # Q=1 infeasible just below: the jump is at the Q=1 feasibility edge.
        return float(10**brentq(lambda x:1.0 if not np.isfinite(diff(x)) else -1.0,left,grid[i],xtol=1e-6))
    return float(10**brentq(diff,left,grid[i],xtol=1e-6))


def regime(Q,Q0):
    if Q<Q0+1e-3:return 'R1_Q_at_baseline'
    if Q>1-1e-3:return 'R3_Q_at_upper_bound'
    return 'R2_Q_interior'


def compact(r,C):
    costs=r['costs_FLOPs']
    return {'budget_FLOPs':C,'context':r['context_tokens'],'cost_function':r['quality_cost'],
            'N_B':r['N_params']/1e9,'D_B':r['D_tokens']/1e9,'Q':r['Q'],
            'tokens_per_param':r['D_tokens']/r['N_params'],
            'loss':r['predicted_loss_scenario'],
            'share_train':costs['train']/C,'share_quality':costs['quality']/C,
            'share_attention':costs['attention']/C,'spent_fraction':r['spent_fraction'],
            'regime':regime(r['Q'],r['Q0']),
            'N_outside_B1':r['N_params']/1e9>r['_B1_N_max'],'D_outside_B1':r['D_tokens']/1e9>r['_B1_D_max']}


def main():
    law_json=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    qjson=json.loads((ROOT/'05_results/models/q2_quality_scenario_B6.json').read_text(encoding='utf-8'))
    law=law_json['params']
    scores=pd.read_parquet(ROOT/'02_data/processed/q1_quality_scores_exploratory.parquet')
    Q0=float(scores.loc[scores.source=='A1','Q_equal'].mean())
    c7=pd.read_csv(ROOT/'02_data/raw/real_attachments/C_efficiency_evolution/model_architecture_metadata.csv')
    ctx=c7.max_position_embeddings.dropna().astype(int)
    contexts=sorted(ctx.unique().tolist())
    base_context=int(np.median(ctx))
    b9=pd.read_csv(ROOT/'02_data/raw/real_attachments/B_scaling_laws/supplementary_large_models.csv')
    nlo,nhi_b1=law_json['training_ranges']['N_B'];dlo,dhi_b1=law_json['training_ranges']['D_B']
    wide=dict(law_json)
    wide['training_ranges']={'N_B':[nlo,float(b9.N_params_B.max())],
                             'D_B':[dlo,float(b9.loc[b9.D_tokens_B>0,'D_tokens_B'].max())]}
    bounds=((nlo*1e9,wide['training_ranges']['N_B'][1]*1e9),(dlo*1e9,wide['training_ranges']['D_B'][1]*1e9))

    def solve(C,context,kind):
        r=optimize_one(float(C),context,kind,Q0,wide,qjson)
        r['_B1_N_max'],r['_B1_D_max']=nhi_b1,dhi_b1
        return compact(r,float(C))

    main_table=[solve(C,base_context,k) for C in BUDGETS for k in KINDS]
    context_table=[solve(C,L,k) for C in BUDGETS for L in contexts for k in KINDS]

    transitions=[]
    for kind in KINDS:
        path=[solve(C,base_context,kind) for C in GRID]
        labels=[p['regime'] for p in path]
        switches=[{'from':labels[i-1],'to':labels[i],
                   'budget_bracket_FLOPs':[float(GRID[i-1]),float(GRID[i])]}
                  for i in range(1,len(labels)) if labels[i]!=labels[i-1]]
        qshare=[p['share_quality'] for p in path]
        peak=int(np.argmax(qshare))
        entry={'cost_function':kind,'context':base_context,
               'g_convexity':'concave' if kind=='logarithmic' else 'convex',
               'analytic_R1_exit_local_condition':start_threshold(Q0,kind,base_context,law,qjson),
               'numerical_regime_switches':switches,
               'quality_share_peak':{'budget_FLOPs':float(GRID[peak]),'share':qshare[peak]},
               'path':[{k:p[k] for k in ['budget_FLOPs','N_B','D_B','Q','tokens_per_param',
                                         'share_train','share_quality','share_attention','regime']}
                       for p in path]}
        if kind=='logarithmic':
            entry['analytic_jump_threshold_C_J']=jump_threshold(Q0,kind,base_context,law,qjson,bounds)
        transitions.append(entry)

    context_thresholds=[]
    for L in contexts:
        row={'context':L,'attention_to_train_ratio':ETA*L/6,
             'attention_share_when_Q_at_baseline':ETA*L/(6+ETA*L),
             'models_in_C7':int((ctx==L).sum())}
        for kind in KINDS:
            if kind=='logarithmic':
                row['C_J_logarithmic']=jump_threshold(Q0,kind,L,law,qjson,bounds)
            else:
                t=start_threshold(Q0,kind,L,law,qjson)
                row[f'C1_{kind}']=t['C'];row[f'N1_{kind}_B']=t['N']/1e9
        context_thresholds.append(row)

    # Regression check: the implicit KKT solver must reproduce the old closed form when theta=0.
    scale_free={'c':qjson['c'],'kappa':qjson['kappa']}
    check=[]
    for kind in ['exponential','power']:
        implicit=start_threshold(Q0,kind,base_context,law,scale_free)['C']
        closed=closed_form_start(Q0,kind,base_context,law,qjson['c'],qjson['kappa'])['C']
        check.append({'cost_function':kind,'implicit_C':implicit,'closed_form_C':closed,
                      'relative_difference':abs(implicit/closed-1)})
        if abs(implicit/closed-1)>1e-6:raise RuntimeError('implicit KKT solver disagrees with closed form')
    report={'model':('L=E+a(N/1e9)^-alpha+b(D/1e9)^-beta+c(N/1e9)^-theta_N(D/1e9)^-theta_D(1-Q)^kappa, '
                     'p fixed at Q1 reference mixture'),
            'parameters':{'E_a_b_alpha_beta':law,
                          'quality_term':{k:qjson[k] for k in ['c','kappa','theta_N','theta_D']},
                          'Q0_A1_equal_weight':Q0,'eta':ETA},
            'closed_form_check_theta_zero':check,
            'bounds_B':wide['training_ranges'],'B1_supported_max_B':{'N':nhi_b1,'D':dhi_b1},
            'baseline_context':base_context,'baseline_context_rule':'median max_position_embeddings in C7',
            'C7_contexts':contexts,'critical_context_tokens':critical_context_length(),
            'structural_transition_definition':(
                'Budget C* at which the KKT active set of {Q>=Q0, Q<=1} changes; '
                'identified analytically (R1 exit, log jump) and by regime labels on a 65-point log grid'),
            'main_allocation':main_table,'context_sensitivity':context_table,
            'transitions':transitions,'context_thresholds':context_thresholds}
    out=ROOT/'05_results/tables/q3_structural_transition.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    print('Q0',round(Q0,4),'base context',base_context)
    for r in main_table:
        print(f"{r['budget_FLOPs']:.0e} {r['cost_function']:<12} N={r['N_B']:.3f}B D={r['D_B']:.1f}B "
              f"Q={r['Q']:.3f} D/N={r['tokens_per_param']:.0f} L={r['loss']:.4f} "
              f"shares T/Q/A={r['share_train']:.3f}/{r['share_quality']:.3f}/{r['share_attention']:.3f} "
              f"{r['regime']} outsideB1={r['N_outside_B1'] or r['D_outside_B1']}")
    for t in transitions:
        print(t['cost_function'],'analytic C1=%.3e'%t['analytic_R1_exit_local_condition']['C'],
              'C_J=%s'%(('%.3e'%t['analytic_jump_threshold_C_J']) if t.get('analytic_jump_threshold_C_J') else '-'),
              [(s['from'][:2],s['to'][:2],'%.2e-%.2e'%tuple(s['budget_bracket_FLOPs'])) for s in t['numerical_regime_switches']],
              'peak share %.3f at %.1e'%(t['quality_share_peak']['share'],t['quality_share_peak']['budget_FLOPs']))
    for r in context_thresholds:
        print(r['context'],'attn/train=%.3f'%r['attention_to_train_ratio'],
              'C1 exp=%.2e pow=%.2e CJ log=%.2e'%(r['C1_exponential'],r['C1_power'],r['C_J_logarithmic'] or float('nan')))


if __name__=='__main__':main()
