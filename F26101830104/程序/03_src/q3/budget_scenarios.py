"""Bounded N,D,Q scenarios; quality effect comes from semi-synthetic B6.

These conditional scenarios are not verified real optimal allocations.
"""

from pathlib import Path
import json
import sys
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_src'))
from fmodel.scaling import classical_loss, quality_penalty
from fmodel.costs import compute_costs, critical_context_length


def reference_mixture():
    saved=joblib.load(ROOT/'05_results/models/q1_square_mixture_model.joblib')
    domain=saved['mixture_columns']
    p=np.asarray(saved['p_reference'],float)
    if len(domain)!=17 or len(p)!=17 or (p<0).any() or abs(p.sum()-1)>1e-8:
        raise ValueError('invalid Q1 reference mixture')
    return {'basis':'mean of top 10% training recipes by equal-domain validation Loss',
            'source':'Q1 saved model output from A4/A5 only',
            'domains':[c.removeprefix('train_the_pile_') for c in domain],
            'weights':p.tolist(),'status':'fixed reference mixture, not cross-scale optimal'}


def optimize_one(C, context, kind, Q0, scaling, quality):
    nlo,nhi=scaling['training_ranges']['N_B']
    dlo,dhi=scaling['training_ranges']['D_B']
    E,a,b,alpha,beta=scaling['params']
    bounds=[(math.log(nlo),math.log(nhi)),(math.log(dlo),math.log(dhi)),(Q0,1)]

    def unpack(x):
        return math.exp(x[0])*1e9, math.exp(x[1])*1e9, x[2]

    def loss(x):
        N,D,Q=unpack(x)
        return float(classical_loss(N,D,scaling['params'],1e9,1e9)+quality_penalty(N,D,Q,quality))

    def slack(x):
        N,D,Q=unpack(x)
        return 1-compute_costs(N,D,Q,Q0,context,kind)['total']/C

    starts=[]
    for fn,fd,fq in [(0,0,0),(.25,.25,.5),(.5,.5,0),(.5,.5,.5),
                     (.75,.25,.75),(.25,.75,.75),(.9,.9,1)]:
        starts.append([bounds[0][0]+fn*(bounds[0][1]-bounds[0][0]),
                       bounds[1][0]+fd*(bounds[1][1]-bounds[1][0]),
                       Q0+fq*(1-Q0)])
    valid=[]
    failed=0
    for start in starts:
        fit=minimize(loss,start,method='SLSQP',bounds=bounds,
                     constraints=[{'type':'ineq','fun':slack}],
                     options={'maxiter':500,'ftol':1e-10})
        if fit.success and np.isfinite(fit.fun) and slack(fit.x)>=-1e-6:
            valid.append(fit)
        else:
            failed+=1
    if not valid:
        raise RuntimeError(f'No validated feasible solution: C={C}, context={context}, kind={kind}')
    fit=min(valid,key=lambda r:r.fun)
    N,D,Q=unpack(fit.x)
    costs=compute_costs(N,D,Q,Q0,context,kind)
    return {'budget_FLOPs':C,'context_tokens':context,'quality_cost':kind,'Q0':Q0,
            'N_params':N,'D_tokens':D,'Q':Q,'predicted_loss_scenario':loss(fit.x),
            'costs_FLOPs':costs,'spent_fraction':costs['total']/C,
            'failed_starts':failed,'accepted_starts':len(valid),
            'N_at_observed_boundary':bool(np.isclose(N/1e9,nlo,rtol=1e-4) or
                                          np.isclose(N/1e9,nhi,rtol=1e-4)),
            'D_at_observed_boundary':bool(np.isclose(D/1e9,dlo,rtol=1e-4) or
                                          np.isclose(D/1e9,dhi,rtol=1e-4)),
            'Q_at_upper_bound':bool(Q>1-1e-5)}


def main():
    scaling=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    quality=json.loads((ROOT/'05_results/models/q2_quality_scenario_B6.json').read_text(encoding='utf-8'))
    scores=pd.read_parquet(ROOT/'02_data/processed/q1_quality_scores_exploratory.parquet')
    sampled=scores.loc[scores.source=='A1']
    q0s={'equal_22':float(sampled.Q_equal.mean()),
         'equal_3_groups':float(sampled.Q_grouped.mean())}
    lengths=pd.read_csv(ROOT/'02_data/raw/real_attachments/C_efficiency_evolution/model_architecture_metadata.csv')
    contexts=sorted(set(int(v) for v in lengths.max_position_embeddings.dropna()))
    if not contexts:
        raise ValueError('C7 has no lengths')
    p=reference_mixture()
    (ROOT/'05_results/models/q1_reference_mixture.json').write_text(
        json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for qlabel,Q0 in q0s.items():
        for C in [1e19,1e22,1e24]:
            for context in contexts:
                for kind in ['exponential','power','logarithmic']:
                    result=optimize_one(C,context,kind,Q0,scaling,quality)
                    result['quality_reference_variant']=qlabel
                    rows.append(result)
    report={'status':'conditional_semi_synthetic_quality_scenarios_not_real_optimal_training_claims',
            'model_components':['B1 fitted classical law','B6 semi-synthetic scale-dependent quality term',
                                'A1 alternative quality scales for Q0'],
            'assumption':'A1 quality scales are numerically treated as comparable to B6 Q; unverified',
            'p_policy':p,'p_effect_in_loss':'not identifiable across A/B; p fixed and not in the loss formula',
            'N_D_support':'bounded to B1 observed minimum and maximum; boundary is evidence limit',
            'context_values_source':'C7 unique max_position_embeddings',
            'critical_context_tokens':critical_context_length(),
            'solutions':rows}
    target=ROOT/'05_results/tables/q3_bounded_budget_scenarios.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    summary=[]
    for C in [1e19,1e22,1e24]:
        sample=[r for r in rows if r['budget_FLOPs']==C]
        summary.append({'budget':C,'n_cases':len(sample),
                        'median_spent_fraction':float(np.median([r['spent_fraction'] for r in sample])),
                        'N_bound_cases':sum(r['N_at_observed_boundary'] for r in sample),
                        'D_bound_cases':sum(r['D_at_observed_boundary'] for r in sample),
                        'Q_upper_cases':sum(r['Q_at_upper_bound'] for r in sample)})
    print(json.dumps({'case_count':len(rows),'C7_contexts':contexts,
                      'q0_alternatives':q0s,'budget_summary':summary},ensure_ascii=False))


if __name__=='__main__':main()
