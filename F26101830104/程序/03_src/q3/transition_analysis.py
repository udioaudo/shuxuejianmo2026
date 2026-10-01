"""Operational Q activation/saturation thresholds on a budget grid."""

from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_src/q3'))
from budget_scenarios import optimize_one

CONTEXT=32768


def bracket(budgets,values,threshold):
    yes=np.flatnonzero(np.asarray(values)>=threshold)
    if len(yes)==0:return {'status':'above_grid','lower_FLOPs':float(budgets[-1])}
    i=int(yes[0])
    if i==0:return {'status':'at_or_below_grid','upper_FLOPs':float(budgets[0])}
    return {'status':'bracketed','lower_FLOPs':float(budgets[i-1]),
            'upper_FLOPs':float(budgets[i])}


def main():
    scaling=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    quality=json.loads((ROOT/'05_results/models/q2_quality_scenario_B6.json').read_text(encoding='utf-8'))
    cases=json.loads((ROOT/'05_results/tables/q3_bounded_budget_scenarios.json').read_text(encoding='utf-8'))
    q0s=sorted({(x['quality_reference_variant'],x['Q0']) for x in cases['solutions']})
    budgets=np.logspace(18,23,41)
    rows=[]
    for label,Q0 in q0s:
        for kind in ['exponential','power','logarithmic']:
            values=[]
            bounds=[]
            for C in budgets:
                result=optimize_one(float(C),CONTEXT,kind,Q0,scaling,quality)
                values.append(result['Q'])
                bounds.append(bool(result['N_at_observed_boundary'] or result['D_at_observed_boundary']))
            rows.append({'Q0_variant':label,'Q0':Q0,'cost_function':kind,
                         'start_investing_in_Q':bracket(budgets,values,Q0+.01),
                         'Q_near_upper_bound':bracket(budgets,values,.999),
                         'first_scale_support_boundary':bracket(budgets,bounds,.5),
                         'nonmonotone_Q_jumps_gt_0p02':int((np.diff(values)<-.02).sum()),
                         'Q_by_budget':[{'budget_FLOPs':float(C),'Q':float(q)}
                                        for C,q in zip(budgets,values)]})
    report={'status':'model-defined constraint transition, not empirically proven technological phase shift',
            'context_tokens':CONTEXT,'budget_grid_size':len(budgets),
            'start_quality_rule':'first C with Q>=Q0+0.01',
            'saturation_rule':'first C with Q>=0.999',
            'range_caveat':'N,D bounded by B1; Q effect from semi-synthetic B6; brackets limited to grid resolution',
            'cases':rows}
    path=ROOT/'05_results/tables/q3_quality_transition_brackets.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'cases':len(rows),'context':CONTEXT,
                      'brackets':[{'Q0':r['Q0_variant'],'cost':r['cost_function'],
                                   'starts':r['start_investing_in_Q'],
                                   'saturates':r['Q_near_upper_bound'],
                                   'nonmonotone':r['nonmonotone_Q_jumps_gt_0p02']}
                                  for r in rows]},ensure_ascii=False))


if __name__=='__main__':main()
