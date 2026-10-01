"""Model-driven large-budget extrapolations alongside support-bounded solutions."""

from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_src/q3'))
from budget_scenarios import optimize_one


def main():
    scaling=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    quality=json.loads((ROOT/'05_results/models/q2_quality_scenario_B6.json').read_text(encoding='utf-8'))
    b9=pd.read_csv(ROOT/'02_data/raw/real_attachments/B_scaling_laws/supplementary_large_models.csv')
    nlo,nhi_real=scaling['training_ranges']['N_B']
    dlo,dhi_real=scaling['training_ranges']['D_B']
    upperN=float(b9.N_params_B.max())
    upperD=float(b9.loc[b9.D_tokens_B>0,'D_tokens_B'].max())
    model=dict(scaling)
    model['training_ranges']={'N_B':[nlo,upperN],'D_B':[dlo,upperD]}
    old=json.loads((ROOT/'05_results/tables/q3_bounded_budget_scenarios.json').read_text(encoding='utf-8'))
    q0s=sorted({(x['quality_reference_variant'],x['Q0']) for x in old['solutions']})
    contexts=sorted({x['context_tokens'] for x in old['solutions']})
    rows=[]
    for label,q0 in q0s:
        for C in [1e19,1e22,1e24]:
            for context in contexts:
                for kind in ['exponential','power','logarithmic']:
                    row=optimize_one(C,context,kind,q0,model,quality)
                    row['quality_reference_variant']=label
                    row['outside_B1_N']=bool(row['N_params']/1e9>nhi_real)
                    row['outside_B1_D']=bool(row['D_tokens']/1e9>dhi_real)
                    rows.append(row)
    output={'status':'model-driven extrapolation, not empirically validated high-budget allocation',
            'source_of_bounds':'B1 lower ranges, B9 metadata maxima (not observed high-scale loss)',
            'B9_N_max_B':upperN,'B9_positive_D_max_B':upperD,
            'B10_status':'loss estimates derived from a scaling model; not independent validation',
            'quality_status':'B6 semi-synthetic correction; A1 Q numeric linkage assumed',
            'p_policy':'fixed Q1 reference mixture, as in bounded scenario',
            'scenarios':rows}
    path=ROOT/'05_results/tables/q3_extrapolated_budget_scenarios.json'
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    high=[x for x in rows if x['budget_FLOPs']==1e24 and x['context_tokens']==32768 and
          x['quality_reference_variant']=='equal_22']
    print(json.dumps({'cases':len(rows),'high_budget_32768':[
        {'kind':x['quality_cost'],'N_B':x['N_params']/1e9,'D_B':x['D_tokens']/1e9,
         'Q':x['Q'],'spent_fraction':x['spent_fraction'],
         'outside_B1_N':x['outside_B1_N'],'outside_B1_D':x['outside_B1_D']}
        for x in high]},ensure_ascii=False))


if __name__=='__main__':main()
