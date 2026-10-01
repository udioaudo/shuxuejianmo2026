"""Connect Q1 mixture surface and Q2 scaling as an explicitly conditional law."""

from pathlib import Path
import json
import math
import sys
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_src'))
from fmodel.scaling import classical_loss, quality_penalty
from scipy.optimize import brentq


def main():
    classical=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    quality=json.loads((ROOT/'05_results/models/q2_quality_scenario_B6.json').read_text(encoding='utf-8'))
    mixture=joblib.load(ROOT/'05_results/models/q1_square_mixture_model.joblib')
    p0=np.asarray(mixture['p_reference'],float)

    def mixture_mean(p):
        p=np.asarray(p,float)
        if p.shape!=(17,) or np.any(p<0) or abs(p.sum()-1)>1e-8:
            raise ValueError('invalid 17-domain mixture')
        features=mixture['polynomial'].transform(p[None,:-1])
        features=features[:,mixture['square_indices']]
        return float(np.mean(mixture['ridge'].predict(mixture['scaler'].transform(features))))

    def law(N_B,D_B,Q,p,lambda_p):
        if not (N_B>0 and D_B>0 and 0<Q<=1 and math.isfinite(lambda_p)):
            raise ValueError('invalid scaling input')
        base=float(classical_loss(N_B*1e9,D_B*1e9,classical['params'],1e9,1e9))
        pen=float(quality_penalty(N_B*1e9,D_B*1e9,Q,quality))
        return base+pen+lambda_p*(mixture_mean(p)-mixture_mean(p0))

    q0=.6751704350867402 # Recorded A1 equal-indicator mean; scenario scale linkage assumed.
    moved=p0.copy()
    donor=int(np.argmax(moved));receiver=int(np.argmin(moved))
    moved[donor]-=.01;moved[receiver]+=.01
    if moved.min()<0:raise ValueError('reference mixture transfer infeasible')
    example={}
    for lam in [0,.5,1]:
        example[str(lam)]={'reference':law(1,100,q0,p0,lam),
                           'one_percentage_point_transfer':law(1,100,q0,moved,lam)}
    # Q +0.1 equivalent parameter count at N=1B, D=100B, p=p_ref (scale-dependent quality term)
    target=law(1,100,q0+.1,p0,0)
    equivalent=float(math.exp(brentq(lambda lg:law(math.exp(lg),100,q0,p0,0)-target,0,40,xtol=1e-12)))
    report={'formula':'L_B1(N,D)+c(N/1e9)^-theta_N(D/1e9)^-theta_D(1-Q)^kappa+lambda_p*[f_Q1(p)-f_Q1(p_ref)]',
            'N_D_units':'billions of parameters and tokens',
            'f_Q1':'equal average of 13 losses predicted by Q1 square-only model',
            'p_reference':p0.tolist(),
            'mixture_transfer':{'from_domain':mixture['mixture_columns'][donor],
                                'to_domain':mixture['mixture_columns'][receiver],
                                'fraction':.01},
            'lambda_p_identification':'not identifiable: B1/B6 lack varied measured mixture p; 0,.5,1 are scenarios',
            'Q_scale_identification':'A1 Q and B6 Q_score not empirically calibrated',
            'conditional_example':example,
            'quality_0p1_equivalent_N_B_at_original_1B_fixed_D_p':equivalent,
            'equivalence_assumptions':'B6 semi-synthetic c,kappa,theta_N,theta_D and A1 Q numeric alignment; not a verified real tradeoff',
            'validity':'descriptive within sampled N,D,Q and feasible p neighborhood only'}
    target=ROOT/'05_results/models/q2_conditional_generalized_law.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'delta_mix_loss_for_lambda_1':example['1']['one_percentage_point_transfer']-
                       example['1']['reference'],
                      'conditional_equivalent_N_B_for_Q_plus_0p1':equivalent,
                      'status':'conditional_not_independently_identified'},ensure_ascii=False))


if __name__=='__main__':main()
