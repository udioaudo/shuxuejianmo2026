"""B1 trajectory-held-out check; all checkpoints of one model stay together."""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_src'))
from fmodel.scaling import fit_classical,classical_loss
from fmodel.mixture import regression_metrics

def main():
    data=pd.read_csv(ROOT/'02_data/raw/real_attachments/B_scaling_laws/pythia_training_log_existing.csv')
    if data[['run_id','N_params_B','D_tokens_B','val_loss']].isna().any().any():
        raise ValueError('missing B1 modeling values')
    runs=[]
    for name in sorted(data.N_params_B.unique()):
        train=data.loc[data.N_params_B!=name]
        held=data.loc[data.N_params_B==name]
        m=fit_classical(train.N_params_B.to_numpy()*1e9,
                        train.D_tokens_B.to_numpy()*1e9,
                        train.val_loss.to_numpy(),Nref=1e9,Dref=1e9,starts=6)
        prediction=classical_loss(held.N_params_B.to_numpy()*1e9,
                                  held.D_tokens_B.to_numpy()*1e9,m['params'],1e9,1e9)
        result=regression_metrics(held.val_loss,prediction)
        result.update({'held_model_size_N_B':float(name),
                       'fit_params':m['params']})
        runs.append(result)
    report={'protocol':'Leave one Pythia parameter size and all 147 associated checkpoints out, refit B1 each time',
            'evidence_status':'internal_cross_trajectory_check_not_independent_model_family',
            'mean_trajectory_mae':float(np.mean([r['mae'] for r in runs])),
            'max_trajectory_mae':float(np.max([r['mae'] for r in runs])),
            'runs':runs,
            'caveat':'Trajectory-held-out errors cannot establish cross-family comparability or data provenance.'}
    target=ROOT/'05_results/tables/q2_B1_trajectory_holdout.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'trajectory_count':len(runs),
                      'mean_trajectory_mae':report['mean_trajectory_mae'],
                      'max_trajectory_mae':report['max_trajectory_mae']},ensure_ascii=False))

if __name__=='__main__':main()
