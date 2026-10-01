"""Test whether B2 needs a source-specific calibration; keep B4 untouched."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'02_data/raw/real_attachments/B_scaling_laws'


def main():
    model=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    E,a,b,alpha,beta=model['params']
    def base(df):
        return E+a*df.N_params_B.to_numpy(float)**(-alpha)+b*df.D_tokens_B.to_numpy(float)**(-beta)
    def features(df):
        d=df.D_tokens_B.to_numpy(float)
        return np.column_stack([np.ones(len(d)),d**(-.5)])
    b2=pd.read_csv(DATA/'cerebras_training_log.csv')
    residual=b2.val_loss.to_numpy(float)-base(b2)
    preds=np.empty(len(b2))
    runs=[]
    for n in sorted(b2.N_params_B.unique()):
        train=b2.N_params_B!=n
        test=~train
        coeff=np.linalg.lstsq(features(b2.loc[train]),residual[train],rcond=None)[0]
        preds[test]=base(b2.loc[test])+features(b2.loc[test])@coeff
        runs.append({'held_N_B':float(n),'rows':int(test.sum()),
                     'base_mae':float(np.mean(abs(residual[test]))),
                     'calibrated_mae':float(np.mean(abs(b2.loc[test].val_loss.to_numpy()-preds[test])))})
    full=np.linalg.lstsq(features(b2),residual,rcond=None)[0]
    b4=pd.read_csv(DATA/'scaling_baseline.csv')
    cerebras=b4[b4.family=='Cerebras-GPT']
    raw=base(cerebras)
    adjusted=raw+features(cerebras)@full
    r={'calibration_data':'B2 semi-synthetic Cerebras trajectories only',
       'formula':'L_B1 + offset + slope / sqrt(D_tokens_B)',
       'B2_model_size_holdout':runs,
       'B2_held_trajectory_mean_MAE':float(np.mean(abs(b2.val_loss.to_numpy()-preds))),
       'B2_unadjusted_MAE':float(np.mean(abs(residual))),
       'B2_full_fit_offset':float(full[0]),'B2_full_fit_slope':float(full[1]),
       'B4_Cerebras_real_cross_family_points':len(cerebras),
       'B4_Cerebras_raw_MAE':float(np.mean(abs(cerebras.val_loss.to_numpy()-raw))),
       'B4_Cerebras_after_B2_calibration_MAE':float(np.mean(abs(cerebras.val_loss.to_numpy()-adjusted))),
       'interpretation':'B2 adaptation is source-specific; worsening on B4 is evidence against a universal family correction.'}
    target=ROOT/'05_results/tables/q2_cerebras_source_calibration.json'
    target.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'heldout_B2_MAE':r['B2_held_trajectory_mean_MAE'],
                      'unadjusted_B2_MAE':r['B2_unadjusted_MAE'],
                      'B4_raw_MAE':r['B4_Cerebras_raw_MAE'],
                      'B4_adjusted_MAE':r['B4_Cerebras_after_B2_calibration_MAE']},ensure_ascii=False))


if __name__=='__main__':main()
