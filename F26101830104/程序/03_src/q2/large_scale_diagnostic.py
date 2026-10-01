"""B9/B10 range and extrapolation checks; B10 losses are estimates."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'02_data/raw/real_attachments/B_scaling_laws'


def main():
    model=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    b9=pd.read_csv(DATA/'supplementary_large_models.csv')
    b10=pd.read_csv(DATA/'supplementary_large_baseline.csv')
    nlo,nhi=model['training_ranges']['N_B']
    dlo,dhi=model['training_ranges']['D_B']
    b9_tokens=pd.to_numeric(b9.D_tokens_B,errors='coerce')
    valid_token=b9_tokens>0
    report={'B1_observed_range_B':{'N':[nlo,nhi],'D':[dlo,dhi]},
            'B9':{'rows':len(b9),'N_range_B':[float(b9.N_params_B.min()),float(b9.N_params_B.max())],
                  'positive_D_range_B':[float(b9_tokens[valid_token].min()),float(b9_tokens[valid_token].max())],
                  'D_zero_or_missing_rows':int((~valid_token).sum()),
                  'N_outside_B1':int(((b9.N_params_B<nlo)|(b9.N_params_B>nhi)).sum()),
                  'positive_D_outside_B1':int(((b9_tokens[valid_token]<dlo)|
                                               (b9_tokens[valid_token]>dhi)).sum())}}
    valid=b10[['N_params_B','D_tokens_B','val_loss']].apply(pd.to_numeric,errors='coerce').dropna()
    if (valid<=0).any().any():raise ValueError('unexpected nonpositive B10 record')
    E,a,b,alpha,beta=model['params']
    pred=E+a*valid.N_params_B.to_numpy()**(-alpha)+b*valid.D_tokens_B.to_numpy()**(-beta)
    report['B10']={'rows':len(b10),'numeric_rows':len(valid),
                   'N_outside_B1':int(((valid.N_params_B<nlo)|(valid.N_params_B>nhi)).sum()),
                   'D_outside_B1':int(((valid.D_tokens_B<dlo)|(valid.D_tokens_B>dhi)).sum()),
                   'MAE_vs_estimated_not_observed_loss':float(np.mean(abs(pred-valid.val_loss.to_numpy()))),
                   'estimated_loss_mean':float(valid.val_loss.mean()),
                   'model_prediction_mean':float(pred.mean())}
    report['conclusion']='B9 shows distance beyond training support; B10 estimates are not independent real verification.'
    target=ROOT/'05_results/tables/q2_large_scale_diagnostic.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
