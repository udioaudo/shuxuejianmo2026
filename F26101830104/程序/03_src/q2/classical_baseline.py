"""B1 classical law and clearly labeled external diagnostics."""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "03_src"))
from fmodel.scaling import fit_classical, classical_loss
from fmodel.mixture import regression_metrics

ROOT = PROJECT / "02_data/raw/real_attachments/B_scaling_laws"
FILES = {"B1": ("pythia_training_log_existing.csv", "real_main_fit"),
         "B2": ("cerebras_training_log.csv", "semi_synthetic_out_of_family"),
         "B4": ("scaling_baseline.csv", "real_cross_family_measurement_basis_uncertain"),
         "B5": ("published_scaling_data.csv", "literature_validation_loss_comparability_uncertain")}


def valid_data(path):
    data = pd.read_csv(path)
    columns = ["N_params_B", "D_tokens_B", "val_loss"]
    if not set(columns).issubset(data.columns):
        raise ValueError(f"Missing required columns in {path}")
    x = data[columns].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(x.to_numpy()).all(axis=1) & (x.to_numpy() > 0).all(axis=1)
    return data.loc[valid].copy(), int((~valid).sum())


def main():
    training, invalid = valid_data(ROOT / FILES["B1"][0])
    if invalid:
        raise ValueError("Unexpected invalid observations in main fitting data")
    N = training["N_params_B"].to_numpy()*1e9
    D = training["D_tokens_B"].to_numpy()*1e9
    y = training["val_loss"].to_numpy()
    model = fit_classical(N, D, y, Nref=1e9, Dref=1e9, starts=16)
    model.update({"parameter_order": ["E", "a", "b", "alpha", "beta"],
                  "original_units": {"N":"parameters", "D":"tokens", "L":"validation cross-entropy"},
                  "training_file": str((ROOT / FILES["B1"][0]).relative_to(PROJECT)),
                  "training_n":len(training),
                  "training_checkpoint_rows":len(training),
                  "training_model_sizes":int(training.N_params_B.nunique()),
                  "training_ranges":{"N_B":[float(training.N_params_B.min()),float(training.N_params_B.max())],
                                     "D_B":[float(training.D_tokens_B.min()),float(training.D_tokens_B.max())]},
                  "limitations":["B2 is semi-synthetic and not independent real observation",
                                 "B4/B5 losses may use different validation corpora/tokenizers",
                                 "no Q or p effect in this baseline"]})
    rows = {}
    for code, (filename, status) in FILES.items():
        source, invalid = valid_data(ROOT / filename)
        n,d=source.N_params_B.to_numpy()*1e9,source.D_tokens_B.to_numpy()*1e9
        observed=source.val_loss.to_numpy()
        pred=classical_loss(n,d,model["params"],Nref=1e9,Dref=1e9)
        met=regression_metrics(observed,pred)
        rho,_=spearmanr(observed,pred)
        met.update({"status":status,"invalid_rows":invalid,"spearman":float(rho),
                    "observed_mean":float(observed.mean()),"predicted_mean":float(pred.mean()),
                    "fraction_N_outside_B1":float(np.mean((n/1e9 < training.N_params_B.min()) |
                                                          (n/1e9 > training.N_params_B.max()))),
                    "fraction_D_outside_B1":float(np.mean((d/1e9 < training.D_tokens_B.min()) |
                                                          (d/1e9 > training.D_tokens_B.max())))})
        if code == "B1":
            by_run=[]
            for run_id, g in source.assign(prediction=pred).groupby("run_id",sort=True):
                by_run.append({"run_id":str(run_id),"n":len(g),
                               "mae":float(np.mean(np.abs(g.prediction-g.val_loss)))})
            met["by_run_training_diagnostics"] = by_run
            met["warning"]="In-sample fit; no run-held-out validation in this result."
        if code == "B2":
            met["warning"]="Semi-synthetic, generated with calibration to real trajectories."
        if code in ("B4", "B5"):
            met["warning"]="Numerical comparison only until validation corpus and tokenizer are aligned."
        rows[code]=met
    out_model=PROJECT/"05_results/models/q2_classical_B1.json"
    out_report=PROJECT/"05_results/tables/q2_classical_validation.json"
    out_model.parent.mkdir(parents=True,exist_ok=True)
    out_report.parent.mkdir(parents=True,exist_ok=True)
    out_model.write_text(json.dumps(model,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    out_report.write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"fitted_params":model["params"],"jacobian_rank":model["jacobian_rank"],
                      "jacobian_condition":model["jacobian_condition"],
                      "by_source":{k:{x:v[x] for x in ("n","mae","spearman","fraction_N_outside_B1",
                                                     "fraction_D_outside_B1")} for k,v in rows.items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
