"""Aggregate real C8 BBH subtasks without using leaderboard summary as input."""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
DETAIL=ROOT/'02_data/raw/real_attachments/C_efficiency_evolution/detailed_results'


def main():
    rows=[]
    invalid=[]
    recovered=[]
    leaderboard=pd.read_csv(ROOT/'02_data/raw/real_attachments/C_efficiency_evolution/leaderboard_cleaned.csv')
    folders=leaderboard.assign(folder=leaderboard.Model.str.replace('/','_',regex=False))
    folder_map={folder:names.iloc[0] for folder,names in folders.groupby('folder').Model
                if names.nunique()==1}
    for path in sorted(DETAIL.rglob('*.json')):
        try:
            with path.open(encoding='utf-8') as source:
                data=json.load(source)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            problem={'file':str(path.relative_to(DETAIL)), 'error':str(exc)[:180]}
            invalid.append(problem)
            text=path.read_text(encoding='utf-8')
            match=re.search(r'"results"\s*:\s*',text)
            if not match or path.parent.name not in folder_map:
                continue
            try:
                task_results,_=json.JSONDecoder().raw_decode(text,match.end())
            except json.JSONDecodeError:
                continue
            if not isinstance(task_results,dict):
                continue
            data={'results':task_results,'model_name':folder_map[path.parent.name],
                  'date':path.stem.removeprefix('results_')}
            recovered.append(problem)
        model=data.get('model_name')
        tasks=data.get('results',{})
        metrics=[]
        for name,record in tasks.items():
            if name.startswith('leaderboard_bbh_') and isinstance(record,dict):
                score=record.get('acc_norm,none')
                try:score=float(score)
                except (TypeError,ValueError):continue
                if np.isfinite(score) and 0 <= score <= 1:
                    metrics.append(score)
        rows.append({'Model':model,'source_file':str(path.relative_to(DETAIL)),
                     'n_valid_bbh_subtasks':len(metrics),
                     'BBH_subtask_mean_pct':100*float(np.mean(metrics)) if metrics else np.nan,
                     'date':str(data.get('date',''))})
    frame=pd.DataFrame(rows)
    if len(frame)+len(invalid)-len(recovered)!=1958:
        raise ValueError(f'C8 JSON count changed: {len(frame)+len(invalid)-len(recovered)}')
    (ROOT/'08_logs/q4_invalid_C8_json.json').write_text(
        json.dumps(invalid,ensure_ascii=False,indent=2),encoding='utf-8')
    frame.to_parquet(ROOT/'02_data/processed/q4_C8_BBH_task_aggregation.parquet',index=False)
    unique=frame.sort_values('date').drop_duplicates('Model',keep='last')
    matched=unique.merge(leaderboard[['Model','BBH']],on='Model',how='inner',validate='one_to_many')
    valid=matched[(matched.n_valid_bbh_subtasks>=10)&matched.BBH.notna()]
    rho=float(spearmanr(valid.BBH,valid.BBH_subtask_mean_pct).statistic) if len(valid)>2 else None
    result={'C8_files_attempted':len(frame)+len(invalid)-len(recovered),
            'C8_files_with_usable_task_results':len(frame),
            'C8_files_fully_parsed':len(frame)-len(recovered),
            'C8_invalid_json_files':len(invalid),
            'C8_task_blocks_recovered_from_invalid_files':len(recovered),
            'C8_invalid_examples':invalid[:5],
            'unique_model_names':int(frame.Model.nunique()),
            'files_with_any_valid_BBH_subtask':int((frame.n_valid_bbh_subtasks>0).sum()),
            'median_BBH_subtasks_per_file':float(frame.n_valid_bbh_subtasks.median()),
            'C1_exact_name_join_rows':len(matched),'C1_joined_with_10plus_tasks':len(valid),
            'C1_BBH_vs_C8_subtask_mean_spearman':rho,
            'C1_BBH_vs_C8_subtask_mean_mae_pct':float(np.mean(abs(valid.BBH-valid.BBH_subtask_mean_pct))) if len(valid) else None,
            'metric_definition':'unweighted mean of available leaderboard_bbh_* acc_norm,none values times 100',
            'scope':'descriptive task aggregation; no equivalence claim to leaderboard BBH protocol'}
    (ROOT/'05_results/tables/q4_C8_task_summary.json').write_text(
        json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
