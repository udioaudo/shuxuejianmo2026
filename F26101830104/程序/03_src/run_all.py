"""Reproduce all current empirical analyses from registered F data."""

from pathlib import Path
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
STEPS=[
    'q1/quality_baseline.py',
    'q1/conflict_resolution.py',
    'q1/text_check_samples.py',
    'q1/lexical_mapping.py',
    'q1/mixture_baseline.py',
    'q1/mixture_interactions.py',
    'q1/external_robustness.py',
    'q1/cross_scale_rank.py',
    'q2/classical_baseline.py',
    'q2/run_holdout.py',
    'q2/quality_scenario.py',
    'q2/conditional_generalized_law.py',
    'q2/large_scale_diagnostic.py',
    'q2/source_calibration.py',
    'q3/budget_scenarios.py',
    'q3/extrapolated_budgets.py',
    'q3/transition_analysis.py',
    'q3/structural_transition.py',
    'q3/q0_sensitivity.py',
    # Reads the Q3 optimum and baseline context, so it runs after structural_transition.
    'q2/elasticity_substitution.py',
    'q4/task_aggregation.py',
    'q4/timeseries_profile.py',
    'q4/loss_bridge.py',
    'q4/loss_bridge_sigmoid.py',
    'q4/scale_time_diagnostic.py',
    'q4/contribution_uncertainty.py',
    'q4/frontier_scenario.py',
    'q4/pretrained_frontier_backtest.py',
    'q4/frontier_revised.py',
    'q4/frontier_uncertainty.py',
    'q4/frontier_dynamics.py',
    'reporting_figures.py',
    'paper_figures_q1q3.py',
    'paper_figures_q2q4.py',
]


def main():
    env=os.environ.copy()
    env.update({'PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    log=ROOT/'08_logs/full_pipeline.log'
    with log.open('w',encoding='utf-8') as record:
        record.write('Started '+datetime.now(timezone.utc).isoformat()+'\n')
        for rel in STEPS:
            command=[sys.executable,str(ROOT/'03_src'/rel)]
            completed=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,
                                     encoding='utf-8',errors='replace')
            message=f'{rel}: exit={completed.returncode}\n{completed.stdout}{completed.stderr}\n'
            record.write(message)
            record.flush()
            print(f'{rel}: '+('ok' if completed.returncode==0 else 'FAILED'),flush=True)
            if completed.returncode:
                raise RuntimeError(f'{rel} failed; see {log}')
        record.write('Completed '+datetime.now(timezone.utc).isoformat()+'\n')
    print('Reproduction log: '+str(log),flush=True)


if __name__=='__main__':main()
