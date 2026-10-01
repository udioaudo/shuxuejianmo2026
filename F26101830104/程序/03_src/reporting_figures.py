"""Scientific figures generated only from saved, labeled research outputs."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'05_results/tables'
FIG=ROOT/'05_results/figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':140,'savefig.dpi':200})

def read(name):return json.loads((RESULT/name).read_text(encoding='utf-8'))

def q1():
    base=read('q1_mixture_baseline_results.json')
    inter=read('q1_mixture_interaction_comparison.json')
    labels=['Train mean','Linear ridge','Squared terms','All pairwise']
    vals=[base['per_set']['test_1m']['baseline_mae_mean_over_13'],
          base['per_set']['test_1m']['mae_mean_over_13'],
          inter['per_set']['test_1m']['square_only_mae_mean_13'],
          inter['per_set']['test_1m']['mae_mean_13']]
    fig,ax=plt.subplots(figsize=(7.1,4.1))
    bars=ax.bar(labels,vals,color=['#AAB6C2','#4D7092','#327B78','#7A93AE'])
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.008,f'{v:.3f}',ha='center')
    ax.set_ylim(0,max(vals)*1.16);ax.set_ylabel('Mean absolute error across 13 validation domains')
    ax.set_title('Problem 1: held-out 1M recipes (256 rows)')
    ax.tick_params(axis='x',rotation=12)
    fig.tight_layout();fig.savefig(FIG/'q1_mixture_1m_validation.png');plt.close(fig)

def q2():
    model=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    from fmodel.scaling import classical_loss
    data=ROOT/'02_data/raw/real_attachments/B_scaling_laws'
    fig,ax=plt.subplots(figsize=(5.6,5.1))
    for fname,label,color in [('pythia_training_log_existing.csv','B1 fit','#327B78'),
                              ('cerebras_training_log.csv','B2 semi-synthetic check','#C68050')]:
        d=pd.read_csv(data/fname)
        pred=classical_loss(d.N_params_B.to_numpy()*1e9,
                            d.D_tokens_B.to_numpy()*1e9,model['params'],1e9,1e9)
        ax.scatter(d.val_loss,pred,s=7,alpha=.35,color=color,label=label)
    ax.plot([1.5,5],[1.5,5],color='#333',lw=1,ls='--')
    ax.set_xlabel('Reference validation Loss');ax.set_ylabel('Classical-law prediction')
    ax.set_title('Problem 2: source transfer check');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(FIG/'q2_cross_source_diagnostic.png');plt.close(fig)

def q3():
    data=read('q3_bounded_budget_scenarios.json')
    rows=[r for r in data['solutions'] if r['context_tokens']==32768 and
          r['quality_reference_variant']=='equal_22']
    fig,ax=plt.subplots(figsize=(6.8,4.4))
    for kind,color in [('exponential','#377D8F'),('power','#A56A50'),('logarithmic','#577E59')]:
        part=sorted([r for r in rows if r['quality_cost']==kind],key=lambda r:r['budget_FLOPs'])
        ax.plot([r['budget_FLOPs'] for r in part],[r['Q'] for r in part],
                marker='o',label=kind,color=color)
    ax.set_xscale('log');ax.set_xticks([1e19,1e22,1e24],['1e19','1e22','1e24'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(.6,1.04);ax.set_xlabel('Budget C (FLOPs)');ax.set_ylabel('Conditional quality Q')
    ax.set_title('Problem 3: 32768-token scenario')
    ax.text(.02,.05,'B6 quality effect is semi-synthetic; N,D bounded to B1 support',
            transform=ax.transAxes,fontsize=8,color='#555')
    ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(FIG/'q3_bounded_quality_scenarios.png');plt.close(fig)

def q3_transition():
    result=read('q3_quality_transition_brackets.json')
    fig,ax=plt.subplots(figsize=(6.9,4.3))
    colors={'exponential':'#377D8F','power':'#A56A50','logarithmic':'#577E59'}
    for row in result['cases']:
        if row['Q0_variant']!='equal_22':continue
        points=row['Q_by_budget']
        ax.plot([x['budget_FLOPs'] for x in points],[x['Q'] for x in points],
                color=colors[row['cost_function']],label=row['cost_function'],lw=2)
    ax.set_xscale('log');ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Budget C (FLOPs)');ax.set_ylabel('Conditional Q')
    ax.set_ylim(.64,1.035)
    ax.set_title('Problem 3: quality activation depends on assumed cost')
    ax.legend(frameon=False)
    ax.text(.02,.05,'32768 tokens; B6 semi-synthetic Q effect; B1 N,D bounds',
            transform=ax.transAxes,fontsize=8,color='#555')
    fig.tight_layout();fig.savefig(FIG/'q3_quality_transition.png');plt.close(fig)

def q4():
    root=ROOT/'02_data/raw/real_attachments/C_efficiency_evolution'
    d=pd.read_csv(root/'loss_benchmark_bridge_expanded.csv')
    fig,ax=plt.subplots(figsize=(6.6,4.5))
    for prefix,color in [('High','#327B78'),('Medium','#C68050')]:
        subset=d[d.Loss_Comparability.str.startswith(prefix)]
        ax.scatter(subset.Val_Loss,subset.LB_Average,s=25,alpha=.75,
                   color=color,label=f'{prefix} comparability, n={len(subset)}')
    ax.set_xlabel('Validation Loss');ax.set_ylabel('Leaderboard average score')
    ax.set_title('Problem 4: bridge records have different comparability')
    ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(FIG/'q4_loss_benchmark_bridge.png');plt.close(fig)

def q4_frontier():
    data=read('q4_frontier_2027_bootstrap_uncertainty.json')
    cases=['no_further_compute_growth','half_2024_to_2025_growth',
           'same_2024_to_2025_growth']
    tnames=['time_frozen_at_last_score','half_time_trend','full_time_trend']
    colors=['#58798B','#A76A4E','#477E72']
    labels=['Frozen time','Half trend','Full trend']
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    for j,tname in enumerate(tnames):
        subset=[next(row for row in data['by_scenario']
                     if row['compute_scenario']==c and row['time_scenario']==tname)
                for c in cases]
        xs=np.arange(3)+(.16*(j-1))
        centers=np.array([r['central_fitted_score'] for r in subset])
        lows=np.array([r['bootstrap_2p5_to_97p5'][0] for r in subset])
        highs=np.array([r['bootstrap_2p5_to_97p5'][1] for r in subset])
        ax.errorbar(xs,centers,yerr=[centers-lows,highs-centers],
                    fmt='o',capsize=3,color=colors[j],label=labels[j],alpha=.9)
    ax.set_xticks(range(3),['No compute growth','Half past growth','Past growth rate'])
    ax.set_ylabel('Six-task score, conditional 90th percentile')
    ax.set_title('Problem 4: 2027 scenarios and bootstrap sensitivity')
    ax.text(.01,.02,'Whiskers show model and driver instability, not a calibrated 2027 prediction interval',
            transform=ax.transAxes,fontsize=8,color='#555')
    ax.legend(frameon=False,loc='upper left',ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'q4_frontier_2027_conditions.png');plt.close(fig)


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT/'03_src'))
    q1();q2();q3();q3_transition();q4();q4_frontier()
    print('Saved six figures in '+str(FIG))
