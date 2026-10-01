"""Q4 frontier forecast: scale/non-scale drift model with rolling-origin backtest.

Frontier dynamics (difference equation, per model type):
    F(t+h) = F(t) + beta_C * g_C * s * h + beta_t * h
beta_C (points per dex of training compute) and beta_t (points per year, non-scale
progress) come from the C1/C4 matched open-weight pretrained regression
ability = b0 + beta_C log10 C + beta_t t  (release date axis, same as the
contribution decomposition). g_C is the growth rate (dex/year) of frontier
open-weight compute in C4, s in {1, 0.5, 0} the compute-slowdown scenario.
F(t) is the observed running maximum of six-task scores of open models released
up to t (C2 Epoch release dates), separately for pretrained and chat/fine-tuned.
Replaces the earlier conditional 90th-percentile scenario, whose L1 penalty
zeroed the time term and whose level sat below the already observed frontier.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'02_data/raw/real_attachments/C_efficiency_evolution'
TASKS=['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO']
T0=pd.Timestamp('2021-01-01')
SEED=20260924
REPS=1000
SCENARIOS={'historical_compute_growth':1.0,'half_compute_growth':0.5,'no_compute_growth':0.0}
TYPES={'pretrained':['🟢 pretrained'],
       'chat_or_finetuned':['💬 chat models (RLHF, DPO, IFT, ...)','🔶 fine-tuned on domain-specific datasets']}


def years(d):return (pd.to_datetime(d)-T0).days/365.25 if np.isscalar(d) or isinstance(d,pd.Timestamp) \
    else (pd.to_datetime(d)-T0).dt.days/365.25


def load_matched():
    d=pd.read_parquet(ROOT/'02_data/processed/q4_C1_C4_matched_pretrained.parquet')
    d['date']=pd.to_datetime(d['Publication date'])
    d['logC']=np.log10(d.C_FLOPs);d['t']=years(d.date)
    d['org']=d.Model_lb.astype(str).str.split('/').str[0]
    return d


def load_frontier_pool():
    c=pd.read_csv(BASE/'leaderboard_enhanced.csv')
    c['release']=pd.to_datetime(c.Epoch_AI_Publication_Date,errors='coerce')
    c=c[c.release.notna()&c[TASKS].notna().all(axis=1)&(c.Epoch_AI_Open_Weights=='Yes')].copy()
    c['score']=c[TASKS].mean(axis=1)
    return c


def load_compute():
    e=pd.read_csv(BASE/'epoch_all_ai_models.csv',low_memory=False)
    e=e[(e.Domain=='Language')&(e['Open model weights?']=='Yes')].copy()
    e['date']=pd.to_datetime(e['Publication date'],errors='coerce')
    e['C']=pd.to_numeric(e['Training compute (FLOP)'],errors='coerce')
    return e[(e.C>0)&e.date.notna()].copy()


def compute_growth(e,end,rng=None,first_year=2021):
    """dex/year slope of the yearly 90th-percentile open-weight compute."""
    x=e[(e.date<=end)&(e.date.dt.year>=first_year)]
    rows=[]
    for y,g in x.groupby(x.date.dt.year):
        if len(g)<5:continue
        vals=g.C.to_numpy()
        if rng is not None:vals=rng.choice(vals,len(vals),replace=True)
        rows.append((y,np.log10(np.quantile(vals,.9))))
    y,v=np.array(rows).T
    return float(np.polyfit(y,v,1)[0]),[{'year':int(a),'log10_q90_C':float(b)} for a,b in rows]


def fit_drift(d):
    m=LinearRegression().fit(d[['logC','t']],d.ability)
    return float(m.coef_[0]),float(m.coef_[1])


def frontier_level(pool,types,at,top=1):
    x=pool[pool.Type.isin(types)&(pool.release<=at)]
    s=np.sort(x.score.to_numpy())[::-1]
    return float(s[:top].mean()),x.loc[x.score.idxmax()] if len(x) else None


def main():
    d=load_matched();pool=load_frontier_pool();e=load_compute()
    rng=np.random.default_rng(SEED)
    data_end=pd.to_datetime(pd.read_csv(BASE/'leaderboard_cleaned.csv')['Submission Date'],errors='coerce').max()

    # full-sample drift coefficients and compute growth
    bC,bt=fit_drift(d)
    gC,gC_table=compute_growth(e,data_end)

    # organization bootstrap for (bC,bt) and year-resampled bootstrap for gC
    orgs=d.groupby('org',sort=False);names=list(orgs.groups)
    boot=[]
    while len(boot)<REPS:
        s=pd.concat([orgs.get_group(k) for k in rng.choice(names,len(names),replace=True)],ignore_index=True)
        if s.logC.nunique()<4 or s.t.nunique()<4:continue
        b1,b2=fit_drift(s);g,_=compute_growth(e,data_end,rng)
        boot.append((b1,b2,g))
    boot=np.array(boot)

    # rolling-origin backtest: origin 2023-12-31, targets +6 and +12 months, pretrained frontier
    backtests=[]
    for origin in [pd.Timestamp('2023-06-30'),pd.Timestamp('2023-12-31')]:
        dtrain=d[d.date<=origin]
        if len(dtrain)<10:continue
        b1,b2=fit_drift(dtrain);g,_=compute_growth(e,origin)
        F0,_=frontier_level(pool,TYPES['pretrained'],origin)
        for h in [.5,1.0]:
            target=origin+pd.DateOffset(months=int(12*h))
            if target>data_end:continue
            pred=F0+b1*g*h+b2*h
            obs,_=frontier_level(pool,TYPES['pretrained'],target)
            backtests.append({'origin':str(origin.date()),'target':str(target.date()),'train_models':len(dtrain),
                              'beta_C':b1,'beta_t':b2,'g_C':g,'F_origin':F0,
                              'predicted':pred,'observed':obs,'error':pred-obs,
                              'persistence_error':F0-obs})

    # forecasts
    forecasts=[]
    for tname,types in TYPES.items():
        F0,best=frontier_level(pool,types,data_end)
        F0_top3,_=frontier_level(pool,types,data_end,top=3)
        for months in [12,24,30]:
            h=months/12
            for sname,sfac in SCENARIOS.items():
                central=F0+bC*gC*sfac*h+bt*h
                draws=F0+boot[:,0]*boot[:,2]*sfac*h+boot[:,1]*h
                forecasts.append({'type':tname,'horizon_months':months,
                                  'target_date':str((data_end+pd.DateOffset(months=months)).date()),
                                  'compute_scenario':sname,'F0':F0,'F0_top3_mean':F0_top3,
                                  'central':float(min(central,100)),
                                  'scale_part':bC*gC*sfac*h,'non_scale_part':bt*h,
                                  'interval_2p5_97p5':list(map(float,np.clip(np.quantile(draws,[.025,.975]),0,100))),
                                  'frontier_model_at_origin':str(best.Model),
                                  'frontier_model_release':str(best.release.date())})

    # frontier history for plotting
    history={}
    for tname,types in TYPES.items():
        x=pool[pool.Type.isin(types)].sort_values('release')
        history[tname]={'release':x.release.dt.strftime('%Y-%m-%d').tolist(),'score':x.score.tolist(),
                        'running_max':x.score.cummax().tolist(),'model':x.Model.tolist()}

    report={'model':'F(t+h)=F(t)+beta_C*g_C*s*h+beta_t*h',
            'data_end_latest_C1_submission':str(data_end.date()),
            'frontier_definition':'running maximum of six-task average among open-weight models (C2 Epoch release date, Epoch open weights = Yes), by type',
            'drift_coefficients':{'beta_C_points_per_dex':bC,'beta_t_points_per_year':bt,
                                  'bootstrap_2p5_97p5':{'beta_C':list(map(float,np.quantile(boot[:,0],[.025,.975]))),
                                                        'beta_t':list(map(float,np.quantile(boot[:,1],[.025,.975])))},
                                  'source':'C1/C4 matched open-weight pretrained (55), release date axis'},
            'compute_growth':{'g_C_dex_per_year':gC,'bootstrap_2p5_97p5':list(map(float,np.quantile(boot[:,2],[.025,.975]))),
                              'yearly_q90':gC_table,'source':'C4 open-weight language models, yearly 90th percentile compute'},
            'annual_frontier_drift_historical':{'scale':bC*gC,'non_scale':bt,'scale_share':bC*gC/(bC*gC+bt)},
            'backtest':backtests,'forecasts':forecasts,'history':history,
            'bootstrap':{'reps':REPS,'seed':SEED,'unit':'organization (coefficients) and model-within-year (compute growth)'},
            'notes':['Scores are bounded by 100; the linear drift is used only for 12-24 month horizons.',
                     'chat/fine-tuned frontier uses the pretrained drift; post-training gain assumed constant.']}
    out=ROOT/'05_results/tables/q4_frontier_dynamics.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('data end',data_end.date(),'beta_C %.3f beta_t %.3f g_C %.3f'%(bC,bt,gC),
          'drift/yr scale %.2f non-scale %.2f'%(bC*gC,bt))
    print('boot',report['drift_coefficients']['bootstrap_2p5_97p5'],report['compute_growth']['bootstrap_2p5_97p5'])
    print(gC_table)
    for b in backtests:print({k:(round(v,2) if isinstance(v,float) else v) for k,v in b.items()})
    for f in forecasts:
        print(f['type'],f['horizon_months'],f['target_date'],f['compute_scenario'],'F0=%.2f central=%.2f [%.2f, %.2f]'%(
            f['F0'],f['central'],*f['interval_2p5_97p5']),f['frontier_model_at_origin'])


if __name__=='__main__':main()
