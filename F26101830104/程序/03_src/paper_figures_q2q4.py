"""Chinese-labelled paper figures for Q2 and Q4 (replace the English working figures)."""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'03_src'))
from fmodel.scaling import classical_loss, quality_penalty, quality_penalty_grads

RESULT=ROOT/'05_results/tables'
FIG=ROOT/'05_results/figures'
plt.rcParams.update({'font.sans-serif':['SimHei','Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'mathtext.fontset':'dejavusans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':140,'savefig.dpi':220})


PLAIN=FuncFormatter(lambda v,_:f'{v:g}')  # avoids U+2212 missing from SimHei in 10^-k labels


def read(name):return json.loads((RESULT/name).read_text(encoding='utf-8'))


def q2_fit():
    model=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    data=ROOT/'02_data/raw/real_attachments/B_scaling_laws'
    with plt.rc_context(BIG):
        fig,axes=plt.subplots(2,1,figsize=(7.8,8.4))
        b1=pd.read_csv(data/'pythia_training_log_existing.csv')
        ax=axes[0]
        for n,g in b1.groupby('N_params_B'):
            g=g.sort_values('D_tokens_B')
            line,=ax.plot(g.D_tokens_B,g.val_loss,'o',ms=2.5,alpha=.5)
            D=np.logspace(np.log10(g.D_tokens_B.min()),np.log10(g.D_tokens_B.max()),100)
            ax.plot(D,classical_loss(np.full_like(D,n*1e9),D*1e9,model['params'],1e9,1e9),
                    color=line.get_color(),lw=1.2,label=f'{n:.2g}B')
        ax.set_xscale('log');ax.xaxis.set_major_formatter(PLAIN);ax.set_xlabel('训练数据量 D（十亿 token）');ax.set_ylabel('验证集 Loss')
        ax.set_title('(a) B1 Pythia 轨迹与经典标度律拟合')
        ax.legend(frameon=False,fontsize=LEG,ncol=2,title='参数量',title_fontsize=LEG)
        ax=axes[1]
        for fname,label,color in [('pythia_training_log_existing.csv','B1 主拟合（真实）','#327B78'),
                                  ('cerebras_training_log.csv','B2 族外验证（半合成）','#C68050')]:
            d=pd.read_csv(data/fname)
            pred=classical_loss(d.N_params_B.to_numpy()*1e9,d.D_tokens_B.to_numpy()*1e9,model['params'],1e9,1e9)
            ax.scatter(d.val_loss,pred,s=8,alpha=.35,color=color,label=label)
        b4=pd.read_csv(data/'scaling_baseline.csv')
        pred=classical_loss(b4.N_params_B.to_numpy()*1e9,b4.D_tokens_B.to_numpy()*1e9,model['params'],1e9,1e9)
        ax.scatter(b4.val_loss,pred,s=26,marker='^',color='#4D7092',label='B4 跨族收敛点（真实）')
        lo,hi=1.5,5.2
        ax.plot([lo,hi],[lo,hi],color='#333',lw=1,ls='--')
        ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
        ax.set_xlabel('观测 Loss');ax.set_ylabel('标度律预测 Loss')
        ax.set_title('(b) 族外与跨族验证');ax.legend(frameon=False,fontsize=LEG,loc='upper left',markerscale=2)
        fig.tight_layout(h_pad=1.5);fig.savefig(FIG/'q2_scaling_fit_validation.png');plt.close(fig)


M0_LABEL='无规模项形式 $M_0$（已被检验拒绝）'
# Paper figures are inserted at 15 cm width (template text width 16 cm). A 7.8 in wide figure is then
# scaled by about 0.76, so 12 pt text prints at about 9 pt and 10.5 pt legends at about 8 pt.
BIG={'font.size':12}
LEG=10.5


def q2_substitution():
    r=read('q2_elasticity_substitution.json')
    p=r['parameters'];s=r['quality_parameter_substitution'];lam=r['lambda_p']
    E,a,b,al,be,Q0=[p[k] for k in ['E','a','b','alpha','beta','Q0']]
    qj=json.loads((ROOT/'05_results/models/q2_quality_scenario_B6.json').read_text(encoding='utf-8'))
    with plt.rc_context(BIG):
        fig=plt.figure(figsize=(7.8,7.6))
        gs=fig.add_gridspec(2,4)
        axes=[fig.add_subplot(gs[0,0:2]),fig.add_subplot(gs[0,2:4]),fig.add_subplot(gs[1,1:3])]
        N=np.logspace(8,12.5,200);D=20*N
        h=quality_penalty(N,D,Q0,qj);hN,hD,hQ=quality_penalty_grads(N,D,Q0,qj)
        L=E+a*(N/1e9)**-al+b*(D/1e9)**-be+h
        ax=axes[0]
        ax.plot(N/1e9,(a*al*(N/1e9)**-al-hN)/L,label=r'$|\varepsilon_N|$')
        ax.plot(N/1e9,(b*be*(D/1e9)**-be-hD)/L,label=r'$|\varepsilon_D|$')
        ax.plot(N/1e9,-hQ*Q0/L,label=r'$|\varepsilon_Q|$')
        ax.set_xscale('log');ax.xaxis.set_major_formatter(PLAIN);ax.set_xlabel('参数量 N（十亿）')
        ax.set_ylabel('Loss 弹性绝对值');ax.set_title('(a) 弹性随规模变化');ax.legend(frameon=False,fontsize=LEG)
        ax=axes[1]
        t=pd.DataFrame(s['table'])
        ax.plot(t.N_B,t.multiplier,'o-',color='#327B78',ms=5,label='本文模型 $M_2$')
        sf=s['scale_free_comparison']
        NB=np.logspace(-1,np.log10(sf['N_max_B']*.98),200)
        ax.plot(NB,((NB)**-al-sf['loss_gain']/a)**(-1/al)/NB,color='#999999',ls='--',label='$M_0$（已被拒绝）')
        ax.set_xscale('log');ax.set_yscale('log');ax.xaxis.set_major_formatter(PLAIN);ax.yaxis.set_major_formatter(PLAIN)
        ax.set_ylim(1,50)
        ax.set_xlabel('当前参数量 N（十亿）');ax.set_ylabel("等效倍数 $N'/N$")
        ax.set_title('(b) 质量提升 0.1 的等效倍数');ax.legend(frameon=False,fontsize=LEG,loc='upper left')
        ax=axes[2]
        Ng=np.logspace(5.5,11.5,200)
        ax.plot(Ng/1e9,1/(1+(Ng/lam['N_h'])**lam['rho']),color='#4D7092',label=r'拟合 $\lambda_p(N)$')
        pts=lam['points']
        ax.scatter([x['N']/1e9 for x in pts],[x['lambda'] for x in pts],color='#C68050',zorder=3,label='问题一实测斜率')
        for x in pts:ax.plot([x['N']/1e9]*2,x['domain_p25_p75'],color='#C68050',lw=1)
        ax.set_xscale('log');ax.xaxis.set_major_formatter(PLAIN)
        ax.set_xlabel('参数量 N（十亿）');ax.set_ylabel(r'配比效应系数 $\lambda_p$')
        ax.set_title('(c) 配比效应随规模衰减');ax.legend(frameon=False,fontsize=LEG)
        fig.tight_layout(h_pad=1.5);fig.savefig(FIG/'q2_elasticity_substitution.png');plt.close(fig)


def q2_quality_scale():
    qj=json.loads((ROOT/'05_results/models/q2_quality_scenario_B6.json').read_text(encoding='utf-8'))
    model=json.loads((ROOT/'05_results/models/q2_classical_B1.json').read_text(encoding='utf-8'))
    data=ROOT/'02_data/raw/real_attachments/B_scaling_laws'
    b6=pd.read_csv(data/'supplementary_NQ_experiment.csv')
    with plt.rc_context(BIG):
        fig,axes=plt.subplots(2,1,figsize=(7.8,8.6))
        ax=axes[0]
        cells=[(n,d,np.polyfit(g.Q_score,g.val_loss,1)[0]) for (n,d),g in b6.groupby(['N_params_B','D_tokens_B'])]
        cells=pd.DataFrame(cells,columns=['N','D','slope'])
        ax.scatter(cells.N,-cells.slope,s=18,color='#C68050',alpha=.7,label='B6 格内斜率（45 个格点）')
        means=cells.groupby('N').slope.mean()
        ax.plot(means.index,-means.values,'o',color='#8B1E3F',ms=7,label='同一 N 的平均')
        m0=qj['model_comparison_B6']['M0_scale_free']['params']
        ax.axhline(m0['c']*m0['kappa'],color='#999999',ls='--',label=M0_LABEL)
        Ng=np.logspace(np.log10(.06),np.log10(14),100)
        for dd,ls in [(10,':'),(100,'-'),(600,'-.')]:
            ax.plot(Ng,qj['c']*qj['kappa']*Ng**-qj['theta_N']*dd**-qj['theta_D']*(.45)**(qj['kappa']-1),
                    color='#327B78',ls=ls,label=f'本文模型 $M_2$，D={dd}B')
        ax.set_xscale('log');ax.xaxis.set_major_formatter(PLAIN)
        ax.set_xlabel('参数量 N（十亿）');ax.set_ylabel(r'质量效应强度 $-\partial L/\partial Q$')
        ax.set_title('(a) 质量效应随模型规模减弱');ax.legend(frameon=False,fontsize=LEG,loc='upper right')
        ax=axes[1]
        b7=pd.read_csv(data/'supplementary_NQ_experiment_expanded.csv')
        b7=b7[~b7.experiment_id.isin(b6.experiment_id)]
        N,D,Q=b7.N_params_B.to_numpy()*1e9,b7.D_tokens_B.to_numpy()*1e9,b7.Q_score.to_numpy()
        base=classical_loss(N,D,model['params'],1e9,1e9)
        for q,label,color in [({'c':m0['c'],'kappa':m0['kappa']},'无规模项形式 $M_0$','#999999'),(qj,'本文模型 $M_2$','#327B78')]:
            pred=base+quality_penalty(N,D,Q,q)
            ax.scatter(b7.val_loss,pred,s=18,alpha=.7,color=color,
                       label=f'{label}（MAE {np.abs(pred-b7.val_loss).mean():.3f}）')
        lo,hi=b7.val_loss.min()-.05,b7.val_loss.max()+.05
        ax.plot([lo,hi],[lo,hi],color='#333',lw=1,ls='--')
        ax.set_xlabel('B7 新增点观测 Loss');ax.set_ylabel('预测 Loss')
        ax.set_title('(b) B7 新增 90 点检验（半合成）');ax.legend(frameon=False,fontsize=LEG,loc='upper left')
        fig.tight_layout(h_pad=1.5);fig.savefig(FIG/'q2_quality_scale_dependence.png');plt.close(fig)


def q4_bridge():
    d=pd.read_csv(ROOT/'02_data/raw/real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv')
    lin=read('q4_loss_bridge_diagnostic.json')['medium']
    sig=read('q4_loss_bridge_sigmoid.json')
    fp=sig['full_fit'];cv=sig['cv_mae'];floor=sig['high_group_floor']
    with plt.rc_context(BIG):
        fig,ax=plt.subplots(figsize=(7.8,5.2))
        for prefix,label,color in [('High','高可比（Pythia）','#327B78'),('Medium','中可比','#C68050')]:
            sub=d[d.Loss_Comparability.str.startswith(prefix)]
            ax.scatter(sub.Val_Loss,sub.LB_Average,s=30,alpha=.75,color=color,label=f'{label}，n={len(sub)}',zorder=3)
        L=np.linspace(1.6,2.9,300)
        ax.plot(L,lin['intercept']+lin['slope_per_loss']*L,color='#4D7092',lw=1.6,ls='--',
                label=f'线性映射（交叉验证 MAE {cv["linear"]:.2f}）')
        S=fp['S0']+(100-fp['S0'])*fp['r']/(1+np.exp((L-fp['L0'])/fp['s']))
        ax.plot(L,S,color='#8B1E3F',lw=2,label=f'S 形映射（交叉验证 MAE {cv["sigmoid"]:.2f}）')
        ax.text(2.88,fp['S0']+.8,f'拟合下限 $S_0$={fp["S0"]:.1f}',ha='right',va='bottom',fontsize=LEG,color='#8B1E3F')
        hl,ha=floor['loss_range'],floor['average_range']
        ax.annotate(f'高可比组 Loss 由 {hl[1]:.2f} 降至 {hl[0]:.2f}，\n均分始终在 {ha[0]:.1f} 至 {ha[1]:.1f} 分',
                    xy=(2.4,6.2),xytext=(2.26,24),fontsize=LEG,ha='left',
                    arrowprops=dict(arrowstyle='->',color='#327B78',lw=1))
        ax.set_xlim(1.6,2.9);ax.set_ylim(0,52)
        ax.set_xlabel('验证集 Loss');ax.set_ylabel('排行榜六项平均分')
        ax.legend(frameon=False,fontsize=LEG,loc='upper right')
        fig.tight_layout();fig.savefig(FIG/'q4_loss_benchmark_bridge_cn.png');plt.close(fig)


def q4_frontier():
    r=read('q4_frontier_dynamics.json')
    names={'pretrained':'(a) 预训练模型（pretrained）','chat_or_finetuned':'(b) 对话/微调模型（chat/finetuned）'}
    cols={'historical_compute_growth':'#327B78','half_compute_growth':'#C68050','no_compute_growth':'#8B1E3F'}
    lab={'historical_compute_growth':'算力按历史增速','half_compute_growth':'算力增速减半','no_compute_growth':'算力停止增长'}
    end=pd.Timestamp(r['data_end_latest_C1_submission'])
    with plt.rc_context(BIG):
        fig,axes=plt.subplots(2,1,figsize=(7.8,8.4),sharex=True,sharey=True)
        for ax,(t,h) in zip(axes,r['history'].items()):
            rel=pd.to_datetime(h['release'])
            ax.scatter(rel,h['score'],s=12,color='#AAB6C2',label='各模型六项均分')
            ax.step(rel,h['running_max'],where='post',color='black',lw=1.5,label='观测前沿（累计最大值）')
            for sname in cols:
                fs=sorted([f for f in r['forecasts'] if f['type']==t and f['compute_scenario']==sname],
                          key=lambda f:f['horizon_months'])
                x=[end]+[pd.Timestamp(f['target_date']) for f in fs]
                y=[fs[0]['F0']]+[f['central'] for f in fs]
                lo=[fs[0]['F0']]+[f['interval_2p5_97p5'][0] for f in fs]
                hi=[fs[0]['F0']]+[f['interval_2p5_97p5'][1] for f in fs]
                ax.plot(x,y,'o-',ms=4,color=cols[sname],label=lab[sname])
                ax.fill_between(x,lo,hi,color=cols[sname],alpha=.12)
            ax.axvline(end,color='#555',ls=':',lw=1)
            ax.set_title(names[t]);ax.set_ylabel('六项平均分')
        axes[0].text(end,2,' 预测起点 2025-03',fontsize=10,color='#555',va='bottom')
        axes[-1].set_xlabel('发布日期')
        axes[0].legend(frameon=False,fontsize=LEG,loc='upper left')
        fig.tight_layout(h_pad=1.5);fig.savefig(FIG/'q4_frontier_forecast_cn.png');plt.close(fig)


def q4_contribution():
    r=read('q4_contribution_uncertainty.json')
    total=r['observed_cohort_score_change'];s=r['full_sample_scale_association'];t=r['full_sample_time_association']
    parts=[('规模扩张\n（算力）',s,r['scale_association_2p5_97p5'],'#4D7092'),
           ('非规模技术进步\n（时间趋势）',t,r['time_association_2p5_97p5'],'#C8843A')]
    with plt.rc_context(BIG):
        fig,ax=plt.subplots(figsize=(6.2,3.4))
        for i,(name,v,ci,col) in enumerate(parts):
            ax.barh(i,v,color=col,height=.6);ax.errorbar(v,i,xerr=[[v-ci[0]],[ci[1]-v]],color='black',capsize=5)
            ax.text(ci[1]+.3,i,f'{v:.2f} 分（{v/total:.0%}）',va='center',fontsize=11)
        ax.set_yticks([0,1],[p[0] for p in parts]);ax.invert_yaxis()
        ax.set_xlim(0,max(p[2][1] for p in parts)*1.5)
        ax.set_xlabel(f'得分提升（分），2023→2024 平均提升 {total:.2f} 分')
        fig.tight_layout();fig.savefig(FIG/'q4_contribution_cn.png');plt.close(fig)


def main():
    q2_fit();q2_substitution();q2_quality_scale();q4_bridge();q4_frontier();q4_contribution()
    print('figures written to',FIG)


if __name__=='__main__':main()
