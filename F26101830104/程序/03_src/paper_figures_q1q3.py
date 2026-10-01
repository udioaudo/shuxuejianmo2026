"""Chinese-labelled paper figures for the Q1 cross-scale check and Q3 transition."""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'05_results/tables'
FIG=ROOT/'05_results/figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.sans-serif':['SimHei','Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'mathtext.fontset':'dejavusans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':140,'savefig.dpi':220})
COLORS={'train':'#4D7092','quality':'#C8843A','attention':'#9AA7B4'}
KIND_CN={'exponential':'指数型','power':'幂函数型','logarithmic':'对数渐进型'}


def read(name):return json.loads((RESULT/name).read_text(encoding='utf-8'))


# Paper sizing: full-width figures are 7.8 in wide and inserted at 15 cm; narrow ones are 6.2 in and inserted
# at 12 cm. Both scale by about 0.76, so 12 pt text prints at about 9 pt and 10.5 pt legends at about 8 pt.
BIG={'font.size':12}
LEG=10.5


def q1_cross_scale():
    r=read('q1_cross_scale_rank.json')['per_set']
    keys=['test_1m','test_60m','test_1B','est_10b','est_70b']
    labels=['1M','60M','1B','10B（估算）','70B（估算）']
    rho=[r[k]['spearman_mean_13'] for k in keys]
    hit=[r[k]['top10pct_hit_rate_mean_13'] for k in keys]
    raw=[r[k]['raw_mae_mean_13'] for k in keys]
    cal=[r[k]['calibrated_mae_mean_13'] for k in keys]
    x=np.arange(len(keys));w=.36
    with plt.rc_context(BIG):
        fig,axes=plt.subplots(2,1,figsize=(7.8,7.2))
        ax=axes[0]
        b1=ax.bar(x-w/2,rho,w,color='#327B78',label='Spearman 秩相关')
        b2=ax.bar(x+w/2,hit,w,color='#AAB6C2',label='前 10% 配方命中率')
        for bars in (b1,b2):
            for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.015,f'{b.get_height():.2f}',
                                  ha='center',fontsize=10)
        ax.axvspan(2.5,4.5,color='#EEEEEE',zorder=0)
        ax.set_xticks(x,labels);ax.set_ylim(0,1.25);ax.set_ylabel('13 个验证域平均值')
        ax.set_title('(a) 1M 模型对各规模配方的排序能力');ax.legend(frameon=False,fontsize=LEG,loc='upper right',ncol=2)
        ax=axes[1]
        b1=ax.bar(x-w/2,raw,w,color='#7A93AE',label='直接套用 1M 模型')
        b2=ax.bar(x+w/2,cal,w,color='#327B78',label='每规模 16 个配方仿射校准后')
        for bars in (b1,b2):
            for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.05,f'{b.get_height():.2f}',
                                  ha='center',fontsize=10)
        ax.axvspan(2.5,4.5,color='#EEEEEE',zorder=0)
        ax.set_xticks(x,labels);ax.set_ylabel('平均绝对误差 MAE');ax.set_ylim(0,max(raw)*1.18)
        ax.set_title('(b) 绝对 Loss 误差与规模校准');ax.legend(frameon=False,fontsize=LEG,loc='upper left')
        fig.tight_layout(h_pad=1.5);fig.savefig(FIG/'q1_cross_scale_rank.png');plt.close(fig)


def q1_domain_effects():
    rows=read('q1_cross_scale_rank.json')['domain_directional_effects']['rows']
    names=[r['domain'] for r in rows];vals=[r['mean_delta_loss_13'] for r in rows]
    with plt.rc_context(BIG):
        fig,ax=plt.subplots(figsize=(6.2,5.8))
        y=np.arange(len(rows))
        ax.barh(y,vals,color=['#327B78' if v<0 else '#C8843A' for v in vals])
        ax.axvline(0,color='black',lw=.8)
        ax.set_yticks(y,names,fontsize=LEG);ax.invert_yaxis()
        ax.set_xlabel('13 域平均 Loss 变化（占比增加一个训练标准差）')
        fig.tight_layout();fig.savefig(FIG/'q1_domain_effects.png');plt.close(fig)


def q3_shares():
    """Portrait layout: one row per cost function, sized for ~15 cm width in the paper (text about 9 pt)."""
    r=read('q3_structural_transition.json')
    with plt.rc_context({'font.size':12}):
        fig,axes=plt.subplots(3,1,figsize=(7.8,9.8),sharex=True)
        for i,(ax,t) in enumerate(zip(axes,r['transitions'])):
            path=t['path']
            C=np.array([p['budget_FLOPs'] for p in path])
            st=np.array([p['share_train'] for p in path])
            sq=np.array([p['share_quality'] for p in path])
            sa=np.array([p['share_attention'] for p in path])
            Q=np.array([p['Q'] for p in path])
            ax.stackplot(C,st,sq,sa,colors=[COLORS['train'],COLORS['quality'],COLORS['attention']],
                         labels=['训练开销','质量提升开销','注意力开销'],alpha=.9)
            ax.set_xscale('log');ax.set_ylim(0,1);ax.set_xlim(C[0],C[-1])
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10,numticks=12))
            cuts=[np.sqrt(np.prod(s['budget_bracket_FLOPs'])) for s in t['numerical_regime_switches']]
            for c in cuts:ax.axvline(c,color='black',ls='--',lw=1)
            for c in [1e19,1e22,1e24]:ax.axvline(c,color='white',lw=.8,alpha=.8)
            edges=[C[0]]+cuts+[C[-1]]
            names=['R1','R2','R3'] if len(cuts)==2 else ['R1','R3']
            for lo,hi,nm in zip(edges[:-1],edges[1:],names):
                ax.text(np.sqrt(lo*hi),.12,nm,ha='center',va='center',color='white',fontsize=11,fontweight='bold')
            tw=ax.twinx();tw.plot(C,Q,color='#8B1E3F',lw=1.8)
            tw.set_ylim(0.6,1.02);tw.spines['top'].set_visible(False);tw.set_ylabel('最优质量 $Q^*$')
            ax.set_ylabel('预算占比')
            ax.set_title(f"({'abc'[i]}) {KIND_CN[t['cost_function']]}质量成本",fontsize=12)
        axes[-1].set_xlabel('总算力预算 C（FLOPs）')
        h1,l1=axes[0].get_legend_handles_labels()
        fig.legend(h1+[plt.Line2D([],[],color='#8B1E3F',lw=1.8),plt.Line2D([],[],color='black',ls='--',lw=1)],
                   l1+['最优质量 $Q^*$（右轴）','区制切换点'],loc='lower center',ncol=3,frameon=False,
                   fontsize=10.5,bbox_to_anchor=(.5,.005))
        fig.tight_layout(rect=(0,.06,1,1));fig.savefig(FIG/'q3_budget_shares.png');plt.close(fig)


def q3_context():
    r=read('q3_structural_transition.json')
    rows=r['context_thresholds']
    L=np.array([x['context'] for x in rows])
    with plt.rc_context(BIG):
        fig,ax=plt.subplots(figsize=(6.2,4.6))
        ax.plot(L,[x['C1_exponential'] for x in rows],'o-',label='指数型：开始提升质量 $C_1$')
        ax.plot(L,[x['C1_power'] for x in rows],'s-',label='幂函数型：开始提升质量 $C_1$')
        ax.plot(L,[x['C_J_logarithmic'] for x in rows],'^-',label='对数渐进型：跳变至 Q=1 的 $C_J$')
        ax.axvline(r['critical_context_tokens'],color='#8B1E3F',ls='--',lw=1)
        ax.text(r['critical_context_tokens']*1.08,3e19,'$L^*_{ctx}=6/\\eta=30000$',color='#8B1E3F',fontsize=LEG)
        ax.set_xscale('log',base=2);ax.set_yscale('log')
        ax.set_xticks(L,[f'{v:,}' for v in L],fontsize=10.5)
        ax.set_xlabel('上下文长度 $L_{ctx}$（C7 可行取值）');ax.set_ylabel('临界预算（FLOPs）')
        ax.legend(frameon=False,fontsize=LEG,loc='lower left')
        fig.tight_layout();fig.savefig(FIG/'q3_context_thresholds.png');plt.close(fig)


def q1_quality_domains():
    import pandas as pd
    s=pd.read_parquet(ROOT/'02_data/processed/q1_quality_scores_exploratory.parquet')
    order=['arxiv','commoncrawl','c4','stackexchange','book','wikipedia','github']
    groups=[s.loc[(s.source=='A1')&(s.domain==d),'Q_equal'].dropna().to_numpy() for d in order]
    labels=[f'{d}（A1）' for d in order]
    groups+= [s.loc[s.source=='A2','Q_equal'].dropna().to_numpy(),s.loc[s.source=='A3','Q_equal'].dropna().to_numpy()]
    labels+=['arxiv（A2 扩展）','github（A3 扩展）']
    with plt.rc_context(BIG):
        fig,ax=plt.subplots(figsize=(7.8,4.8))
        bp=ax.boxplot(groups,showfliers=False,patch_artist=True,widths=.55)
        for i,b in enumerate(bp['boxes']):
            b.set_facecolor('#C8843A' if i>=7 else '#9FB6C8');b.set_edgecolor('#333')
        for m in bp['medians']:m.set_color('black')
        ax.scatter(range(1,len(groups)+1),[g.mean() for g in groups],marker='D',color='#8B1E3F',s=24,zorder=3,label='域均值 $Q_d$')
        ax.set_xticks(range(1,len(groups)+1),labels,fontsize=LEG,rotation=30,ha='right',rotation_mode='anchor')
        ax.set_ylabel('样本质量分 $Q_i$（22 项等权）')
        ax.legend(frameon=False,loc='lower left',fontsize=LEG)
        fig.tight_layout();fig.savefig(FIG/'q1_quality_domains.png');plt.close(fig)


def q0_workflow():
    """Vertical flow: one row per question, model box on the left, outputs on the right."""
    from matplotlib.patches import FancyBboxPatch
    rows=[('问题一','质量评价与冲突消解（A1–A3）\n领域配比模型（A4–A15）','质量分 $Q_0$，配比模型 $f(p)$\n跨规模斜率 $b_s$'),
          ('问题二','经典标度律（B1）\n质量项（B6、B7），配比项 $\\lambda_p(N)$','广义标度律 $L(N,D,Q,p)$\n弹性与替代条件'),
          ('问题三','预算约束优化与 KKT 分析\n上下文长度取值（C7）','最优 $N^*,D^*,Q^*$\n结构性转移临界预算'),
          ('问题四','贡献分解（C1、C3、C4）\n前沿动力学，桥接（C6），C8 聚合','规模与非规模贡献\n12/24 个月前沿预测')]
    with plt.rc_context(BIG):
        W,H=7.8,6.6
        fig,ax=plt.subplots(figsize=(W,H));ax.axis('off');ax.set_xlim(0,W);ax.set_ylim(0,H)
        bh,step=1.15,1.62
        lx,lw,rx,rw=0.08,4.35,4.95,2.77
        for i,(title,body,out) in enumerate(rows):
            yc=H-0.72-i*step
            ax.add_patch(FancyBboxPatch((lx,yc-bh/2),lw,bh,boxstyle='round,pad=0.03',fc='#EEF2F6',ec='#4D7092'))
            ax.text(lx+0.12,yc,title,ha='left',va='center',fontsize=12.5,fontweight='bold')
            ax.text(lx+1.05,yc,body,ha='left',va='center',fontsize=11,linespacing=1.45)
            ax.add_patch(FancyBboxPatch((rx,yc-bh/2),rw,bh,boxstyle='round,pad=0.03',fc='#FBF1E6',ec='#C8843A'))
            ax.text(rx+rw/2,yc,out,ha='center',va='center',fontsize=11,linespacing=1.45)
            ax.annotate('',xy=(rx-0.04,yc),xytext=(lx+lw+0.05,yc),arrowprops=dict(arrowstyle='->',lw=1.4,color='#555'))
            if i<len(rows)-1:
                ybot=yc-bh/2-0.04;ymid=yc-step/2;ytop=yc-step+bh/2+0.04
                ax.plot([rx+rw/2,rx+rw/2,lx+lw/2,lx+lw/2],[ybot,ymid,ymid,ymid],color='#8B1E3F',lw=1.4)
                ax.annotate('',xy=(lx+lw/2,ytop),xytext=(lx+lw/2,ymid),arrowprops=dict(arrowstyle='->',lw=1.4,color='#8B1E3F'))
        fig.tight_layout(pad=0.2);fig.savefig(FIG/'q0_workflow.png');plt.close(fig)


def main():
    q1_cross_scale();q1_domain_effects();q3_shares();q3_context();q1_quality_domains();q0_workflow()
    print('figures written to',FIG)


if __name__=='__main__':main()
