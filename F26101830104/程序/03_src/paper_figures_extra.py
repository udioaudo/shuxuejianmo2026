"""论文补充图：Q0 灵敏度、算力增速情景、最优配置随预算的变化、C3 分布演变、质量指标相关系数。

数据均取自 05_results/tables 中已有的结果文件；C3 的分项统计由本程序直接从附件计算，
结果另存为 05_results/tables/q4_C3_distribution.json。
运行：.venv/Scripts/python.exe 03_src/paper_figures_extra.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / '05_results/tables'
FIG = ROOT / '05_results/figures'
plt.rcParams.update({'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'mathtext.fontset': 'dejavusans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 140, 'savefig.dpi': 220})
KIND = [('exponential', '指数型', '#2F5D8A', 'o'), ('power', '幂函数型', '#C8843A', 's'),
        ('logarithmic', '对数渐进型', '#327B78', '^')]
LEG = 10.5


def read(name):
    return json.loads((RESULT / name).read_text(encoding='utf-8'))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, facecolor='white')
    plt.close(fig)
    print('saved', name)


def q0_sensitivity():
    rows = read('q3_q0_sensitivity.json')['rows']
    q0 = [r['Q0'] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for key, lab, c, m in KIND:
        col = {'exponential': 'C1_exponential', 'power': 'C1_power', 'logarithmic': 'C_J_logarithmic'}[key]
        ax.plot(q0, [r[col] for r in rows], marker=m, ms=5, color=c, lw=1.8,
                label=lab + ('（跳变点 $C_J$）' if key == 'logarithmic' else '（$C_1$）'))
    ax.axvline(0.6752, color='#888888', ls='--', lw=1)
    ax.text(0.6752, 1.2e19 * 0 + 5e19, ' 基线 $Q_0=0.675$', color='#555555', fontsize=10, va='bottom')
    ax.axhline(1e19, color='#BBBBBB', ls=':', lw=1)
    ax.text(0.782, 1.08e19, '$10^{19}$ 档', color='#777777', fontsize=10)
    ax.set_yscale('log')
    ax.set_yticks([1e18, 3e18, 1e19, 3e19, 1e20], ['$10^{18}$', r'$3{\times}10^{18}$', '$10^{19}$', r'$3{\times}10^{19}$', '$10^{20}$'])
    ax.minorticks_off()
    ax.set_xlabel('基线质量 $Q_0$')
    ax.set_ylabel('临界预算（FLOPs）')
    ax.legend(frameon=False, fontsize=LEG, loc='upper left')
    save(fig, 'q3_q0_sensitivity_cn.png')


def compute_scenarios():
    fc = read('q4_frontier_dynamics.json')['forecasts']
    sc = [('historical_compute_growth', '算力按历史增速', '#2F5D8A'), ('half_compute_growth', '增速减半', '#C8843A'),
          ('no_compute_growth', '算力停止增长', '#A0413C')]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), sharey=True)
    for ax, typ, title in zip(axes, ['pretrained', 'chat_or_finetuned'], ['(a) 预训练模型', '(b) 对话/微调模型']):
        for k, (key, lab, c) in enumerate(sc):
            r = sorted([f for f in fc if f['type'] == typ and f['compute_scenario'] == key],
                       key=lambda f: f['horizon_months'])
            h = np.array([f['horizon_months'] for f in r]) + (k - 1) * 0.9
            mid = np.array([f['central'] for f in r])
            lo = mid - np.array([f['interval_2p5_97p5'][0] for f in r])
            hi = np.array([f['interval_2p5_97p5'][1] for f in r]) - mid
            ax.errorbar(h, mid, yerr=[lo, hi], fmt='o-', color=c, ms=5, lw=1.5, capsize=3, label=lab)
        f0 = [f for f in fc if f['type'] == typ][0]['F0']
        ax.axhline(f0, color='#999999', ls='--', lw=1)
        ax.text(26.5, f0 - 2.2, f'起点 {f0:.1f}', color='#666666', fontsize=10)
        ax.set_xticks([12, 24, 30])
        ax.set_xlabel('预测期（月）')
        ax.set_title(title, fontsize=12)
    axes[0].set_ylabel('六项平均分（前沿）')
    axes[0].legend(frameon=False, fontsize=LEG, loc='upper left')
    save(fig, 'q4_compute_scenarios_cn.png')


def budget_path():
    tr = {t['cost_function']: t for t in read('q3_structural_transition.json')['transitions'] if t['context'] == 4096}
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.3))
    for key, lab, c, m in KIND:
        p = tr[key]['path']
        C = [r['budget_FLOPs'] for r in p]
        axes[0].plot(C, [r['N_B'] for r in p], color=c, lw=1.6, label=lab)
        axes[0].plot(C, [r['D_B'] for r in p], color=c, lw=1.6, ls='--')
        axes[1].plot(C, [r['Q'] for r in p], color=c, lw=1.6, label=lab)
        axes[2].plot(C, [r['tokens_per_param'] for r in p], color=c, lw=1.6, label=lab)
    axes[0].set_yscale('log')
    axes[0].set_yticks([0.1, 1, 10, 100, 1000, 10000], ['0.1', '1', '10', '100', '1000', '10000'])
    axes[0].minorticks_off()
    axes[0].set_ylabel('规模（B）')
    axes[0].set_title('(a) $N^*$（实线）与 $D^*$（虚线）', fontsize=11)
    axes[1].set_ylabel('$Q^*$')
    axes[1].set_title('(b) 最优质量', fontsize=11)
    axes[2].set_ylabel('token/参数比')
    axes[2].set_title('(c) $D^*/N^*$', fontsize=11)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('总预算（FLOPs）', fontsize=11)
        ax.tick_params(labelsize=10)
    axes[1].legend(frameon=False, fontsize=9.5, loc='lower right')
    save(fig, 'q3_budget_path_cn.png')


def c3_distribution():
    c = pd.read_csv(ROOT / '02_data/raw/real_attachments/C_efficiency_evolution/leaderboard_extended_timeseries.csv')
    lb = c[c.Source == 'Open LLM Leaderboard']
    tasks = ['IFEval', 'BBH', 'MATH_Lvl5', 'GPQA', 'MUSR', 'MMLU_PRO']
    stat = {}
    for y in (2024, 2025):
        g = lb[lb.Year == y]
        stat[str(y)] = {'n': int(len(g)), 'average_median': float(g.Average.median()),
                        'average_q90': float(g.Average.quantile(.9)),
                        'task_median': {t: float(g[t].median()) for t in tasks},
                        'task_q90': {t: float(g[t].quantile(.9)) for t in tasks}}
    (RESULT / 'q4_C3_distribution.json').write_text(json.dumps(stat, ensure_ascii=False, indent=1), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw={'width_ratios': [1, 1.35]})
    ax = axes[0]
    bins = np.linspace(0, 55, 28)
    for y, col in ((2024, '#9AA7B4'), (2025, '#2F5D8A')):
        ax.hist(lb[lb.Year == y].Average, bins=bins, density=True, histtype='stepfilled' if y == 2024 else 'step',
                alpha=.6 if y == 2024 else 1, lw=1.8, color=col, label=f'{y} 年（{stat[str(y)]["n"]:,} 条）')
        ax.axvline(stat[str(y)]['average_median'], color=col, lw=1.6, ls='--')
    ax.set_xlabel('六项平均分')
    ax.set_ylabel('密度')
    ax.set_title('(a) 平均分分布（虚线为中位数）', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.set_ylim(0, 0.075)
    ax = axes[1]
    x = np.arange(len(tasks))
    w = .38
    for k, (y, col) in enumerate(((2024, '#9AA7B4'), (2025, '#2F5D8A'))):
        v = [stat[str(y)]['task_median'][t] for t in tasks]
        ax.bar(x + (k - .5) * w, v, w, color=col, label=f'{y} 年')
    ax.set_xticks(x, ['IFEval', 'BBH', 'MATH', 'GPQA', 'MUSR', 'MMLU-Pro'], fontsize=10)
    ax.set_ylabel('中位数')
    ax.set_title('(b) 各项任务得分的中位数', fontsize=11)
    ax.legend(frameon=False, fontsize=9.5)
    save(fig, 'q4_C3_distribution_cn.png')


def indicator_heatmap():
    d = read('q1_indicator_correlation.json')
    r = np.array(d['spearman'])
    cn = {'rps_doc_frac_chars_top_2gram': '重复 2-gram 占比', 'rps_doc_frac_chars_top_3gram': '重复 3-gram 占比',
          'rps_doc_frac_no_alph_words': '非字母词占比', 'rps_doc_frac_unique_words': '独特词占比',
          'rps_doc_mean_word_length': '平均词长', 'rps_doc_num_sentences': '句数',
          'rps_doc_unigram_entropy': '单词熵', 'rps_doc_word_count': '词数',
          'rps_lines_ending_with_terminal_punctution_mark': '句末标点行占比',
          'rps_lines_numerical_chars_fraction': '数字字符行占比', 'rps_lines_uppercase_letter_fraction': '大写字母行占比',
          'dsir_books': 'DSIR 书籍', 'dsir_math': 'DSIR 数学', 'dsir_wiki': 'DSIR 维基',
          'ad_en': '无广告', 'fineweb_edu': '教育价值', 'fluency_en': '流畅度', 'modernbert_cleanliness': '整洁度',
          'modernbert_professionalism': '专业性', 'modernbert_readability': '可读性',
          'modernbert_reasoning': '推理性', 'qurater': 'QuRating'}
    labels = [cn[n] for n in d['order']]
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(r, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(22), labels, rotation=90, fontsize=9)
    ax.set_yticks(range(22), labels, fontsize=9)
    for b in (10.5, 13.5):
        ax.axhline(b, color='black', lw=1.2)
        ax.axvline(b, color='black', lw=1.2)
    pairs = [('fineweb_edu', 'ad_en'), ('qurater', 'modernbert_cleanliness'),
             ('modernbert_reasoning', 'modernbert_readability'), ('fluency_en', 'modernbert_cleanliness')]
    for a, b in pairs:
        i, j = d['order'].index(a), d['order'].index(b)
        for (u, v) in ((i, j), (j, i)):
            ax.add_patch(plt.Rectangle((v - .5, u - .5), 1, 1, fill=False, ec='#111111', lw=1.6))
            ax.text(v, u, f'{r[u, v]:.2f}', ha='center', va='center', fontsize=6.5, color='black')
    for s_, pos in (('自然语言规则', 5), ('DSIR', 12), ('模型评分', 17.5)):
        ax.text(pos, -1.0, s_, ha='center', va='bottom', fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
    cb.set_label('Spearman 相关系数', fontsize=10)
    ax.spines[:].set_visible(True)
    save(fig, 'q1_indicator_correlation_cn.png')


def jacobian_heatmap():
    d = read('q2_elasticity_substitution.json')['domain_substitution_complementarity']['jacobian_17x13']
    v = np.array(d['values'])
    rows = [x.replace('train_the_pile_', '') for x in d['domains']]
    cols = [t.split('/')[-1].replace('the_pile_', '').replace('_val_loss', '') for t in d['targets']]
    from matplotlib.colors import TwoSlopeNorm
    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    im = ax.imshow(v, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=v.min(), vcenter=0, vmax=max(v.max(), 1)))
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha='right', fontsize=9.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=9.5)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if r == c or abs(v[i, j]) >= 4:
                ax.text(j, i, f'{v[i, j]:.1f}', ha='center', va='center', fontsize=7.5,
                        color='white' if v[i, j] < -8 else 'black', fontweight='bold' if r == c else 'normal')
            if r == c:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='black', lw=1.3))
    ax.set_xlabel('验证域')
    ax.set_ylabel('训练领域（占比增加）')
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_label('验证 Loss 对占比的变化率', fontsize=10)
    ax.spines[:].set_visible(True)
    save(fig, 'q2_domain_jacobian_cn.png')


def backtest_plot():
    d = read('q4_frontier_dynamics.json')
    h = d['history']['pretrained']
    rel = pd.to_datetime(h['release'])
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.step(rel, h['running_max'], where='post', color='black', lw=1.6, label='观测前沿（累计最大值）')
    cols = {'2023-06-30': '#C8843A', '2023-12-31': '#2F5D8A'}
    done = set()
    for b in d['backtest']:
        o, t = pd.Timestamp(b['origin']), pd.Timestamp(b['target'])
        c = cols[b['origin']]
        lab = f"{b['origin'][:7]} 起点的预测" if b['origin'] not in done else None
        done.add(b['origin'])
        ax.plot([o, t], [b['F_origin'], b['predicted']], ls='--', marker='o', ms=5, color=c, lw=1.5, label=lab)
        dx = pd.Timedelta(days=-10 if b['origin'] == '2023-06-30' else 10)
        ax.annotate('', xy=(t + dx, b['observed']), xytext=(t + dx, b['predicted']),
                    arrowprops=dict(arrowstyle='->', color='#A0413C', lw=1.2))
        ax.text(t + dx * 1.6, (b['observed'] + b['predicted']) / 2, f"{b['error']:.1f}", color='#A0413C', fontsize=9.5,
                ha='right' if b['origin'] == '2023-06-30' else 'left')
    names = {'huggyllama/llama-65b': 'llama-65b', '01-ai/Yi-34B': 'Yi-34B', 'Qwen/Qwen1.5-110B': 'Qwen1.5-110B',
             'Qwen/Qwen2-72B': 'Qwen2-72B', 'Qwen/Qwen2.5-72B': 'Qwen2.5-72B'}
    for r, sc, m in zip(rel, h['score'], h['model']):
        if m in names and sc >= 13:
            ax.text(r, sc + 1.2, names[m], fontsize=9, ha='center', color='#444444')
    ax.set_xlim(pd.Timestamp('2022-12-01'), pd.Timestamp('2025-03-31'))
    ax.set_ylim(8, 45)
    ax.set_ylabel('六项平均分')
    ax.set_xlabel('发布日期')
    ax.legend(frameon=False, fontsize=LEG, loc='upper left')
    save(fig, 'q4_backtest_cn.png')


def main():
    q0_sensitivity()
    compute_scenarios()
    budget_path()
    c3_distribution()
    indicator_heatmap()
    jacobian_heatmap()
    backtest_plot()


if __name__ == '__main__':
    main()
