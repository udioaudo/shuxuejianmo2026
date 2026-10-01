"""22 项质量指标的描述统计与 Spearman 相关系数（A1 抽样集，方向统一与截断归一化之后）。

同时给出两种加权方案下各质量域的组均值，用于解释域级排序的变化（论文 6.1.1 节），
以及三组等权方案下的基线质量 Q0（论文 6.1.1 节）。
预处理与 quality_baseline.py 完全相同：平均词长取与 A1 中位数距离的负值，词数与句数取 ln(1+x)，
以 A1 第 1、99 百分位截断归一化，负向指标取 1−z。
运行：.venv/Scripts/python.exe 03_src/q1/indicator_correlation.py
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quality_baseline as QB  # noqa: E402

PROJECT = QB.PROJECT
OUT = PROJECT / '05_results/tables/q1_indicator_correlation.json'


def group_of(name):
    return 'DSIR 相似度' if name in QB.DSIR else '模型评分' if name in QB.MODEL else '自然语言规则'


def main():
    x, ids, domains = QB.read_source('A1', QB.FILES['A1'])
    j = QB.FIELDS.index('rps_doc_mean_word_length')
    x[:, j] = -np.abs(x[:, j] - np.nanmedian(x[:, j]))
    lo, hi = np.nanquantile(x, [.01, .99], axis=0)
    z = np.clip((x - lo) / (hi - lo), 0, 1)
    for k, name in enumerate(QB.FIELDS):
        if name in QB.REVERSE:
            z[:, k] = 1 - z[:, k]
    order = sorted(QB.FIELDS, key=lambda n: (['自然语言规则', 'DSIR 相似度', '模型评分'].index(group_of(n)), n))
    idx = [QB.FIELDS.index(n) for n in order]
    zz = z[:, idx]
    frame = pd.DataFrame(zz, columns=order)
    n_missing = frame.isna().sum().to_dict()
    rho = frame.corr(method='spearman').to_numpy()  # 两两完整样本
    groups = [group_of(n) for n in order]
    within, between = [], []
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            (within if groups[a] == groups[b] else between).append(rho[a, b])
    pairs = {'fineweb_edu|ad_en': ('fineweb_edu', 'ad_en'),
             'qurater|modernbert_cleanliness': ('qurater', 'modernbert_cleanliness'),
             'modernbert_reasoning|modernbert_readability': ('modernbert_reasoning', 'modernbert_readability'),
             'fluency_en|modernbert_cleanliness': ('fluency_en', 'modernbert_cleanliness')}
    pair_rho = {k: float(rho[order.index(a), order.index(b)]) for k, (a, b) in pairs.items()}
    # 各域的组均值
    df = pd.DataFrame(zz, columns=order)
    df['domain'] = domains
    gm = {}
    for g in ['自然语言规则', 'DSIR 相似度', '模型评分']:
        cols = [n for n in order if group_of(n) == g]
        gm[g] = df.groupby('domain')[cols].mean().mean(axis=1).round(4).to_dict()
    scores = pd.read_parquet(PROJECT / '02_data/processed/q1_quality_scores_exploratory.parquet')
    a1 = scores[scores.source == 'A1']
    dom = a1.groupby('domain')[['Q_equal', 'Q_grouped']].mean()
    rank_rho = float(spearmanr(dom.Q_equal, dom.Q_grouped).statistic)
    result = {
        'data': 'A1 sample, 51,230 records; direction-unified, clipped 1-99 percentile normalization',
        'order': order, 'groups': groups,
        'spearman': np.round(rho, 4).tolist(),
        'mean': np.round(np.nanmean(zz, axis=0), 4).tolist(), 'std': np.round(np.nanstd(zz, axis=0), 4).tolist(),
        'n_missing': {k: int(v) for k, v in n_missing.items() if v},
        'within_group_mean_abs_rho': float(np.mean(np.abs(within))),
        'between_group_mean_abs_rho': float(np.mean(np.abs(between))),
        'conflict_pair_rho': pair_rho,
        'domain_group_means': gm,
        'group_weight_equal_scheme': {g: groups.count(g) / 22 for g in set(groups)},
        'Q0_equal': float(a1.Q_equal.mean()), 'Q0_grouped': float(a1.Q_grouped.mean()),
        'domain_Q': dom.round(4).to_dict(),
        'domain_rank_spearman_equal_vs_grouped': rank_rho,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('spearman', 'order', 'groups', 'mean', 'std')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
