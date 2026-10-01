"""Draw A1 samples for manual text verification of the quality score (Q1 requirement).

Random draws with a fixed seed (no cherry-picking) from: top 1% Q, bottom 1% Q,
and each conflict type with enough cases. Writes a Markdown reading sheet with the
score, key normalized indicators, head/middle/tail excerpts (whole text if short),
optional Chinese reading notes (05_results/text_check/q1_text_check_cn_notes.json), and the
readers' judgements (05_results/text_check/q1_text_check_judgments.json, blank if absent).
With judgements present it also scores the conflict arbitration against them: for every
sample judged inconsistent with a stated direction (Q too high / too low), does the
arbitrated Q move the same way? Written to 05_results/tables/q1_text_check_validation.json.
"""

from pathlib import Path
import json
import lzma
import sys
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / '03_src'))
sys.path.insert(0, str(PROJECT / '03_src/q1'))
from fmodel.quality import semantic_conflict
import quality_baseline as qb
import conflict_resolution as cr

SHORT = 1500          # documents up to this length are shown in full
HEAD, MID, TAIL = 600, 400, 400
SHOW = ['fineweb_edu', 'ad_en', 'fluency_en', 'modernbert_cleanliness', 'modernbert_readability',
        'modernbert_reasoning', 'qurater']
CN = {'fineweb_edu': '教育价值', 'ad_en': '无广告', 'fluency_en': '流畅度', 'modernbert_cleanliness': '整洁度',
      'modernbert_readability': '可读性', 'modernbert_reasoning': '推理性', 'qurater': 'QuRating'}


def snippets(doc):
    """Whole text if short, else head / middle / tail windows with their character offsets."""
    if len(doc) <= SHORT:
        return [('全文', doc)]
    m = len(doc) // 2 - MID // 2
    return [(f'开头（第 1 至 {HEAD:,} 字符）', doc[:HEAD]),
            (f'中间（第 {m + 1:,} 至 {m + MID:,} 字符）', doc[m:m + MID]),
            (f'结尾（最后 {TAIL:,} 字符）', doc[-TAIL:])]


GROUP_SHORT = ['高分样本', '低分样本', '教育价值与无广告冲突', '推理性与可读性冲突', '流畅度与整洁度冲突']


def summarize(groups, ids, scores, judged):
    by_group, moves = [], []
    for (title, idx), short in zip(groups, GROUP_SHORT):
        rows = [judged[ids[i]] for i in idx if ids[i] in judged]
        by_group.append({'group': short, 'n': len(rows),
                         'consistent': sum(r['consistent'] == '一致' for r in rows),
                         'inconsistent_samples': [r['sample'] for r in rows if r['consistent'] != '一致']})
        for i in idx:
            j = judged.get(ids[i])
            row = scores.loc[ids[i]]
            if not j or not bool(row.any_conflict):
                continue
            want = {'too_high': -1, 'too_low': 1}.get(j['direction'], 0)
            for rule, col in [('v1_reliability', 'Q_resolved_v1_reliability'), ('information', 'Q_resolved')]:
                delta = float(row[col] - row.Q_equal)
                sign = 0 if abs(delta) < 1e-9 else int(np.sign(delta))
                moves.append({'sample': j['sample'], 'group': short, 'rule': rule, 'delta': delta,
                              'wanted': want, 'verdict': ('unchanged' if sign == 0 else
                                                          'toward_reader' if want and sign == want else
                                                          'against_reader' if want else 'moved_while_consistent')})
    summary = {}
    for rule in ['v1_reliability', 'information']:
        m = [x for x in moves if x['rule'] == rule]
        summary[rule] = {k: sum(x['verdict'] == k for x in m) for k in
                         ['toward_reader', 'against_reader', 'moved_while_consistent']}
        summary[rule]['unchanged'] = sum(x['verdict'] == 'unchanged' and x['wanted'] != 0 for x in m)
        summary[rule]['unchanged_while_consistent'] = sum(x['verdict'] == 'unchanged' and x['wanted'] == 0 for x in m)
        summary[rule]['mean_abs_delta_when_consistent'] = float(np.mean(
            [abs(x['delta']) for x in m if x['wanted'] == 0] or [0]))
    total = sum(g['n'] for g in by_group)
    report = {'groups': by_group, 'total': total, 'consistent_total': sum(g['consistent'] for g in by_group),
              'arbitration_vs_readers': summary, 'moves': moves,
              'note': ('readers judged the equal-weight Q; direction inferred from their stated reasons; '
                       'the information rule uses no manual labels, so this is an out-of-sample check')}
    out = PROJECT / '05_results/tables/q1_text_check_validation.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report


def validation_lines(v):
    lines = ['## 核验结果汇总', '', '| 组别 | 条数 | 一致 | 不一致的样本 |', '|---|---:|---:|---|']
    for g in v['groups']:
        lines.append(f"| {g['group']} | {g['n']} | {g['consistent']} | "
                     f"{'、'.join(map(str, g['inconsistent_samples'])) or '无'} |")
    lines.append(f"| 合计 | {v['total']} | {v['consistent_total']} | |")
    a, b = v['arbitration_vs_readers']['v1_reliability'], v['arbitration_vs_readers']['information']
    lines += ['', '冲突样本中，人工判断指出 Q 偏高或偏低的，仲裁后 Q 的移动方向：', '',
              '| 仲裁规则 | 与人工判断同向 | 与人工判断反向 | 不变 |', '|---|---:|---:|---:|',
              f"| 修正前：按共识可信度 | {a['toward_reader']} | {a['against_reader']} | {a['unchanged']} |",
              f"| 修正后：按判定信息量，教育价值与无广告不仲裁 | {b['toward_reader']} | {b['against_reader']} | {b['unchanged']} |",
              '', '注：修正后的规则不使用人工判断，这一对照属于样本外检验。', '']
    return lines


def main():
    x, ids, domains = qb.read_source('A1', qb.FILES['A1'])
    z, q, _, _ = cr.normalized({'A1': (x, ids, domains)})['A1']
    rng = np.random.default_rng(2026)
    groups = []
    order = np.argsort(q)
    k = max(1, len(q) // 100)
    groups.append(('高分样本（全体前 1% 中随机抽取）', rng.choice(order[-k:], 6, replace=False)))
    groups.append(('低分样本（全体后 1% 中随机抽取）', rng.choice(order[:k], 6, replace=False)))
    for name, (hi, lo) in cr.PAIRS.items():
        flag = semantic_conflict(z, qb.FIELDS.index(hi), qb.FIELDS.index(lo))['conflict']
        idx = np.flatnonzero(flag)
        if len(idx) >= 5:
            groups.append((f'冲突样本：{CN[hi]}高而{CN[lo]}低（{name}，共 {len(idx)} 条，随机抽取）',
                           rng.choice(idx, min(5, len(idx)), replace=False)))
    wanted = {int(i) for _, g in groups for i in g}
    text = {}
    with lzma.open(qb.FILES['A1'], 'rt', encoding='utf-8') as handle:
        for n, line in enumerate(handle):
            if n in wanted:
                text[n] = json.loads(line)['content']
    lines = ['# 问题一：质量评分文本核验样本', '',
             '由 `03_src/q1/text_check_samples.py` 按固定随机种子从 A1 抽取，不经人工挑选。'
             '指标值为 A1 固定尺度上的归一化值，已统一为越高越好（无广告一栏越高表示越不像广告）。', '',
             '阅读每条原文后，在「人工判断」一栏填写高、中、低，在「是否一致」一栏填写一致或不一致，并用一句话写明理由。', '']
    judge_path = PROJECT / '05_results/text_check/q1_text_check_judgments.json'
    judged = json.loads(judge_path.read_text(encoding='utf-8')) if judge_path.exists() else {}
    scores = pd.read_parquet(PROJECT / '02_data/processed/q1_quality_scores_reliability.parquet')
    scores = scores[scores.source == 'A1'].set_index('id')
    validation = summarize(groups, ids, scores, judged) if judged else None
    notes_path = PROJECT / '05_results/text_check/q1_text_check_cn_notes.json'
    notes = json.loads(notes_path.read_text(encoding='utf-8')) if notes_path.exists() else {}
    lines[3:3] = ['质量分按整篇文档计算。短文档（不超过 1,500 字符）给出全文；长文档给出开头、中间、结尾三段摘录，'
                  '用来判断整篇是否一致，例如开头正常而后半部分是导航栏或乱码。「中文概要」为 AI 辅助阅读后撰写，'
                  '只作理解原文的参考，人工判断应以原文为准。', '']
    if validation:
        lines[5:5] = validation_lines(validation)
    n = 0
    for title, idx in groups:
        lines += ['## ' + title, '']
        for i in idx:
            n += 1
            vals = '，'.join(f'{CN[s]} {z[i, qb.FIELDS.index(s)]:.2f}' for s in SHOW)
            doc = text[int(i)]
            lines += [f'### 样本 {n}　领域 {domains[i]}，质量分 Q = {q[i]:.3f}，全文 {len(doc):,} 字符', '',
                      f'编号 `{ids[i]}`；{vals}', '']
            row = scores.loc[ids[i]]
            if bool(row.any_conflict):
                lines += [f'冲突仲裁：等权 Q = {row.Q_equal:.3f}；修正前（按共识可信度）{row.Q_resolved_v1_reliability:.3f}；'
                          f'修正后（按判定信息量）{row.Q_resolved:.3f}', '']
            if ids[i] in notes:
                lines += ['中文概要：' + notes[ids[i]], '']
            for label, part in snippets(doc):
                lines += [f'**{label}**', '', '```text', part.replace('```', "'''"), '```', '']
            j = judged.get(ids[i])
            if j:
                lines += [f'人工判断：{j["grade"] or "（未填）"}　　是否一致：{j["consistent"]}　　判断人：{j["reader"]}', '',
                          f'理由：{j["reason"]}', '']
            else:
                lines += ['人工判断：　　　　是否一致：　　　　理由：', '']
    out = PROJECT / '05_results/text_check/q1_text_check_samples.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(out, n, 'samples')


if __name__ == '__main__':
    main()
