# -*- coding: utf-8 -*-
"""

build_package.py：把 F 题的说明文档、结果图、结果文件与代码整理到单独文件夹

======================================================================

输出目录：<竞赛目录>/F题_解答材料/

    01_解答说明文档/    Word 与 PDF 版解答说明，以及三份 Markdown 要点

    02_结果图/          论文用中文图与图表清单

    03_结果文件/        主要结果 JSON、模型参数、处理后数据、运行日志；过程文件单列

    04_代码/            可独立复现的程序目录（原始附件需另行放入 02_data/raw）

    05_支撑材料压缩包/  代码、结果文件与结果图的压缩包，文件名待填队号

重复运行时只删除本脚本生成过的目录（以标记文件判断），不会动其他内容。

"""
from __future__ import annotations
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / 'F题_解答材料'
MARK = '.generated_by_build_package'

FIGURES = [
    ('flow_f1_overview.png', '图 1', '总体技术路线与四问之间的数据传递'),
    ('flow_f2_q1.png', '图 2', '问题一的求解思路'),
    ('flow_f3_q2.png', '图 3', '问题二的求解思路'),
    ('flow_f4_q3.png', '图 4', '问题三的求解思路'),
    ('flow_f5_q4.png', '图 5', '问题四的求解思路'),
    ('flow_f6_preprocess.png', '图 6', '数据预处理流程'),
    ('q1_quality_domains.png', '图 7', '各质量域样本质量分的分布'),
    ('q1_indicator_correlation_cn.png', '图 8', 'A1 抽样集上 22 项质量指标的 Spearman 相关系数'),
    ('flow_f7_conflict.png', '图 9', '冲突识别与仲裁流程'),
    ('q1_cross_scale_rank.png', '图 10', '1M 配比模型的跨规模排序能力与规模校准效果'),
    ('q1_domain_effects.png', '图 11', '各训练领域在参考配比处的方向效应'),
    ('q2_scaling_fit_validation.png', '图 12', 'B1 轨迹拟合与族外、跨族验证'),
    ('flow_f8_quality.png', '图 13', '质量项形式的选择流程'),
    ('q2_quality_scale_dependence.png', '图 14', '质量效应随模型规模减弱与 B7 新增点检验'),
    ('q2_domain_jacobian_cn.png', '图 15', '参考配比处训练领域占比对各验证域 Loss 的雅可比矩阵'),
    ('q2_elasticity_substitution.png', '图 16', '弹性、质量提升 0.1 的等效参数倍数、配比效应系数'),
    ('flow_f9_q3_algorithm.png', '图 17', '问题三的求解算法流程'),
    ('q3_budget_shares.png', '图 18', '三类质量成本下预算占比与最优质量随总预算的变化'),
    ('q3_budget_path_cn.png', '图 19', '最优参数量、数据量、质量与 token/参数比随总预算的变化'),
    ('q3_context_thresholds.png', '图 20', '上下文长度对结构性转移临界预算的影响'),
    ('flow_f10_q4_sample.png', '图 21', '问题四的样本筛选与各部分使用的数据'),
    ('q4_C3_distribution_cn.png', '图 22', 'C3 排行榜记录 2024 年与 2025 年的得分分布'),
    ('q4_contribution_cn.png', '图 23', '平均得分提升的规模与非规模分解'),
    ('q4_frontier_forecast_cn.png', '图 24', '开源模型能力前沿的历史与预测'),
    ('q4_backtest_cn.png', '图 25', '滚动起点回测的预测与观测前沿'),
    ('q4_loss_benchmark_bridge_cn.png', '图 26', 'C6 中 Loss 与排行榜平均分的桥接记录及两种映射'),
    ('q3_q0_sensitivity_cn.png', '图 27', '基线质量 Q0 对三类成本临界预算的影响'),
    ('q4_compute_scenarios_cn.png', '图 28', '三种算力增速情景下的能力前沿预测'),
    ('flow_f11_error.png', '图 29', '各问的误差来源、传递与对结论的影响'),
]

MAIN_TABLES = [
    'q1_quality_baseline.json', 'q1_lexical_domain_mapping.json', 'q1_mixture_baseline_results.json',
    'q1_mixture_interaction_comparison.json', 'q1_external_robustness.json', 'q1_cross_scale_rank.json',
    'q1_conflict_resolution.json', 'q1_text_check_validation.json', 'q3_q0_sensitivity.json', 'q4_loss_bridge_sigmoid.json',
    'q2_classical_validation.json', 'q2_B1_trajectory_holdout.json', 'q2_cerebras_source_calibration.json',
    'q2_large_scale_diagnostic.json', 'q2_elasticity_substitution.json',
    'q3_structural_transition.json', 'q3_quality_transition_brackets.json',
    'q4_C8_task_summary.json', 'q4_scale_time_diagnostic.json', 'q4_contribution_uncertainty.json',
    'q4_frontier_dynamics.json', 'q4_loss_bridge_diagnostic.json',
]

DOCS = ['F题解答说明.docx', 'F题解答说明.pdf', 'F题各问解答要点.md', '论文框架与写作指引.md', '第一问与第三问修订结果.md']

FILES = ['q1/cross_scale_rank.py', 'q2/elasticity_substitution.py', 'q3/structural_transition.py',
                'q2/quality_scenario.py（质量项改为随规模衰减）', 'q2/conditional_generalized_law.py（同上）',
                'q3/budget_scenarios.py（同上）', 'fmodel/scaling.py（新增质量项函数）',
                'q1/conflict_resolution.py', 'q1/text_check_samples.py', 'q3/q0_sensitivity.py', 'q4/loss_bridge_sigmoid.py',
                'q4/frontier_dynamics.py', 'paper_figures_q1q3.py', 'paper_figures_q2q4.py',
                'tools/make_solution_docx.py', 'tools/build_package.py', 'tools/make_revision_kit.py',
                'tools/make_revision_kit2.py', 'tools/make_revision_kit3.py', 'tools/add_ai_headers.py', 'tools/make_final_v6.py', 'tools/make_revision_kit4.py',
                'tools/make_final2.py', 'tools/make_final3.py', 'tools/final3_content.py', 'paper_flowcharts.py', 'paper_figures_extra.py',
                'q1/indicator_correlation.py', 'data_overview.py', 'run_all.py（新增步骤）']


def fresh(path: Path):
    """只清理本脚本生成过的目录。"""
    if path.exists():
        children = list(path.iterdir())
        if children and not (path / MARK).exists():
            raise RuntimeError(f'{path} 已存在且不是本脚本生成的，拒绝覆盖')
        # 只清空内容、保留目录本身：目录在资源管理器中打开时无法整体删除
        for c in children:
            shutil.rmtree(c) if c.is_dir() else c.unlink()
    path.mkdir(parents=True, exist_ok=True)
    (path / MARK).write_text('generated', encoding='utf-8')


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + '\n', encoding='utf-8')


def ignore_cache(_, names):
    return [n for n in names if n in ('__pycache__', 'test_fixtures') or n.endswith('.pyc')]


def main():
    fresh(OUT)
    d1, d2, d3, d4, d5 = [OUT / n for n in ['01_解答说明文档', '02_结果图', '03_结果文件', '04_代码', '05_支撑材料压缩包']]

    # 01 文档
    d1.mkdir()
    for name in DOCS:
        if (ROOT / '06_paper' / name).exists():
            shutil.copy2(ROOT / '06_paper' / name, d1 / name)
        elif not name.endswith('.pdf'):
            raise FileNotFoundError(ROOT / '06_paper' / name)

    # 02 图
    d2.mkdir()
    lines = ['# 图表清单', '', '图号与论文定稿3一致。', '', '| 图号 | 文件 | 内容 |', '|---|---|---|']
    for fname, num, cap in FIGURES:
        shutil.copy2(ROOT / '05_results/figures' / fname, d2 / fname)
        lines.append(f'| {num} | {fname} | {cap} |')
    write(d2 / '图表清单.md', '\n'.join(lines))

    # 03 结果文件
    (d3 / 'tables').mkdir(parents=True)
    (d3 / '过程文件').mkdir()
    for p in sorted((ROOT / '05_results/tables').glob('*.json')):
        shutil.copy2(p, (d3 / 'tables' / p.name) if p.name in MAIN_TABLES else (d3 / '过程文件' / p.name))
    shutil.copytree(ROOT / '05_results/models', d3 / 'models')
    shutil.copytree(ROOT / '02_data/processed', d3 / 'processed_data')
    shutil.copytree(ROOT / '05_results/text_check', d3 / 'text_check')
    shutil.copy2(ROOT / '08_logs/full_pipeline.log', d3 / 'full_pipeline.log')
    write(d3 / '说明.md', '''

# 结果文件说明

- `tables/`：论文与《F题解答说明》引用的主要结果，每个文件对应的程序见解答说明附录中的结果文件与程序对照表。

- `过程文件/`：中间结果与已被替代的早期版本，仅供追溯。其中 `q4_frontier_2027_*.json`、

  `q4_frontier_scenarios_exploratory.json` 为已废弃的问题四早期预测（原因见解答说明 5.4 节注），

  `q3_bounded_budget_scenarios.json`、`q3_extrapolated_budget_scenarios.json` 为问题三早期情景，

  已由 `tables/q3_structural_transition.json` 取代。

- `models/`：经典标度律参数、质量项参数、配比模型（joblib）与参考配比。

- `processed_data/`：质量评分逐样本结果、C1/C4 匹配模型、C8 BBH 聚合结果。

- `full_pipeline.log`：最近一次全流程运行日志。

''')

    # 04 代码
    d4.mkdir()
    shutil.copytree(ROOT / '03_src', d4 / '03_src', ignore=ignore_cache)
    shutil.copytree(ROOT / 'tests', d4 / 'tests', ignore=ignore_cache)
    shutil.copytree(ROOT / 'configs', d4 / 'configs')
    shutil.copy2(ROOT / '09_environment/requirements-lock.txt', d4 / 'requirements-lock.txt')
    for sub in ['02_data/raw', '02_data/processed', '05_results/tables', '05_results/figures',
                '05_results/models', '08_logs', '00_admin', '06_paper']:
        (d4 / sub).mkdir(parents=True, exist_ok=True)
    write(d4 / '02_data/raw/数据放置说明.md', '''

# 数据放置说明

把赛题附件解压后的 `real_attachments/` 整个目录放到本目录下，即

`02_data/raw/real_attachments/A_data_value/...`、`.../B_scaling_laws/...`、`.../C_efficiency_evolution/...`。

原始附件约 0.55 GB，未随代码打包。

''')
    write(d4 / '运行说明.md', '''

# 运行说明

## 环境

Python 3.13。安装依赖：

```

python -m pip install -r requirements-lock.txt

```

## 运行

1. 按 `02_data/raw/数据放置说明.md` 放入原始附件。

2. 在本目录（04_代码）下运行全流程，共 34 个步骤，约 4 分钟，结果写入 `05_results/`，日志写入 `08_logs/full_pipeline.log`：

```

python 03_src/run_all.py

```

3. 基础验算（28 项，均为构造数据，不依赖赛题附件）：

```

python -m unittest discover -s tests

```

4. 生成《F题解答说明》Word 文档（需要本机安装 Microsoft Office，用到其自带的 MML2OMML.XSL 转换公式）：

```

python 03_src/tools/make_solution_docx.py

```

## 目录

- `03_src/q1`–`q4`：各问程序；`fmodel/`：公用函数；`run_all.py`：全流程入口。

- `03_src/paper_figures_q1q3.py`、`paper_figures_q2q4.py`：论文用中文图。

- `03_src/tools/`：文档生成与打包脚本。

- `configs/datasets.json`：附件编号与文件路径登记。

''')

    # 05 支撑材料压缩包（代码 + 结果文件 + 结果图，不含原始数据与内部说明文档）
    d5.mkdir()
    zpath = d5 / 'F题支撑材料_队号待填.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for base, arc in [(d4, '程序'), (d3, '结果文件'), (d2, '结果图')]:
            for p in sorted(base.rglob('*')):
                if p.is_file():
                    z.write(p, f'{arc}/{p.relative_to(base).as_posix()}')
    size_mb = zpath.stat().st_size / 2 ** 20
    write(d5 / '说明.md', f'''

# 支撑材料压缩包

`F题支撑材料_队号待填.zip`（{size_mb:.1f} MB，上限 50 MB）包含：程序（04_代码）、结果文件（03_结果文件）、结果图（02_结果图）。

不含原始附件与内部说明文档。

提交前须完成：

1. 按竞赛要求改名为「题号+队号」，例如 `F26xxxxxxxxx.zip`（竞赛通知示例为 .rar，以系统要求为准）。

2. 确认压缩包内没有队伍身份信息。

3. 核实 `程序/AI辅助标注说明.md` 中的版本与发布日期。

4. 上传时间：2026 年 9 月 29 日 8:00 至 9 月 30 日 24:00。

''')

    # 顶层 README
    write(OUT / 'README.md', f'''

# F 题解答材料

生成日期：2026 年 9 月 24 日。本目录由 `04_代码/03_src/tools/build_package.py` 生成，重新运行会整体重建。

| 目录 | 内容 |
|---|---|
| 01_解答说明文档 | **先看 `F题解答说明.docx`**：每个问题的做法、公式推导、结果表与结果图（30 页，43 个公式，19 张表，11 张图）。另有 PDF 版与三份 Markdown 要点 |
| 02_结果图 | 论文用中文图 11 张，图号见 `图表清单.md` |
| 03_结果文件 | 主要结果 JSON、模型参数、处理后数据、运行日志 |
| 04_代码 | 全部程序、测试、依赖清单、运行说明、AI 辅助标注说明 |
| 05_支撑材料压缩包 | 待改名上传的支撑材料压缩包（{size_mb:.1f} MB） |

## 时间节点

| 事项 | 时间 |
|---|---|
| 论文 MD5 提交 | 9 月 26 日 12:00 至 9 月 27 日 12:00（提交后 PDF 不能再改） |
| 论文 PDF 上传 | 9 月 27 日 14:00 至 9 月 28 日 24:00 |
| 支撑材料上传 | 9 月 29 日 8:00 至 9 月 30 日 24:00 |

''')
    print('写出', OUT, '支撑材料压缩包 %.1f MB' % size_mb)


if __name__ == '__main__':
    main()
