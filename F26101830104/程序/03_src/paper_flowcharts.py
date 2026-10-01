"""论文中的流程图：总体技术路线、各问求解思路与主要算法流程。

版式约定：圆角框，浅色底配同色系深色边框；蓝色为数据输入，绿色为模型与方法，黄色为计算与求解，
紫色为检验与判断，橙色为结果输出。坐标单位为英寸，通栏图宽 7.8 英寸，插入论文时缩放到 15 cm。
运行：.venv/Scripts/python.exe 03_src/paper_flowcharts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / '05_results/figures'
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': False})
REG = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
BOLD = FontProperties(fname='C:/Windows/Fonts/msyhbd.ttc')

PAL = {  # (底色, 边框)
    'blue': ('#EAF1F8', '#2F5D8A'),
    'green': ('#E8F3EC', '#2E7D5B'),
    'yellow': ('#FDF5E4', '#B7862B'),
    'purple': ('#F0ECF7', '#6E5A9E'),
    'orange': ('#FDECE3', '#B8663A'),
    'gray': ('#F5F6F8', '#8A96A3'),
}
TITLE_C = '#1F3A5F'
TEXT_C = '#333333'
ARROW_C = '#4A6F96'
FS = 10          # 说明文字字号（磅，绘图尺度）
LH = FS * 1.5 / 72  # 行高（英寸）


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax


def rbox(ax, x, y, w, h, color, lw=1.2, ls='-', r=0.08, fill=None, z=2):
    face, edge = PAL[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                fc=fill or face, ec=edge, lw=lw, ls=ls, zorder=z))


def text(ax, x, y, s, size=FS, bold=False, color=TEXT_C, ha='center', va='center', z=5, bg=None):
    kw = {}
    if bg:
        kw['bbox'] = dict(boxstyle='round,pad=0.12', fc=bg, ec='none')
    ax.text(x, y, s, fontproperties=BOLD if bold else REG, fontsize=size, color=color,
            ha=ha, va=va, zorder=z, **kw)


def box(ax, cx, cy, w, h, title=None, lines=(), color='blue', size=FS, tsize=None):
    """以 (cx, cy) 为中心的圆角框；title 粗体，lines 为说明行。"""
    rbox(ax, cx - w / 2, cy - h / 2, w, h, color)
    rows = ([(title, True)] if title else []) + [(s, False) for s in lines]
    lh = size * 1.55 / 72
    top = cy + (len(rows) - 1) * lh / 2
    for k, (s, b) in enumerate(rows):
        text(ax, cx, top - k * lh, s, size=(tsize or size + 0.5) if b else size, bold=b,
             color=TITLE_C if b else TEXT_C)


def panel(ax, x, y, w, h, title, color, bar=0.34, size=FS + 1):
    """带标题条的分栏：标题条为边框色实底、白色粗体字。"""
    face, edge = PAL[color]
    rbox(ax, x, y, w, h, color, lw=1.3, fill='#FFFFFF' if color == 'gray' else face, z=1)
    ax.add_patch(FancyBboxPatch((x, y + h - bar), w, bar, boxstyle='round,pad=0,rounding_size=0.08',
                                fc=edge, ec=edge, lw=1.3, zorder=2))
    text(ax, x + w / 2, y + h - bar / 2, title, size=size, bold=True, color='white')


def group(ax, x, y, w, h, label=None):
    rbox(ax, x, y, w, h, 'gray', lw=1.0, ls=(0, (4, 3)), fill='none', r=0.1, z=0)
    if label:
        text(ax, x + 0.12, y + h, label, size=FS - 0.5, bold=True, color='#5B6773', ha='left', bg='white')


def arrow(ax, pts, color=ARROW_C, lw=1.5, label=None, lpos=None, lsize=FS - 0.5, lcolor=None, ha='center'):
    """折线箭头，箭头画在最后一段。"""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle='butt', zorder=3)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=12,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=3))
    if label:
        text(ax, *lpos, label, size=lsize, color=lcolor or color, ha=ha, bg='white')


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, facecolor='white')
    plt.close(fig)
    print('saved', name)


# ---------------------------------------------------------------- F1 总体技术路线
def f1_overview():
    W, H = 7.8, 7.6
    fig, ax = canvas(W, H)
    xs = (0.95, 3.65, 6.65)          # 三列中心
    ws = (1.55, 3.0, 2.05)
    heads = ('输入数据', '模型与方法', '主要输出')
    for x, w, s in zip(xs, ws, heads):
        text(ax, x, H - 0.25, s, size=FS + 1, bold=True, color=TITLE_C)
    rows = [
        ('A1–A3 质量信号', 'A4–A15 配比实验',
         '问题一　质量评价与配比建模', ('22 项指标综合评价与冲突仲裁', '平方项岭回归配比模型与外推检验'),
         ('域级质量分，$Q_0=0.675$', '配比模型 $f(\\mathbf{p})$，斜率 $b_s$')),
        ('B1、B2、B4、B5', 'B6、B7 半合成',
         '问题二　广义标度律', ('经典项拟合、留出与跨族验证', '质量项形式检验，配比项随规模衰减'),
         ('$L(N,D,Q,\\mathbf{p})$ 及参数', '弹性与替代条件')),
        ('三类成本 $g(Q)$', 'C7 上下文长度',
         '问题三　预算约束下的最优配置', ('KKT 起作用集与临界方程', '65 点预算网格数值求解'),
         ('最优 $N^*$、$D^*$、$Q^*$', '临界预算与三种区制')),
        ('C1、C2、C4', 'C3、C6、C8',
         '问题四　能力演进与前沿预测', ('规模与非规模贡献分解', '前沿动力学、Loss 映射'),
         ('贡献分解结果', '12、24 个月能力前沿')),
    ]
    ys = [H - 1.15 - k * 1.62 for k in range(4)]
    bh = 1.02
    for (d1, d2, title, lines, outs), y in zip(rows, ys):
        box(ax, xs[0], y, ws[0], bh, None, (d1, d2), 'blue')
        box(ax, xs[1], y, ws[1], bh, title, lines, 'green')
        box(ax, xs[2], y, ws[2], bh, None, outs, 'orange')
        arrow(ax, [(xs[0] + ws[0] / 2, y), (xs[1] - ws[1] / 2, y)])
        arrow(ax, [(xs[1] + ws[1] / 2, y), (xs[2] - ws[2] / 2, y)])
    labels = ('质量分 $Q_0$、斜率 $b_s$', '广义标度律作为优化目标', 'Loss 层面的规律，经桥接映射衔接')
    for k, lab in enumerate(labels):
        y0, y1 = ys[k] - bh / 2, ys[k + 1] + bh / 2
        ym = (y0 + y1) / 2
        arrow(ax, [(xs[2], y0), (xs[2], ym), (xs[1], ym), (xs[1], y1)], color='#A0413C', lw=1.4,
              label=lab, lpos=((xs[1] + xs[2]) / 2 + 0.1, ym), lcolor='#A0413C', lsize=FS - 1)
    # 检验链条
    yb = 0.5
    rbox(ax, 0.2, yb - 0.36, W - 0.4, 0.72, 'purple')
    text(ax, W / 2, yb + 0.14, '检验与分析', size=FS + 0.5, bold=True, color=TITLE_C)
    text(ax, W / 2, yb - 0.15, '原文核验　→　留出与跨族验证　→　新增点检验　→　外推与时间外推　→　滚动回测　→　灵敏度分析',
         size=FS - 0.5)
    arrow(ax, [(xs[1], ys[3] - bh / 2), (xs[1], yb + 0.36)], color='#6E5A9E')
    save(fig, 'flow_f1_overview.png')


# ---------------------------------------------------------------- 分栏式求解思路图（F2–F5）
def three_panels(name, panels, output, H=4.7):
    """panels: [(标题, 颜色, [(框标题, [说明行])...])]；output: 底部输出条的若干段文字。"""
    W = 7.8
    fig, ax = canvas(W, H)
    n = len(panels)
    gap = 0.34
    pw = (W - 0.3 - gap * (n - 1)) / n
    top = H - 0.12
    ph = H - 1.3
    for k, (title, color, items) in enumerate(panels):
        x = 0.15 + k * (pw + gap)
        panel(ax, x, top - ph, pw, ph, title, color)
        m = len(items)
        avail = ph - 0.34 - 0.2
        bh = min(0.66, (avail - 0.2 * (m - 1)) / m)
        step = (avail - bh) / max(m - 1, 1)
        for j, (t, lines) in enumerate(items):
            cy = top - 0.34 - 0.1 - bh / 2 - j * step
            box(ax, x + pw / 2, cy, pw - 0.3, bh, t, lines, 'gray' if color == 'gray' else color, size=FS - 0.5,
                tsize=FS)
            if j < m - 1:
                arrow(ax, [(x + pw / 2, cy - bh / 2), (x + pw / 2, cy - step + bh / 2)], lw=1.2)
        if k < n - 1:
            ym = top - ph / 2 - 0.1
            arrow(ax, [(x + pw + 0.02, ym), (x + pw + gap - 0.02, ym)], lw=2.2)
    # 输出条
    rbox(ax, 0.15, 0.12, W - 0.3, 0.78, 'orange')
    text(ax, 0.42, 0.51, '输出', size=FS + 0.5, bold=True, color='#8A4A25')
    segs = len(output)
    sw = (W - 1.0) / segs
    for k, s in enumerate(output):
        text(ax, 0.75 + sw * (k + 0.5), 0.51, s, size=FS - 0.5)
        if k:
            ax.plot([0.75 + sw * k] * 2, [0.28, 0.74], color='#D9B39C', lw=1, zorder=4)
    for k in range(n):
        x = 0.15 + k * (pw + gap) + pw / 2
        arrow(ax, [(x, top - ph), (x, 0.9)], lw=1.2, color='#B8663A')
    save(fig, name)


def f2_q1():
    three_panels('flow_f2_q1.png', [
        ('数据预处理', 'blue', [('质量信号', ['A1–A3 共 272,505 条']), ('列表字段压缩', ['8 项取期望或概率']),
                               ('方向统一', ['5 项负向，1 项非单调']), ('固定尺度归一化', ['A1 第 1、99 百分位'])]),
        ('质量评价与冲突消解', 'green', [('综合评价', ['22 项等权与三组等权']), ('冲突识别', ['四组指标对，阈值 0.8/0.2']),
                                    ('冲突仲裁', ['按判定信息量分配权重']), ('原文核验', ['固定种子抽取 27 条'])]),
        ('领域配比建模', 'yellow', [('训练数据', ['A4/A5 共 512 个配方']), ('配比模型', ['一次项与平方项岭回归']),
                                 ('1M 检验', ['A6/A7 共 256 个配方']), ('外推检验', ['60M、1B 秩相关与校准'])]),
    ], ['域级质量分与 $Q_0$', '配比模型 $f(\\mathbf{p})$、$\\mathbf{p}_{\\mathrm{ref}}$', '配比斜率 $b_s$ → 问题二'])


def f3_q2():
    three_panels('flow_f3_q2.png', [
        ('经典标度律', 'blue', [('主拟合', ['B1 共 1,176 个检查点']), ('参数估计', ['有界非线性最小二乘']),
                              ('轨迹留出', ['逐一留出 8 条轨迹']), ('跨族验证', ['B2、B4、B5'])]),
        ('质量项', 'green', [('格内斜率', ['B6 共 45 个格点']), ('嵌套形式', ['$M_0$、$M_1$、$M_2$']),
                           ('形式选择', ['似然比检验与 BIC']), ('新增点检验', ['B7 新增 90 点'])]),
        ('配比项与要素分析', 'yellow', [('配比斜率', ['三个规模 1.00、0.90、0.39']), ('衰减函数', ['拟合 $\\lambda_p(N)$']),
                                    ('弹性与边际效用', ['每 FLOP 的 Loss 下降量']), ('替代条件', ['质量提升 0.1 的等效参数量'])]),
    ], ['广义标度律 $L(N,D,Q,\\mathbf{p})$', '三要素弹性与替代关系', '参数 → 问题三'])


def f4_q3():
    three_panels('flow_f4_q3.png', [
        ('优化模型', 'blue', [('目标', ['最小化 $L(N,D,Q)$']), ('预算约束', ['训练、质量与注意力开销']),
                            ('质量成本', ['指数型、幂函数型、对数型']), ('上下文长度', ['临界值 $L^*_{ctx}=6/\\eta$'])]),
        ('KKT 分析', 'green', [('拉格朗日函数', ['预算乘子与质量界乘子']), ('区制定义', ['起作用集 R1、R2、R3']),
                             ('临界方程', ['R1 退出条件一维求根']), ('闭式解对照', ['$\\theta_N=\\theta_D=0$ 时'])]),
        ('数值求解', 'yellow', [('预算网格', ['$10^{17}$ 至 $10^{25}$，65 点']), ('SLSQP', ['对数坐标，7 组初值']),
                              ('区制标注', ['定位区制切换点']), ('情景比较', ['三档预算与上下文长度'])]),
    ], ['最优 $N^*$、$D^*$、$Q^*$', '临界预算与结构性转移', '预算分配占比'])


def f5_q4():
    three_panels('flow_f5_q4.png', [
        ('数据整理', 'blue', [('能力指标', ['C1 六项任务等权平均']), ('样本筛选', ['开放权重，区分模型类型']),
                            ('匹配', ['C2 发布日期，C4 训练算力']), ('交叉核对', ['C3 分布，C8 逐任务聚合'])]),
        ('贡献分解', 'green', [('回归模型', ['55 个匹配模型']), ('分解', ['规模项与时间项']),
                             ('不确定性', ['18 个机构整群重采样']), ('时间外推检验', ['2024 年 32 个模型'])]),
        ('前沿预测', 'yellow', [('前沿动力学', ['差分方程，分两类模型']), ('算力情景', ['历史增速、减半、停止']),
                              ('滚动回测', ['4 次起点检验']), ('Loss 映射', ['C6 线性与 S 形映射'])]),
    ], ['规模与非规模贡献', '12、24 个月能力前沿', '映射误差评估'])


# ---------------------------------------------------------------- F6 数据预处理
def f6_preprocess():
    W, H = 7.8, 3.55
    fig, ax = canvas(W, H)
    top = [('数据登记', ['40 个编号文件']), ('性质分层', ['六层，见表 2']), ('列表字段压缩', ['8 项转为标量']),
           ('方向统一', ['负向补转换'])]
    bot = [('截断归一化', ['A1 第 1、99 百分位']), ('缺失处理', ['覆盖率阈值 80%']), ('无效内容剔除', ['随附文档建议性文字']),
           ('进入各问建模', ['同一尺度原样复用'])]
    colors_t = ['blue', 'blue', 'green', 'green']
    colors_b = ['yellow', 'yellow', 'purple', 'orange']
    bw, bh, gap = 1.55, 0.82, 0.42
    x0 = (W - 4 * bw - 3 * gap) / 2
    yt, yb = H - 0.75, 1.1
    for k, ((t, l), c) in enumerate(zip(top, colors_t)):
        cx = x0 + bw / 2 + k * (bw + gap)
        box(ax, cx, yt, bw, bh, t, l, c)
        if k < 3:
            arrow(ax, [(cx + bw / 2, yt), (cx + bw / 2 + gap, yt)])
    for k, ((t, l), c) in enumerate(zip(bot, colors_b)):
        cx = x0 + bw / 2 + k * (bw + gap)
        box(ax, cx, yb, bw, bh, t, l, c)
        if k < 3:
            arrow(ax, [(cx + bw / 2, yb), (cx + bw / 2 + gap, yb)])
    xr = x0 + bw / 2 + 3 * (bw + gap)
    xl = x0 + bw / 2
    ym = (yt + yb) / 2
    arrow(ax, [(xr, yt - bh / 2), (xr, ym), (xl, ym), (xl, yb + bh / 2)])
    text(ax, W / 2, 0.3, '注：C8 中 4 个截断文件只恢复完整的任务分数块；配比数据按行归一化为 1。', size=FS - 1, color='#5B6773')
    save(fig, 'flow_f6_preprocess.png')


# ---------------------------------------------------------------- 纵向算法流程（F7–F9）
def vflow(name, steps, W=7.0, side=None):
    """steps: [(标题, [说明], 颜色, 右侧分支文字或 None)]，自上而下；side 为右侧分支框的说明。"""
    n = len(steps)
    step = 0.8
    H = n * step + 0.3
    fig, ax = canvas(W, H)
    cx, bw, bh = 2.35, 3.9, 0.62
    for k, (t, lines, c, branch) in enumerate(steps):
        cy = H - 0.15 - step / 2 - k * step
        box(ax, cx, cy, bw, bh, t, lines, c, size=FS - 0.5, tsize=FS)
        if k < n - 1:
            arrow(ax, [(cx, cy - bh / 2), (cx, cy - step + bh / 2)], lw=1.3,
                  label=branch[0] if branch and branch[0] else None, lpos=(cx + 0.35, cy - step / 2),
                  lcolor='#2E7D5B', ha='left')
        if branch and branch[1]:
            lab, (bt, bl) = branch[1]
            bx = 5.85
            box(ax, bx, cy, 2.0, bh, bt, bl, 'orange' if lab != '拒绝' else 'gray', size=FS - 0.5, tsize=FS)
            arrow(ax, [(cx + bw / 2, cy), (bx - 1.0, cy)], lw=1.3, color='#A0413C',
                  label=lab, lpos=(cx + bw / 2 + 0.28, cy + 0.2), lcolor='#A0413C', lsize=FS - 1.5, ha='left')
    save(fig, name)


def f7_conflict():
    vflow('flow_f7_conflict.png', [
        ('样本指标', ['22 项归一化指标 $z_{kj}$'], 'blue', None),
        ('冲突判定', ['四组指标对：一项 ≥0.8，另一项 ≤0.2'], 'purple',
         ('其余三组', ('此组', ('不仲裁', ['教育价值与无广告', '衡量不同维度，$s=1/2$'])))),
        ('判定频率', ['按文本类型 $\\tau$ 统计 $P_\\tau^+(i)$、$P_\\tau^-(j)$'], 'yellow', None),
        ('判定信息量', ['罕见判定的自信息量大'], 'yellow', None),
        ('权重分配', ['冲突对总权重 2/22 不变，按份额 $s$ 分配'], 'green', None),
        ('重算质量分', ['各域质量分变化不超过 0.001'], 'green', None),
        ('方向检查', ['9 个有方向判断的样本：6 同向，0 反向'], 'purple', None),
    ])


def f8_quality():
    vflow('flow_f8_quality.png', [
        ('B6 半合成实验', ['45 个 $(N,D)$ 格点，每格 8 个质量水平'], 'blue', None),
        ('格内斜率', ['质量效应随模型增大单调减弱'], 'yellow', None),
        ('构造嵌套形式', ['$M_0$ 无规模项，$M_1$ 随 $N$ 衰减，$M_2$ 随 $N$、$D$ 衰减'], 'green', None),
        ('似然比检验 $M_0$→$M_1$', ['统计量 276.6，$p=4.1\\times10^{-62}$'], 'purple',
         ('', ('拒绝', ('$M_0$ 被拒绝', ['质量效应与规模有关'])))),
        ('似然比检验 $M_1$→$M_2$', ['统计量 20.3，$p=6.5\\times10^{-6}$'], 'purple', None),
        ('BIC 选择', ['选定 $M_2$：$\\kappa=0.992$，$\\theta_N=0.162$，$\\theta_D=0.040$'], 'green', None),
        ('整群重采样', ['按 45 个格点重采样 500 次，给出 95% 区间'], 'yellow', None),
        ('B7 新增点检验', ['90 点，MAE 由 0.045 降至 0.032'], 'orange', None),
    ])


def f9_q3_algorithm():
    vflow('flow_f9_q3_algorithm.png', [
        ('输入', ['广义标度律参数、$Q_0$、$g(Q)$、$L_{ctx}$、$\\eta$'], 'blue', None),
        ('预算网格', ['$10^{17}$ 至 $10^{25}$ FLOPs 取 65 个对数等距点'], 'yellow', None),
        ('逐点求解', ['对数坐标下 7 组初值，SLSQP 求解'], 'yellow', None),
        ('可行性检查', ['收敛且预算约束满足（容差 $10^{-6}$）'], 'purple',
         ('通过', ('未通过', ('舍弃该初值', ['记录失败次数'])))),
        ('取最优解', ['各初值中目标值最低者'], 'green', None),
        ('区制标注', ['按 $Q^*$ 与 $Q_0$、1 的关系标为 R1、R2、R3'], 'green', None),
        ('临界预算', ['区制变化处对临界方程一维求根'], 'yellow', None),
        ('闭式解检验', ['$\\theta_N=\\theta_D=0$ 时相对误差不超过 $3\\times10^{-13}$'], 'purple', None),
        ('输出', ['临界预算、三档最优配置与预算占比'], 'orange', None),
    ])


# ---------------------------------------------------------------- F10 问题四样本筛选（数量由 q4 程序输出核对）
def f10_q4_sample(counts):
    W, H = 7.8, 3.7
    fig, ax = canvas(W, H)
    n = len(counts)
    bw, gap = 1.34, 0.24
    x0 = (W - n * bw - (n - 1) * gap) / 2
    y = H - 0.8
    for k, (t, l, c) in enumerate(counts):
        cx = x0 + bw / 2 + k * (bw + gap)
        box(ax, cx, y, bw, 1.2, t, l, c, size=FS - 1.2, tsize=FS - 0.5)
        if k < n - 1:
            arrow(ax, [(cx + bw / 2, y), (cx + bw / 2 + gap, y)], lw=1.3)
    outs = [('贡献分解', ['55 个模型，18 个发布机构']), ('时间外推检验', ['2023 年及以前 23 个估计', '2024 年 32 个检验'])]
    xs = [W * 0.3, W * 0.7]
    xl = x0 + bw / 2 + (n - 1) * (bw + gap)
    ym = 1.62
    ax.plot([xl, xl], [y - 0.6, ym], color=ARROW_C, lw=1.3, zorder=3)
    ax.plot([xs[0], xl], [ym, ym], color=ARROW_C, lw=1.3, zorder=3)
    for (t, l), x in zip(outs, xs):
        arrow(ax, [(x, ym), (x, 1.3)], lw=1.3)
        box(ax, x, 0.88, 2.5, 0.8, t, l, 'orange', size=FS - 0.5, tsize=FS)
    text(ax, W / 2, 0.2, '注：前沿动力学使用 C1 中开放权重、有 C2 发布日期且六项完整的全部模型（预训练 65 个，对话/微调 231 个），不要求与 C4 匹配。',
         size=FS - 1.5, color='#5B6773')
    save(fig, 'flow_f10_q4_sample.png')


# ---------------------------------------------------------------- F11 误差来源与传递
def f11_error():
    W, H = 7.8, 4.6
    fig, ax = canvas(W, H)
    qs = [('问题一', ['质量分、配比模型']), ('问题二', ['广义标度律']), ('问题三', ['最优配置']), ('问题四', ['前沿预测'])]
    src = [['无人工标注真值', '11 个配比域缺质量映射'], ['质量项依赖 B6 半合成', '跨族 MAE 0.17 至 0.23'],
           ['$10^{24}$ 档为外推', '成本函数形式'], ['映射误差 9 至 10 分', '回测低估 8 至 21 分']]
    eff = [['影响 $Q_0$ 与配比斜率', '由 6.1 节扫描覆盖'], ['影响质量相关的', '临界预算与弹性'],
           ['影响低预算档定量结果', '三段式结构不变'], ['中心值偏保守', '区间不含结构误差']]
    bw, gap = 1.62, 0.33
    x0 = (W - 4 * bw - 3 * gap) / 2
    for k in range(4):
        cx = x0 + bw / 2 + k * (bw + gap)
        box(ax, cx, H - 0.62, bw, 0.9, '误差来源', src[k], 'purple', size=FS - 1.2, tsize=FS - 0.5)
        box(ax, cx, H / 2, bw, 0.8, qs[k][0], qs[k][1], 'green', size=FS - 0.5, tsize=FS)
        box(ax, cx, 0.6, bw, 0.9, '对结论的影响', eff[k], 'orange', size=FS - 1.2, tsize=FS - 0.5)
        arrow(ax, [(cx, H - 1.07), (cx, H / 2 + 0.4)], lw=1.2, color='#6E5A9E')
        arrow(ax, [(cx, H / 2 - 0.4), (cx, 1.05)], lw=1.2, color='#B8663A')
        if k < 3:
            arrow(ax, [(cx + bw / 2, H / 2), (cx + bw / 2 + gap, H / 2)], lw=1.8, color='#A0413C')
    save(fig, 'flow_f11_error.png')


def q4_counts():
    """问题四样本筛选各环节的数量，取自 C1、C4 与 q4 结果文件。"""
    import json
    import pandas as pd
    base = ROOT / '02_data/raw/real_attachments/C_efficiency_evolution'
    c = pd.read_csv(base / 'leaderboard_cleaned.csv')
    e = pd.read_csv(base / 'epoch_all_ai_models.csv', low_memory=False)
    st = json.loads((ROOT / '05_results/tables/q4_scale_time_diagnostic.json').read_text(encoding='utf-8'))
    n_pre = int(c.Type.str.contains('pretrained', case=False, na=False).sum())
    n_open = int(((e.Domain == 'Language') & (e['Open model weights?'] == 'Yes')).sum())
    return [('C1 全部记录', [f'{len(c):,} 个模型'], 'blue'),
            ('预训练类模型', [f'{n_pre} 个', '（含继续预训练）'], 'blue'),
            ('与 C4 名称匹配', [f'C4 开放权重 {n_open} 个', f'唯一匹配 {st["candidate_unique_name_matches"]} 个'], 'yellow'),
            ('一致性核对', ['参数量比 0.8 至 1.25', '算力为正，六项完整'], 'purple'),
            ('进入回归', [f'{st["verified_matches"]} 个预训练模型'], 'green')]


def main():
    f1_overview()
    f2_q1()
    f3_q2()
    f4_q3()
    f5_q4()
    f6_preprocess()
    f7_conflict()
    f8_quality()
    f9_q3_algorithm()
    f10_q4_sample(q4_counts())
    f11_error()


if __name__ == '__main__':
    import sys
    names = sys.argv[1:]
    if names:
        for n in names:
            globals()[n]()
    else:
        main()
