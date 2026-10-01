"""F 题支撑材料运行入口。

python run.py            完整复现：检查环境与原始附件，依次运行全流程 34 步、附件审计与论文补充程序 4 步，再运行单元测试
python run.py --check    不需要原始附件：核对随附结果文件是否齐全，并运行单元测试
"""
import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTRA = ['03_src/q1/indicator_correlation.py', '03_src/data_overview.py',
         '03_src/paper_figures_extra.py', '03_src/paper_flowcharts.py']
RAW = ROOT / '02_data/raw/real_attachments'
NEED = ['A_data_value', 'B_scaling_laws', 'C_efficiency_evolution']
DEPS = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'pyarrow', 'joblib', 'openpyxl']
RESULTS = ['05_results/tables', '05_results/models', '05_results/text_check', '02_data/processed']


def env():
    e = os.environ.copy()
    e.update({'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    return e


def run(args):
    print('运行', ' '.join(args[1:]), flush=True)
    done = subprocess.run(args, cwd=ROOT, env=env())
    if done.returncode:
        raise SystemExit(f'失败：{" ".join(args[1:])}')


def check_results():
    missing = [d for d in RESULTS if not any((ROOT / d).glob('*'))]
    if missing:
        raise SystemExit('随附结果缺失：' + '、'.join(missing))
    n = sum(1 for d in RESULTS for p in (ROOT / d).glob('*') if p.is_file())
    print(f'随附结果文件 {n} 个，目录齐全')


def main():
    parser = argparse.ArgumentParser(description='F 题全流程复现与检查')
    parser.add_argument('--check', action='store_true', help='只核对随附结果并运行单元测试，不需要原始附件')
    a = parser.parse_args()
    if sys.version_info < (3, 11):
        raise SystemExit('需要 Python 3.11 或更新版本，开发时使用 3.13。')
    for name in DEPS:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise SystemExit('请先安装依赖：python -m pip install -r requirements.txt') from exc
    for d in ['05_results/tables', '05_results/figures', '05_results/models', '05_results/text_check',
              '02_data/processed', '02_data/interim', '08_logs', '00_admin']:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    if a.check:
        check_results()
    else:
        lack = [n for n in NEED if not (RAW / n).is_dir()]
        if lack:
            raise SystemExit('缺少原始附件目录：' + '、'.join(lack) + '，放置方法见 02_data/raw/数据放置说明.md')
        run([sys.executable, '03_src/run_all.py'])
        run([sys.executable, '03_src/run.py', 'audit'])   # 附件登记审计，data_overview.py 读取其结果
        for rel in EXTRA:
            run([sys.executable, rel])
    run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'])
    print('完成。结果在 05_results/，运行日志在 08_logs/full_pipeline.log')


if __name__ == '__main__':
    main()
