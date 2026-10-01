"""统计 40 个编号附件的记录数与字段数（论文 4.1 节附件数据概况表）。

CSV 计行数与列数；jsonl.xz 计行数与字段数；目录计文件数。结果写入 05_results/tables/data_overview.json。
运行：.venv/Scripts/python.exe 03_src/data_overview.py
"""
from pathlib import Path
import json
import lzma

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '02_data/raw'
AUDIT = json.loads((ROOT / '08_logs/data_audit.json').read_text(encoding='utf-8'))


def describe(path):
    if path.is_dir():
        files = [p for p in path.rglob('*') if p.is_file()]
        return {'kind': 'dir', 'files': len(files)}
    name = path.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(path, low_memory=False)
        return {'kind': 'csv', 'rows': int(len(df)), 'cols': int(df.shape[1])}
    if name.endswith('.jsonl.xz'):
        n, keys = 0, None
        with lzma.open(path, 'rt', encoding='utf-8') as fh:
            for line in fh:
                if keys is None:
                    keys = len(json.loads(line))
                n += 1
        return {'kind': 'jsonl', 'rows': n, 'cols': keys}
    return {'kind': path.suffix.lstrip('.'), 'bytes': path.stat().st_size}


def main():
    out = {}
    for ds in AUDIT['datasets']:
        items = []
        for f in ds['files']:
            rel = f.get('path') or f.get('directory') or f.get('dir')
            if rel is None:
                items.append({'kind': 'unknown', 'raw': f})
                continue
            p = RAW / rel
            items.append({'path': rel, **(describe(p) if p.exists() else {'kind': 'missing'})})
        out[ds['id']] = items
        brief = [(i.get('kind'), i.get('rows', i.get('files')), i.get('cols')) for i in items]
        print(ds['id'], len(items), brief[:3])
    (ROOT / '05_results/tables/data_overview.json').write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                               encoding='utf-8')


if __name__ == '__main__':
    main()
