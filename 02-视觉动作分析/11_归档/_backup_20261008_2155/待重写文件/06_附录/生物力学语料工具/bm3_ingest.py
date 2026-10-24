#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把浏览器下载到 ~/Downloads 的 PDF 按 DOI 归位到语料三线程目录。

用法：python3 bm3_ingest.py <DOI> [下载文件名]
  - 只处理 index.json 里真实存在、且当前尚未入库的条目
  - 目标文件名沿用 build_biomech.py 的规则：题名清洗后截断 60 字符
  - 已并入 biomech_download_report.json（追加式合并，不覆盖他人条目）
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BM = ROOT / '01_参考论文' / '生物力学线'
REPORT = HERE / 'biomech_download_report.json'
DL = Path.home() / 'Downloads'


def safe_name(t, n=60):
    """必须与 build_biomech.py 的 safe_name(…, NAME_MAXLEN=60) 完全一致，
    否则 PDF 与 _Abstract_Card.md 名字对不上，verify_kb.sh Case 7 会 FAIL。
    差异点：非法字符是「替换成 _」而不是「删掉」，这会影响 60 字符切片的落点。"""
    s = re.sub(r'[\\/:*?"<>|]', '_', (t or '').strip())
    s = re.sub(r'\s+', ' ', s)
    return s[:n].rstrip()


def thread_dir(th):
    for d in BM.iterdir():
        if d.is_dir() and d.name.startswith(th + '_'):
            return d
    return None


def already(td, title):
    if not td:
        return None
    for f in list(td.glob('*.pdf')) + list(td.glob('*.xml')):
        if safe_name(title)[:38] in f.stem or f.stem[:38] in safe_name(title):
            return f
    return None


def main():
    doi = sys.argv[1].strip().lower()
    fname = sys.argv[2] if len(sys.argv) > 2 else None
    idx = json.loads((BM / 'index.json').read_text(encoding='utf-8'))
    p = next((x for x in idx['papers'] if (x.get('doi') or '').lower() == doi), None)
    if p is None:
        print('❌ 索引里没有该 DOI：%s' % doi)
        return 1
    td = thread_dir(p['thread'])
    hit = already(td, p['title'])
    if hit:
        print('ℹ️  已在库，跳过：%s' % hit.name[:60])
        return 0
    if fname:
        src = DL / fname
    else:                                    # 按 DOI 猜下载文件名：只认「刚刚下载的」
        import time
        cutoff = time.time() - 300            # 5 分钟内
        cands = [f for f in DL.glob('*.pdf') if f.stat().st_mtime > cutoff]
        if not cands:
            print('❌ Downloads 里没有 5 分钟内新增的 PDF —— 浏览器多半是内嵌预览而非下载，'
                  '请在 PDF 阅读器里点「下载」按钮后再执行')
            return 1
        src = max(cands, key=lambda f: f.stat().st_mtime)
    if src is None or not src.exists():
        print('❌ 找不到下载文件')
        return 1
    head = src.open('rb').read(5)
    if head[:4] != b'%PDF':
        print('❌ 不是 PDF：%s（前 5 字节 %r）' % (src.name, head))
        return 1
    size = src.stat().st_size
    if size < 20000:
        print('❌ 文件过小 %d B，疑似错误页' % size)
        return 1
    if size > 80 * 1048576:
        print('❌ 文件 %d MB 超出单篇论文合理上限，拒绝入库：%s' % (size // 1048576, src.name))
        return 1
    dst = td / (safe_name(p['title']) + '.pdf')
    src.replace(dst)
    r = json.loads(REPORT.read_text(encoding='utf-8'))
    items = r['items'] if isinstance(r, dict) else r
    key = next((i.get('key') for i in items
                if (i.get('title') or '').strip() == p['title'].strip()), '')
    if not any((i.get('title') or '').strip() == p['title'].strip() for i in items):
        items.append({'key': key or '', 'title': p['title'], 'thread': p['thread'],
                      'status': 'ok', 'path': str(dst.relative_to(BM)), 'bytes': size,
                      'url': 'https://doi.org/%s' % p['doi'], 'used_url': '浏览器人工访问（出版商官方 OA）',
                      'reason': '浏览器下载，出版商官方开放获取'})
    REPORT.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
    print('✅ %.1f MB → %s' % (size / 1048576, dst.relative_to(ROOT)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
