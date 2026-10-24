#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把散在 01_参考论文/ 根目录的 PDF 归位到对应语料/线程目录。

用户手动下载的件默认落在 `01_参考论文/` 根目录，不在任何语料线程里，
后果有三：① 被算进「主语料」口径，污染计数；② 缺定位卡/译文孪生，Case 7 报错；
③ 软链接视图找不到它们。

本脚本按 **题名 / DOI** 匹配三条语料的 index，命中即移动到线程目录并改成规范名
（与建卡脚本同名规则：非法字符替换为 `_` + 截断 60）。未命中的只报告不移动。

隔离点：只做移动，不删除；未命中项交人工判断。
"""
import json
import re
import subprocess
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PAPERS = ROOT / '01_参考论文'
REPORT = HERE / 'ingest_root_pdfs_report.json'

CORPORA = [
    ('Transformer_发展线', PAPERS / 'Transformer_发展线', 'thread', 'pdf_path'),
    ('生物力学线', PAPERS / '生物力学线', 'thread', None),
    ('决策模型线_JEV', PAPERS / '决策模型线_JEV', 'thread', None),
]


def norm(s):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', s or '').lower())


DOI_RE = re.compile(r'10\.\d{4,9}/[A-Za-z0-9._;()/:+-]{6,60}')


def pdf_dois(f: Path):
    """用 pdftotext 提 DOI。出版商原始文件名（1-s2.0-…）不含题名，只能靠这个。

    2026-10-01 弃用早先的「读 PDF 字节正则」方案 —— 正文在压缩文本流里，
    字节正则提不出 DOI，误判成「不属于任何语料」。"""
    try:
        out = subprocess.run(['pdftotext', '-f', '1', '-l', '2', '-q', str(f), '-'],
                             capture_output=True, timeout=60)
        txt = out.stdout.decode('utf-8', 'ignore')
    except Exception:                                        # noqa: BLE001
        return []
    ds = []
    for m in DOI_RE.finditer(txt):
        d = re.sub(r'[).,;]+$', '', m.group(0))
        if d not in ds:
            ds.append(d)
    return ds


def safe(s, n=60):
    s = re.sub(r'[\\/:*?"<>|]', '_', (s or '').strip())
    return re.sub(r'\s+', ' ', s)[:n].rstrip()


def thread_dir(base, thread, thread_name_field=None, name=None):
    """在语料根下找线程目录。支持 `B1_名称` / `T1_名称` / `D1_名称` 两种前缀。"""
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith(str(thread) + '_'):
            return d
    return None


def main():
    loose = sorted(PAPERS.glob('*.pdf'))
    if not loose:
        print('根目录没有散落 PDF，无需归位')
        return
    print('根目录散落 PDF：%d 个\n' % len(loose))

    # 建立匹配表
    table = []                                  # (norm_title, norm_doi, corpus, thread_dir, canonical_stem)
    for cname, base, tfield, _ in CORPORA:
        if not base.exists():
            continue
        idx = base / 'index.json'
        if not idx.exists():
            continue
        data = json.loads(idx.read_text(encoding='utf-8'))
        for p in data.get('papers', []):
            title = p.get('title') or ''
            if not title.strip():
                continue
            doi = (p.get('doi') or '')
            td = thread_dir(base, p.get(tfield), name=p.get('thread_name'))
            if td is None:
                continue
            table.append((norm(title), norm(doi.split('/')[-1]) if doi else '', cname, td, safe(title)))

    moved, dup, unmatched = [], [], []
    for f in loose:
        nf = norm(f.stem)
        best = None
        for nt, nd, cname, td, stem in table:
            if nt[:34] and (nt[:34] in nf or nf[:34] in nt):
                best = (cname, td, stem, nt); break
            if len(nd) > 6 and nd in nf:
                best = (cname, td, stem, nt); break
        # 文件名匹配不到时，从 PDF 字节里抽 DOI 再试一轮
        if best is None:
            fdois = [norm(d.split('/')[-1]) for d in pdf_dois(f)]
            if fdois:
                for nd, cname, td, stem, nt in ((t[1], t[2], t[3], t[4], t[0]) for t in table):
                    if nd and nd in fdois:
                        best = (cname, td, stem, nt); break
        if best is None:
            unmatched.append(f.name)
            continue
        cname, td, stem, nt = best
        dst = td / (stem + '.pdf')
        if dst.exists():
            dup.append({'kept': str(dst.relative_to(ROOT)), 'dropped': f.name,
                        'bytes': f.stat().st_size})
            continue
        size = f.stat().st_size
        f.rename(dst)
        moved.append({'from': f.name, 'to': str(dst.relative_to(ROOT)),
                      'corpus': cname, 'bytes': size})

    print('✅ 已归位 %d 个' % len(moved))
    for m in moved:
        print('   %-22s → %s' % (m['corpus'], m['to'][-62:]))
    if dup:
        print('\n⚠️  重复 %d 个（目标已存在，未动，原件留在根目录待你确认删除）' % len(dup))
        for d in dup:
            print('   %s  ← %s' % (d['dropped'][:52], d['kept'][-52:]))
    if unmatched:
        print('\n❓ 未匹配 %d 个（不属于三条语料，需人工判断归属）' % len(unmatched))
        for u in unmatched:
            print('   %s' % u[:74])
    REPORT.write_text(json.dumps({'moved': moved, 'dup': dup, 'unmatched': unmatched},
                                 ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n报告：%s' % REPORT.relative_to(ROOT))


if __name__ == '__main__':
    main()
