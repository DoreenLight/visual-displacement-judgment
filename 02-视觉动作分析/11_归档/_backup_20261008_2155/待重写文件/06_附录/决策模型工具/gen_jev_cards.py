#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料四（决策模型线）定位卡生成器。

从 jev_candidates.json + index.json 生成 `<PDF同名>_Abstract_Card.md`。
卡片显式自述「不是全文翻译」，与语料二同一约定（verify_kb.sh Case 7 / Case 18）。
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEST = (ROOT / '01_参考论文' / '决策模型线_JEV').resolve()
CAND = HERE / 'jev_candidates.json'

TPL = """# {title}

> **本卡片是摘要级定位卡，不是全文翻译。** 全文请读索引所列 PDF/JATS XML（跨库引用保留原件路径）。
> 由 `06_附录/决策模型工具/gen_jev_cards.py` 生成，请勿手改。

| 字段 | 值 |
|---|---|
| 线程 | {thread} {thread_name} |
| 年份 | {year} |
| 载体 | {venue} |
| DOI | {doi} |
| arXiv | {arxiv} |
| 证据等级 | {evidence} / {grade} |
| **编者结论** | **{verdict}** |

## 链接

- 摘要页：{url_abs}
{url_pdf}

## 在本项目中的作用

{lens}

## 关键数字（证据等级：{grade}）

{key}

## 备注

{note}

## 一手来源

{authors}
"""


def esc(s):
    return (s or '').replace('|', '\\|')


def main():
    cand = json.loads(CAND.read_text(encoding='utf-8'))
    idx = json.loads((DEST / 'index.json').read_text(encoding='utf-8'))
    by_title = {p['title'].strip(): p for p in idx['papers']}
    n = ok = 0
    for c in cand['papers']:
        p = by_title.get(c['title'].strip())
        if not p or not p.get('fulltext_path', p.get('pdf_path')):
            continue
        pdf = ROOT / (p.get('fulltext_path') or p['pdf_path'])
        # 跨库工件及其定位卡归原语料管理，不能覆盖或添加第二份卡。
        if not pdf.exists() or not pdf.resolve().is_relative_to(DEST):
            continue
        card = pdf.with_name(pdf.stem + '_Abstract_Card.md')
        if card.exists():
            continue
        n += 1
        body = TPL.format(
            title=esc(c['title']),
            thread=c['thread'], thread_name=c['thread_name'],
            year=c.get('year', '—'), venue=esc(c.get('venue') or '—'),
            doi=c.get('doi') or '—', arxiv=c.get('arxiv_id') or '—',
            evidence=c.get('evidence', '—'), grade=c.get('evidence_grade', '—'),
            verdict=c.get('verdict', '—'),
            url_abs='<%s>' % c.get('url_abs') if c.get('url_abs') else '—',
            url_pdf=('- PDF 直链：<%s>' % c['url_pdf']) if c.get('url_pdf') else '- PDF 直链：—',
            lens=c.get('lens', '—'),
            key=c.get('key_number', '—'),
            note=c.get('note', '—'),
            authors='、'.join(c.get('authors', [])) or '—')
        card.write_text(body, encoding='utf-8')
        ok += 1
    print('语料四定位卡：新建 %d 张（目标 %d）' % (ok, n))


if __name__ == '__main__':
    main()
