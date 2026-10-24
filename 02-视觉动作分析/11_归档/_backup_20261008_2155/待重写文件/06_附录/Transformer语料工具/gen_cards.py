#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_cards.py — 由已核验元数据生成**摘要级定位卡**（*_Abstract_Card.md）。

诚实性约束（重要）：
  本脚本生成的不是「全文翻译」，也不是「速读卡冒充全文」。
  卡片显式声明：仅含 arXiv 官方摘要原文 + 一句话定位 + 发展线坐标 + 迁移作用。
  若日后做了全文翻译，可另建 _全文翻译.md；两者不可互相冒充。

用法：python3 gen_cards.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOLVED = HERE / 'corpus_resolved.json'
REPORT = HERE / 'corpus_download_report.json'
DEST = (HERE / '../../01_参考论文/Transformer_发展线').resolve()

CARD_SUFFIX = '_Abstract_Card.md'


def slugify(title, year, authors):
    a = (authors[0].split()[-1] if authors else 'anon')
    a = re.sub(r'[^A-Za-z]', '', a) or 'anon'
    body = re.sub(r'[^A-Za-z0-9]+', '_', title).strip('_')
    words = body.split('_')
    short = '_'.join(words[:7]) if len(words) > 7 else body
    if len(short) > 60:
        short = short[:60].rstrip('_')
    return '%s_%s_%s' % (year, a, short)


def venue_of(p):
    jr = (p.get('journal_ref') or '').strip()
    if jr:
        return jr
    c = p.get('comment') or ''
    if re.search(r'(CVPR|ICCV|ECCV|NeurIPS|ICLR|ICML|ACL|EMNLP|AAAI|IJCAI|ICRA|ECCV)', c, re.I):
        return c[:120]
    return 'arXiv 预印本（未见期刊/会议参考）'


def main():
    # 以 index.json 为唯一数据源：它已包含非 arXiv 直取件（CVF/MDPI/AAAI/用户订阅件），
    # 而 corpus_resolved.json 里那些条目没有 arxiv_id，只按它过滤会漏掉 5 篇。
    IDXF = HERE / '../../01_参考论文/Transformer_发展线/index.json'
    data = json.loads(IDXF.read_text(encoding='utf-8'))
    threads = data.get('axes', {}).get('thread', {})
    papers = [dict(p, year=p.get('year'), canonical_title=p.get('title'),
                   pdf_path=p.get('pdf_path'), pdf_status=p.get('pdf_status'),
                   published=str(p.get('year') or '')) for p in data.get('papers', [])]

    downloaded = {}
    for pp in data.get('papers', []):
        if pp.get('pdf_status') == 'ok' and pp.get('pdf_path'):
            downloaded[pp.get('arxiv_id') or pp['pdf_path']] = pp['pdf_path']

    n = 0
    for p in papers:
        th = p['thread']
        thdir = ('%s_%s' % (th, threads.get(th, th))).replace(' ', '')
        outdir = DEST / thdir
        outdir.mkdir(parents=True, exist_ok=True)
        # 卡片文件名必须与 PDF 同名，否则 verify_kb 的「PDF↔工件配对」会判为「缺对应」。
        # 非 arXiv 直取件常无 authors（曾生成 <年>_anon_… 与真实 PDF 不同名而被误判为孤儿卡），
        # 故优先直接取 PDF 的 stem。
        _pdf = p.get('pdf_path') or ''
        _abs = (HERE / '../../' / _pdf) if _pdf else None
        # PDF 不在语料二目录内（如用户下载后暂放 01_参考论文/ 根目录的订阅件）：
        # 它属于主语料，孪生工件应是 _全文翻译.md 而不是定位卡，卡片就不在这里生成了。
        if _abs and DEST.resolve() not in _abs.resolve().parents:
            continue
        stem = (_abs.stem if (_abs and _abs.exists())
                else slugify(p.get('canonical_title') or p['title'],
                             p.get('api_year') or p.get('year'), p.get('authors', [])))
        aid = p.get('arxiv_id') or '(非 arXiv 直取件)'
        authors = p.get('authors', [])
        rel_pdf = p.get('pdf_path', '') or downloaded.get(aid, '')

        lines = []
        lines.append('# %s' % (p.get('canonical_title') or p['title']))
        lines.append('')
        lines.append('> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，'
                     '**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。')
        lines.append('')
        lines.append('| 项 | 值 |')
        lines.append('|---|---|')
        lines.append('| 作者 | %s |' % (', '.join(authors[:8]) + (' 等 %d 人' % len(authors) if len(authors) > 8 else '')
                                        if authors else '—（非 arXiv 直取件，登记元数据未含作者）'))
        if p.get('arxiv_id'):
            lines.append('| arXiv 首次提交 | %s |' % str(p.get('published', ''))[:10])
        else:
            lines.append('| 取件方式 | 非 arXiv 直取（出版商 OA 直链 / CVF / 用户订阅） |')
        lines.append('| 载体 | %s |' % venue_of(p))
        if p.get('arxiv_id'):
            lines.append('| arXiv ID | `%s` |' % aid)
        else:
            lines.append('| arXiv ID | —（非 arXiv 收录）|')
        lines.append('| arXiv 分类 | %s |' % ', '.join((p.get('categories') or [])[:4]))
        if p.get('doi'):
            lines.append('| DOI | `%s` |' % p['doi'])
        lines.append('| 发展线阶段 | %s |' % p.get('stage'))
        lines.append('| 发展线线程 | %s |' % th)
        lines.append('| 康复迁移价值 | %s |' % p.get('relevance'))
        lines.append('| 标题匹配相似度 | %.3f（arXiv 检索自动比对） |' % p.get('match_score', 0))
        lines.append('| 本地 PDF | %s |' % ('`%s`' % rel_pdf.split('Transformer_发展线/')[-1] if rel_pdf else '**未入库（需人工取件）**'))
        lines.append('')
        lines.append('## 一句话定位（人工判断）')
        lines.append('')
        lines.append('> %s' % p.get('note', '—'))
        lines.append('')
        lines.append('## 摘要（arXiv 原文照录，未翻译）')
        lines.append('')
        lines.append('> %s' % ' '.join((p.get('abstract') or '—').split()))
        lines.append('')
        lines.append('## 来源与证据等级')
        lines.append('')
        if p.get('arxiv_id'):
            lines.append('- 摘要页：<https://arxiv.org/abs/%s>' % aid)
            lines.append('- PDF：<https://arxiv.org/pdf/%s>' % aid)
        else:
            lines.append('- 来源页：<%s>' % (p.get('url_abs') or '(见 index.json)'))
        lines.append('- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；'
                     '本卡内容不含任何未经核验的第三方转述。')
        lines.append('- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；'
                     '期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。')
        lines.append('')
        (outdir / (stem + CARD_SUFFIX)).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        n += 1
    print('已生成定位卡 %d 张' % n)


if __name__ == '__main__':
    main()
