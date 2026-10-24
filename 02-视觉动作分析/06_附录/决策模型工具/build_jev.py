#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料四（决策模型线 / JEV）索引生成器。

从 jev_candidates.json（人工核验过的候选源）生成：
  index.json · README.md · 6 份 LIST.md · 取件入口.md（指针 → 本轮纯待下载清单 06_附录/待下载论文总清单.md）

**口径纪律**：本文件只认 index.json 里的状态，不自己判断论文真假。
候选来源见 jev_candidates.json 的 evidence 字段（官方一手 / 第三方实测 / 厂商自述 / 推断）。
"""
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEST = (ROOT / '01_参考论文' / '整合系统/决策模型线_JEV').resolve()
SRC = HERE / 'jev_candidates.json'

THREAD_ORDER = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
VERDICT_ORDER = ['可行主路线', '条件可行', '不可行但需引证', '仅作背景', '命名占用标记']


def safe(s, n=60):
    s = re.sub(r'[\\/:*?"<>|]', '_', (s or '').strip())
    return re.sub(r'\s+', ' ', s)[:n].rstrip()


def main():
    data = json.loads(SRC.read_text(encoding='utf-8'))
    papers = []
    # 一次性建立「线程目录 → 规范文件名」映射，避免在同一目录里反复 rename 导致文件名越改越短
    want = {}
    for c in data['papers']:
        want.setdefault('%s_%s' % (c['thread'], c['thread_name']), set()).add(safe(c['title']) + '.pdf')
    for c in data['papers']:
        th = DEST / ('%s_%s' % (c['thread'], c['thread_name']))
        pdf_name = safe(c['title']) + '.pdf'
        pdf = th / pdf_name
        if not pdf.exists():
            # 只在「目录内现有 PDF 数量 == 缺口数量」时才做一次性归位，
            # 否则拒绝自动改名 —— 网页型证据本就没有 PDF，猜错会毁掉文件名。
            existing = sorted(x.name for x in th.glob('*.pdf'))
            if len(existing) == 1 and len(want[str(th.relative_to(DEST))] - set(existing)) >= 1:
                pass                      # 保持原样，交给人工按 LIST.md 归位
        # 全文可为本线程 PDF/JATS XML，或人工核验过的跨库引用；不复制原件。
        reference = c.get('fulltext_reference', '')
        fulltext = (ROOT / reference).resolve() if reference else pdf
        if not fulltext.is_relative_to(ROOT.resolve()):
            raise ValueError('全文引用越出项目：%s' % reference)
        if not reference and not fulltext.exists():
            fulltext = pdf.with_suffix('.xml')
        status = 'ok' if fulltext.is_file() else 'absent'
        fmt = fulltext.suffix.lstrip('.') if status == 'ok' else ''
        if status == 'ok':
            if fmt == 'pdf':
                with fulltext.open('rb') as stream:
                    if stream.read(5) != b'%PDF-':
                        raise ValueError('非 PDF 正文：%s' % fulltext)
            elif fmt == 'xml':
                if ET.parse(fulltext).find('.//body') is None:
                    raise ValueError('XML 缺正文：%s' % fulltext)
            else:
                raise ValueError('不支持的全文格式：%s' % fulltext)
        papers.append({
            'title': c['title'],
            'thread': c['thread'],
            'thread_name': c['thread_name'],
            'year': c.get('year', ''),
            'venue': c.get('venue', ''),
            'doi': c.get('doi', ''),
            'arxiv_id': c.get('arxiv_id', ''),
            'url_abs': c.get('url_abs', ''),
            'url_pdf': c.get('url_pdf', ''),
            'authors': c.get('authors', []),
            'verdict': c.get('verdict', ''),
            'evidence': c.get('evidence', ''),
            'evidence_grade': c.get('evidence_grade', ''),
            'key_number': c.get('key_number', ''),
            'note': c.get('note', ''),
            'lens': c.get('lens', ''),
            'pdf_status': status,
            'pdf_path': str(fulltext.relative_to(ROOT)) if status == 'ok' and fmt == 'pdf' else '',
            'pdf_bytes': fulltext.stat().st_size if status == 'ok' and fmt == 'pdf' else 0,
            'xml_path': str(fulltext.relative_to(ROOT)) if status == 'ok' and fmt == 'xml' else '',
            'fulltext_path': str(fulltext.relative_to(ROOT)) if status == 'ok' else '',
            'fulltext_format': fmt,
            'fulltext_reference': reference,
            'acquisition_check': c.get('acquisition_check', ''),
        })

    idx = {
        'generated': '由 06_附录/决策模型工具/build_jev.py 生成，勿手改',
        'purpose': ('语料四：非自回归「决策模型」路线（System One）专题。'
                    '起因是「康复领域不用 Transformer 是因为太重，而 JEV 出现了」这一提问。'
                    '本语料的目的是**证伪或证实该路线在康复场景的可行性**，不是收集厂商宣传。'),
        'axes': {
            'thread': {t: next(c['thread_name'] for c in data['papers'] if c['thread'] == t)
                       for t in THREAD_ORDER},
            'verdict': {v: '' for v in VERDICT_ORDER},
            'note': ('thread 是主题轴；verdict 是**结论轴**（编者判断，非论文原话）；'
                     'evidence 标注每条数字的来源等级，三者严格分开，不可混写。'),
        },
        'search_space': {
            'candidates': len(data['papers']),
            'with_pdf': sum(1 for p in papers if p['pdf_path']),
            'with_fulltext': sum(1 for p in papers if p['pdf_status'] == 'ok'),
            'need_fetch': sum(1 for p in papers if p['pdf_status'] != 'ok'),
            'verdict_dist': dict(Counter(p['verdict'] for p in papers)),
            'evidence_dist': dict(Counter(p['evidence'] for p in papers)),
        },
        'papers': papers,
    }
    (DEST / 'index.json').write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding='utf-8')

    # ---- 各线程 LIST.md ----
    for t in THREAD_ORDER:
        items = [p for p in papers if p['thread'] == t]
        d = next(x for x in DEST.iterdir() if x.is_dir() and x.name.startswith(t + '_'))
        L = ['# %s' % d.name.replace('_', ' ', 1), '',
             '> %d 篇 ｜ ✅ 已入库 %d ｜ 需人工取件 %d' % (
                 len(items), sum(1 for p in items if p['pdf_status'] == 'ok'),
                 sum(1 for p in items if p['pdf_status'] != 'ok')),
             '> 本文件由 `build_jev.py` 生成。数字口径见 `../index.json`。', '',
             '| # | 年 | 论文题名 | 载体 | 链接 | 证据等级 | 编者结论 | 本地 |',
             '|---|---|---|---|---|---|---|---|']
        for i, p in enumerate(sorted(items, key=lambda x: str(x['year'])), 1):
            link = ('[%s](%s)' % (p['arxiv_id'] or 'DOI', p['url_abs'] or p['url_pdf'])) if (p['url_abs'] or p['url_pdf']) else '—'
            L.append('| %d | %s | %s | %s | %s | %s | %s | %s |' % (
                i, p['year'], p['title'].replace('|', '\\|'), (p['venue'] or '—').replace('|', '\\|'),
                link, p['evidence'], p['verdict'], '✅' if p['pdf_status'] == 'ok' else '需取件'))
        L += ['', '---', '']
        for p in sorted(items, key=lambda x: str(x['year'])):
            L.append('### %s（%s，%s）' % (p['title'], p['year'], p['venue'] or p['arxiv_id'] or '—'))
            L.append('')
            if p['authors']:
                L.append('- **作者**：%s' % '、'.join(p['authors'][:4]))
            L.append('- **来源**：<%s>' % (p['url_abs'] or p['url_pdf']))
            if p['key_number']:
                L.append('- **关键数字（%s）**：%s' % (p['evidence_grade'] or p['evidence'], p['key_number']))
            L.append('- **证据等级**：%s' % p['evidence'])
            L.append('- **编者结论**：%s' % p['verdict'])
            L.append('- **本地工件**：%s' % ('%s 全文已在库：`%s`%s' % (p['fulltext_format'].upper(), p['fulltext_path'], '（跨库引用，不重复下载）' if p['fulltext_reference'] else '') if p['pdf_status'] == 'ok' else '**全文未入库，需补件或核查网页证据**'))
            if p['note']:
                L.append('- **备注**：%s' % p['note'])
            L.append('')
        (d / 'LIST.md').write_text('\n'.join(L), encoding='utf-8')

    # ---- 取件入口（指针） ----
    # 2026-10-03：不再单独产出「需人工取件清单」。
    # 起因：全项目曾有 5 份待下载类清单互相重叠且各自过时，已统一并入
    # `06_附录/待下载论文总清单.md`（由 `06_附录/gen_acquisition_list.py` 生成）。
    # ⚠️ 该总清单对语料四改用**三档**分类（有标识 / 无标识但可取 / 网页型）——
    # 原两档分类会把 PMC、Zenodo 这类**有可取全文**的条目误标成「网页型、无 PDF 可下」，
    # 等于把能下的论文从待办里抹掉。
    need = [p for p in papers if p['pdf_status'] != 'ok']
    (DEST / '取件入口.md').write_text('\n'.join([
        '# 语料四取件入口（指针文件）', '',
        '> **2026-10-03 起本语料不再单独维护取件清单。**',
        '> [当前需要你下载的论文](../../06_附录/待下载论文总清单.md)', '',
        '本语料当前状态（由 `build_jev.py` 每次构建现算）：', '',
        '| 类别 | 条数 |', '|---|---|',
        '| ✅ 已取得全文 | %d |' % sum(1 for p in papers if p['pdf_status'] == 'ok'),
        '| 无本地全文的旧库条目（含非PDF证据，不是用户待办） | %d |' % len(need), '',
        '> 下载清单只列本轮纳入且仍缺全文的论文。网页型证据和已取得全文不进入清单。', '',
        '> 📌 大量条目为网页型证据（厂商文档、社区实测报告、黑盒探测记录），没有 PDF 可下，',
        '> 已在 index.json 中以 `url_abs` 记录。这类证据的等级低于同行评审论文，',
        '> 但**不能因为没有 PDF 就当作传闻** —— 恰恰相反，JEV 至今没有一篇厂商撰写的架构论文，',
        '> 这个「事实缺失」本身就是本语料的核心发现之一。', '',
        '> **全文口径**：兼容字段 `pdf_status=ok` 表示全文已在库，包括 PDF、JATS XML 和经核验的引用。',
        '> 实际格式见 `fulltext_format`，路径见 `fulltext_path`；`fulltext_reference` 保留已有原件，禁止重复下载。',
    ]), encoding='utf-8')

    # ---- README ----
    R = ['# 语料四 · 决策模型线（JEV / System One）', '',
         '> **定位**：回答一个具体问题 —— *「康复领域不用 Transformer 是因为太重，现在出现了 JEV，这条路可行吗？」*', '',
         '## 结论（先给答案）', '',
         '**不可行作为主路线。** 决定性证据是 arXiv `2609.36154`：',
         'JEV 在三个公开人体活动识别数据集上 macro-F1 仅 **0.038 / 0.118 / 0.089**，',
         '而同样数据上的监督模型是 **0.686–0.907**。这不是「差一点」，是**接近随机猜测**。', '',
         '但这条线索有三点值得迁移到本项目，见 `../README.md` 的「可迁移的三点」。', '',
         '---', '',
         '## 六个线程', '',
         '> 已入库按条目计数，包含 XML 和已有原件引用，不等于本目录独立 PDF 文件数。', '']
    for t in THREAD_ORDER:
        items = [p for p in papers if p['thread'] == t]
        d = next(x for x in DEST.iterdir() if x.is_dir() and x.name.startswith(t + '_'))
        R.append('| 线程 | 名称 | 篇数 | 已入库 | 入口 |')
        R.append('|---|---|---|---|---|')
        R.append('| %s | %s | %d | %d | [%s](%s/LIST.md) |' % (
            t, items[0]['thread_name'] if items else '', len(items),
            sum(1 for p in items if p['pdf_status'] == 'ok'),
            d.name, d.name))
    R += ['', '---', '',
          '## 三条口径纪律', '',
          '1. **证据等级与编者结论分开记**。`evidence` 是来源性质（官方一手 / 第三方实测 / 厂商自述 / 推断），',
          '   `verdict` 是我给出的判断。厂商的 193.6× / 444.6× 是**厂商自述**，',
          '   Konak 的 macro-F1 是**第三方实测**，两者不可并列引用。',
          '2. **网页型证据不等于传闻**。JEV 至今没有架构论文，这是事实；但厂商文档、',
          '   社区黑盒探测、第三方复现各自记级，不混为一谈。',
          '3. **本语料不收录厂商宣传稿的转述**。同一篇厂商博客被十个站点转载仍算**一条**证据。', '']
    (DEST / 'README.md').write_text('\n'.join(R), encoding='utf-8')

    print('语料四 index.json / README / 6 份 LIST.md / 取件入口 已生成')
    print('候选 %d ｜ 已入库 %d ｜ 需人工取件 %d' % (
        len(papers), sum(1 for p in papers if p['pdf_status'] == 'ok'), len(need)))
    print('结论分布：%s' % dict(Counter(p['verdict'] for p in papers)))


if __name__ == '__main__':
    main()
