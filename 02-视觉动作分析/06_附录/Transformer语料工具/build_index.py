#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_index.py — 生成可检索索引与各线程清单。

产物：
  01_参考论文/运动理解/Transformer_发展线/index.json          机器可检索索引（字段化，供脚本/页面/后续 grep）
  01_参考论文/运动理解/Transformer_发展线/README.md            语料库总入口（双轴说明 + 口径声明）
  01_参考论文/运动理解/Transformer_发展线/<线程>/LIST.md      每线程论文清单
  （取件清单已统一）  → 06_附录/待下载论文总清单.md，只列本轮纳入且仍缺全文；由 06_附录/gen_acquisition_list.py 生成
  01_参考论文/运动理解/Transformer_发展线/取件入口.md      指针文件（本轮纯清单；旧库统计另见语料README）

口径声明（必须与 README 一致）：
  - 理论搜索空间 = corpus_candidates.json 的候选条目数
  - 实际入库     = 磁盘上真实存在的 PDF 数（机器复算）
  - 元数据核验   = arXiv API 确认且相似度 ≥ 阈值
  三者严格分开，不得混写成一个「N 篇」。

用法：python3 build_index.py
"""
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOLVED = HERE / 'corpus_resolved.json'
REPORT = HERE / 'corpus_download_report.json'
CAND = HERE / 'corpus_candidates.json'
DEST = (HERE / '../../01_参考论文/运动理解/Transformer_发展线').resolve()

STAGE_ORDER = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
THREAD_ORDER = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']


def venue_of(p):
    jr = (p.get('journal_ref') or '').strip()
    if jr:
        return jr
    return 'arXiv 预印本'


def md_escape(s):
    return (s or '').replace('|', '\\|').replace('\n', ' ')


def main():
    data = json.loads(RESOLVED.read_text(encoding='utf-8'))
    cand = json.loads(CAND.read_text(encoding='utf-8'))
    stages, threads = cand['meta']['axes']['stage'], cand['meta']['axes']['thread']

    dl = {}
    if REPORT.exists():
        for it in json.loads(REPORT.read_text(encoding='utf-8'))['items']:
            dl[it['arxiv_id']] = it

    papers = []
    for p in data['papers']:
        # 条目须有 arXiv ID（arXiv 论文）或 pdf_rel（会议/期刊直取件，如 CVF、MDPI、AAAI）
        if p.get('status') not in ('confirmed', 'review'):
            continue
        if not p.get('arxiv_id') and not p.get('pdf_rel'):
            continue
        if p.get('pdf_rel'):
            # 非 arXiv 直取件：路径与状态直接由磁盘判定，不经下载报告
            _abs = HERE / '../../' / p['pdf_rel']
            it = {'status': 'ok' if _abs.exists() else 'absent',
                  'path': p['pdf_rel'],
                  'bytes': _abs.stat().st_size if _abs.exists() else 0}
        else:
            it = dl.get(p['arxiv_id'], {})
        papers.append(OrderedDict([
            ('title', p.get('canonical_title') or p['title']),
            ('query_title', p['title']),
            ('authors', p.get('authors', [])),
            ('year', p.get('api_year') or p.get('year')),
            ('arxiv_id', p.get('arxiv_id', '')),
            ('doi', p.get('doi', '')),
            ('venue', p.get('venue') or venue_of(p)),
            ('published', (p.get('published') or '')[:10]),
            ('categories', p.get('categories', [])),
            ('stage', p['stage']),
            ('thread', p['thread']),
            ('relevance', p.get('relevance', '')),
            ('note', p.get('note', '')),
            ('match_score', p.get('match_score')),
            ('abstract', p.get('abstract', '')),
            ('pdf_status', it.get('status', 'absent')),
            ('pdf_path', it.get('path', '')),
            ('pdf_bytes', it.get('bytes', 0)),
            ('url_abs', ('https://arxiv.org/abs/%s' % p['arxiv_id']) if p.get('arxiv_id')
                        else (p.get('source_url') or '')),
            ('url_pdf', ('https://arxiv.org/pdf/%s' % p['arxiv_id']) if p.get('arxiv_id') else ''),
            ('venue_claimed', p.get('venue', '')),
        ]))
    papers.sort(key=lambda x: (THREAD_ORDER.index(x['thread']), x['year'] or 0, x['title']))

    ok = [p for p in papers if p['pdf_status'] == 'ok']
    need = [p for p in papers if p['pdf_status'] != 'ok']
    unresolved = [p for p in data['papers'] if p.get('status') in ('unresolved', 'error', 'review')]

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'index.json').write_text(json.dumps({
        'generated': '由 06_附录/Transformer语料工具/build_index.py 生成，勿手改',
        'axes': {'stage': stages, 'thread': threads},
        'search_space': {
            '候选条目数': len(cand['papers']),
            '元数据已核验': len(papers),
            'PDF 实际入库': len(ok),
            '需人工补件': len(need),
            '未能核验': len(unresolved),
        },
        'papers': papers,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    # ---------------- README（总入口）
    r = []
    r.append('# Transformer 发展线语料库（双轴：时间阶段 × 技术线程）')
    r.append('')
    r.append('> **定位**：为 ViKiT 研究主线提供「Transformer 领域发展线」的可核验依据，用于推演康复领域（重点：上肢腕–肘–肩）可用的后续方向与创新点。')
    r.append('> **工件性质**：本目录是**参考资料库**，不是已完成精读的论文精读库。')
    r.append('> **生成方式**：全部条目由 `06_附录/Transformer语料工具/` 的脚本从 arXiv 官方 API 解析生成，标题相似度 ≥ 0.90 才自动采信。')
    r.append('')
    r.append('## 1. 四个数字必须分开看（禁止混写）')
    r.append('')
    r.append('| 口径 | 数值 | 含义 |')
    r.append('|---|---|---|')
    r.append('| **理论搜索空间** | %d | `corpus_candidates.json` 人工编制的候选条目总数（含尚未核验的） |' % len(cand['papers']))
    r.append('| **元数据已核验** | %d | arXiv API 确认存在、标题相似度达标的条目 |' % len(papers))
    r.append('| **PDF 实际入库** | %d | 磁盘上真实存在且通过 PDF 有效性校验的文件数 |' % len(ok))
    r.append('| **需人工补件** | %d | 元数据已核验但 PDF 未取到，须经学校数据库人工补齐 |' % len(need))
    r.append('| **未能核验** | %d | arXiv 多次检索未命中，须人工确认是否收录 |' % len(unresolved))
    r.append('')
    r.append('> 「覆盖率」若要引用，只能用 **PDF 实际入库 ÷ 元数据已核验 = %d/%d = %.1f%%**；'
             '不得把「候选 %d 条」写成「已收录 %d 篇」。' % (len(ok), len(papers), 100.0 * len(ok) / max(1, len(papers)), len(cand['papers']), len(cand['papers'])))
    r.append('')
    r.append('## 2. 双轴定义')
    r.append('')
    r.append('**横轴 · 时间阶段**')
    r.append('')
    r.append('| 阶段 | 区间 | 主题 |')
    r.append('|---|---|---|')
    for s in STAGE_ORDER:
        r.append('| `%s` | %s | %s |' % (s, stages[s].split(' ', 1)[0], stages[s].split(' ', 1)[1] if ' ' in stages[s] else ''))
    r.append('')
    r.append('**纵轴 · 技术线程**')
    r.append('')
    r.append('| 线程 | 名称 | 迁移价值「高」的篇数 |')
    r.append('|---|---|---|')
    for t in THREAD_ORDER:
        hi = sum(1 for p in papers if p['thread'] == t and p['relevance'] == '高')
        r.append('| `%s` | %s | %d |' % (t, threads[t], hi))
    r.append('')
    r.append('## 3. 目录结构')
    r.append('')
    r.append('```')
    r.append('01_参考论文/运动理解/Transformer_发展线/')
    r.append('├── index.json            机器可检索索引（全部字段化：线程/阶段/年份/迁移价值/来源 URL）')
    r.append('├── README.md             本文件：总入口与口径声明')
    r.append('├── 取件入口.md                → 本轮纯待下载清单 06_附录/待下载论文总清单.md')
    r.append('├── T1_架构与注意力机制/    <paper>.pdf + <paper>_Abstract_Card.md + LIST.md')
    r.append('├── T2_效率与长序列/')
    r.append('├── T3_位置编码与长度外推/')
    r.append('├── T4_视觉·多模态·人体/')
    r.append('├── T5_动作生成与离散token/')
    r.append('├── T6_物理约束·世界模型/')
    r.append('└── T7_状态空间与混合架构/')
    r.append('```')
    r.append('')
    r.append('### 3.1 与主语料的重叠条目')
    r.append('')
    r.append('以下条目在**主语料**（`01_参考论文/` 其余子目录）中已有 PDF **且已有全文中文翻译**：')
    r.append('')
    r.append('| 论文 | 主语料位置 | 说明 |')
    r.append('|---|---|---|')
    r.append('| TokenHMR (2404.16752) | `TokenHMR_PoseTokens/` | 主语料有 `_全文翻译.md`，本语料只需摘要卡 |')
    r.append('| PoseFormerV2 (2303.17472) | `PoseFormer_MotionBERT/` | 同上 |')
    r.append('| MotionAGFormer (2310.16288) | `PoseFormer_MotionBERT/` | 同上 |')
    r.append('')
    r.append('> 这 3 篇在本语料中保留副本，是为了让「双轴索引」自洽可检索（不跨目录 join）。')
    r.append('> **要读中文全文请去主语料**；本语料的定位卡不含全文翻译。')
    r.append('')
    r.append('## 4. 工件类型说明（重要）')
    r.append('')
    r.append('| 工件 | 含义 | 是否等同全文翻译 |')
    r.append('|---|---|---|')
    r.append('| `<paper>.pdf` | arXiv 开放获取原文 | — |')
    r.append('| `<paper>_Abstract_Card.md` | **摘要级定位卡**：官方摘要原文 + 人工一句话定位 + 发展线坐标 | ❌ 不是 |')
    r.append('| `<paper>_全文翻译.md` | 全文中文翻译 | ✅ 是（本目录**暂无**此类文件） |')
    r.append('')
    r.append('> 本目录的卡片只到摘要级，**不得**在论文或汇报中表述为「已精读/已全文翻译」。')
    r.append('')
    r.append('## 5. 证据分级')
    r.append('')
    r.append('| 级别 | 含义 | 本目录中的用法 |')
    r.append('|---|---|---|')
    r.append('| **官方一手** | arXiv 官方 API 记录、作者自述原文 | 标题、作者、日期、摘要、分类、期刊参考 |')
    r.append('| **人工判断** | 由本项目编者依据全文/摘要做出的分类与评价 | 阶段归属、线程归属、迁移价值、note |')
    r.append('| **推断** | 尚无实验支撑的前瞻推演 | 康复创新点方向（见 `02_创新点矩阵/`） |')
    r.append('| **未核验** | 第三方转述、媒体说法、无法确认的数字 | **不入库**；确有需要时单列并标注 |')
    r.append('')
    r.append('## 6. 可视化展示页')
    r.append('')
    r.append('- [`08_三维动作分析展示台/Transformer_发展线图谱.html`](../../08_三维动作分析展示台/Transformer_发展线图谱.html) — 双轴矩阵 + 阶段时间线 + 可检索清单（本地自包含页面）')
    r.append('')
    r.append('## 7. 复现方式')
    r.append('')
    r.append('```bash')
    r.append('cd 06_附录/Transformer语料工具')
    r.append('python3 resolve_corpus.py     # 标题 → arXiv 规范元数据（增量）')
    r.append('python3 download_pdfs.py     # 下载 OA PDF，输出需人工补件清单')
    r.append('python3 gen_cards.py         # 生成摘要级定位卡')
    r.append('python3 build_index.py       # 生成 index.json / README / 各线程 LIST.md')
    r.append('python3 build_page.py        # 生成可视化页面')
    r.append('```')
    r.append('')
    (DEST / 'README.md').write_text('\n'.join(r) + '\n', encoding='utf-8')

    # ---------------- 各线程 LIST.md
    for t in THREAD_ORDER:
        items = [p for p in papers if p['thread'] == t]
        if not items:
            continue
        # 目录名规则：<线程号>_<线程名>，去空格 —— 必须与 download_pdfs.py / gen_cards.py 完全一致
        d = DEST / ('%s_%s' % (t, threads[t])).replace(' ', '')
        d.mkdir(parents=True, exist_ok=True)
        d.mkdir(parents=True, exist_ok=True)
        L = []
        L.append('# %s %s — 论文清单' % (t, threads[t]))
        L.append('')
        L.append('> 共 %d 篇（PDF 入库 %d 篇，需人工取件 %d 篇）。'
                 '迁移价值「高」= %d 篇。' % (len(items), sum(1 for p in items if p['pdf_status'] == 'ok'),
                                        sum(1 for p in items if p['pdf_status'] != 'ok'),
                                        sum(1 for p in items if p['relevance'] == '高')))
        L.append('')
        L.append('| # | 年 | 论文 | 载体 | arXiv | 阶段 | 迁移价值 | PDF |')
        L.append('|---|---|---|---|---|---|---|---|')
        for i, p in enumerate(items, 1):
            pdf = '✅' if p['pdf_status'] == 'ok' else '需取件'
            L.append('| %d | %s | [%s](%s) | %s | [`%s`](%s) | %s | %s | %s |' % (
                i, p['year'], md_escape(p['title']), p['url_abs'], md_escape(p['venue']),
                p['arxiv_id'], p['url_abs'], p['stage'], p['relevance'], pdf))
        L.append('')
        L.append('## 逐篇定位（人工判断）')
        L.append('')
        for p in items:
            L.append('### %s（%s，%s）' % (p['title'], p['year'], p['arxiv_id']))
            L.append('')
            L.append('- **作者**：%s' % ', '.join(p['authors'][:6]) + (' 等' if len(p['authors']) > 6 else ''))
            L.append('- **载体**：%s' % p['venue'])
            L.append('- **阶段/线程**：%s / %s' % (p['stage'], p['thread']))
            L.append('- **康复迁移价值**：%s' % p['relevance'])
            L.append('- **定位**：%s' % p['note'])
            L.append('- **来源**：<%s> ｜ <%s>' % (p['url_abs'], p['url_pdf']))
            L.append('- **证据等级**：官方一手（arXiv 官方 API，标题相似度 %.3f）' % (p['match_score'] or 0))
            L.append('- **本地工件**：%s' % ('PDF 已入库 + 摘要级定位卡' if p['pdf_status'] == 'ok' else '**PDF 未入库，待人工补件**'))
            L.append('')
        (d / 'LIST.md').write_text('\n'.join(L) + '\n', encoding='utf-8')

    # ---------------- 取件入口（指针） ----------------
    # 2026-10-03：不再单独产出「需人工取件清单」。
    # 起因：全项目曾有 5 份待下载类清单互相重叠且各自过时，已统一并入
    # `06_附录/待下载论文总清单.md`（由 `06_附录/gen_acquisition_list.py` 生成）。
    # ⚠️ 特别提醒：旧清单的 B 节「arXiv 多次检索未命中（%d 条）」**不是待下载条目** ——
    # 那些是建语料时写进候选池的**预期标题**，arXiv 检索未命中，**从未被收编**。
    # 放在「需人工取件清单」标题下会被误读成「这些论文还没下」。
    # 检索未命中记录留在源候选及解析报告，不混入用户下载清单。
    (DEST / '取件入口.md').write_text('\n'.join([
        '# 语料二取件入口（指针文件）', '',
        '> [当前需要你下载的论文](../../06_附录/待下载论文总清单.md)', '',
        '只列本轮纳入且仍缺全文的论文；已取得和检索未命中记录不进入下载清单。',
        '本语料统计另见[语料状态](README.md)。',
    ]), encoding='utf-8')

    print('index.json / README / %d 份 LIST.md / 取件入口 已生成' % len(THREAD_ORDER))
    print('候选 %d ｜ 已核验 %d ｜ 已入库 %d ｜ 需人工取件 %d ｜ 未核验 %d'
          % (len(cand['papers']), len(papers), len(ok), len(need), len(unresolved)))


if __name__ == '__main__':
    main()
