#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_biomech.py — 生成语料三的可检索索引、线程清单与摘要级定位卡

产物：
  01_参考论文/生物力学线/index.json              机器可检索索引（字段化）
  01_参考论文/生物力学线/README.md               语料库总入口（双轴 + 口径声明）
  01_参考论文/生物力学线/B1_…/LIST.md            每线程论文清单
  （取件清单已统一）  → 06_附录/待下载论文总清单.md，仅列本轮纳入且仍缺全文；由 06_附录/gen_acquisition_list.py 生成
  01_参考论文/生物力学线/取件入口.md                指针文件（本轮纯清单；旧库统计另见语料README）
  01_参考论文/生物力学线/B?_*/<title>_Abstract_Card.md   摘要级定位卡

口径声明（必须与 README 一致，三者严格分开，不得混写成一个「N 篇」）：
  - 理论搜索空间 = biomech_seeds.json 的 queries 条数
  - 实际命中     = discover.py 落盘的候选池去重条数
  - 人工裁定     = select_candidates.py 写入 candidates 的条数
  - 元数据证实   = resolve_multisource.py 状态为 confirmed* 的条数
  - 实际入库     = 磁盘上真实存在的全文数（PDF + Europe PMC JATS XML）
定位卡显式自述「不是全文翻译」（与语料二同一纪律，由 verify_kb.sh Case 18 同类检查）。

用法：python3 build_biomech.py
"""
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOLVED = HERE / 'biomech_resolved.json'
CAND = HERE / 'biomech_candidates.json'
SEEDS = HERE / 'biomech_seeds.json'
POOL = HERE / 'biomech_pool.json'
REPORT = HERE / 'biomech_download_report.json'
DEST = (HERE / '../../01_参考论文/生物力学线').resolve()

NAME_MAXLEN = 60   # PDF 文件名与定位卡名必须同一切片长度（download_oa.py 用同一常量）


def _is_shouty(s):
    """题名是否被上游整成全大写（Crossref 对部分旧 Elsevier 记录的原样返回）。"""
    letters = [c for c in (s or '') if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if c.isupper()) / len(letters) > 0.8


def title_of(p):
    """统一取标题。

    踩过的坑：arXiv 优先件在候选清单里 `title` 是**空串**（元数据随后由
    `canonical_title` 承载）。若拿 `p['title']` 生成文件名，会算出 `.pdf`，
    于是 10 篇在盘的 PDF 全被判成「报告称 OK 但磁盘缺失」。
    展示、文件名、下载对账必须走同一个函数。

    2026-10-03 补（K44）：`canonical_title` 优先还有第二个坑 ——
    Crossref 对部分旧记录（Elsevier 1994–2007）返回的题名是**整串全大写**，
    照用会生成 `AN EXTENSION TO THE ISB WRIST…` 这样的卡名与 PDF 名。
    实测 3/111 条命中（ISB 腕系延伸 / 前臂旋前旋后生物力学 / 上肢康复肩肘角估计），
    且这 3 条都另有正常大小写的 `title` 可用。
    规则：**全大写且存在非空 `title` 时，取 `title`**；
    `title` 为空的 arXiv 优先件仍走 `canonical_title`（原行为不变）。
    判据用「大写字母占比 > 0.8」而不是「== 全大写」——
    像 `VQ-HPS`、`Kinematic Tokenization` 这类本就含大写缩写的题名不会被误判。
    """
    ct, ti = p.get('canonical_title') or '', p.get('title') or ''
    if ct and _is_shouty(ct) and ti.strip():
        return ti
    return ct or ti


THREAD_ORDER = ['B1', 'B2', 'B3', 'B4', 'B5']
LENS_ORDER = ['M0', 'M1', 'M2', 'M3', 'M4']


def md_escape(s):
    return (s or '').replace('|', '\\|').replace('\n', ' ')


def safe_name(s, maxlen=72):
    s = re.sub(r'[\\/:*?"<>|]', '_', (s or '').strip())
    s = re.sub(r'\s+', ' ', s)
    return s[:maxlen].rstrip()


def disk_status(p, dl, thread_labels):
    """**以磁盘实际存在为准**，不轻信下载报告。返回 `(状态, 参照)`。

    实测教训：报告里记着 28 篇 ok，但其中 2 个是暂存目录的副本，
    在「并入规范路径」时被删掉了 —— 只读报告会虚报 2 篇。

    2026-10-03 提出为模块级函数：原先是 `main()` 内的闭包，
    导致 `gen_acquisition_list.py`（统一取件清单生成器）无法复用，
    只能另写一份判定 —— 那是**第二份真相**，分类迟早漂移。
    任何「某篇到底算不算已入库」的判断都必须走这里。
    """
    key = p.get('doi') or p.get('arxiv_id') or title_of(p)[:60]
    it = dl.get(key)
    sub = DEST / ('%s_%s' % (p['thread'], thread_labels[p['thread']]))
    disk = sub / ('%s.pdf' % safe_name(title_of(p), NAME_MAXLEN))
    if disk.exists() and disk.stat().st_size > 0:
        return 'ok', str(disk.relative_to(DEST))
    xml = disk.with_suffix('.xml')
    if xml.exists() and xml.stat().st_size > 0:
        # Europe PMC JATS 全文。**不是翻译，也不是 PDF**，但含结构化正文与表格。
        return 'ok', str(xml.relative_to(DEST))
    if it and it['status'] == 'ok' and it.get('path'):
        # 报告说 ok 但磁盘没有 → 记为「已删除/未落盘」，不计入已入库
        if not (DEST / it['path']).exists():
            return 'ok_but_missing', it.get('url', '')
        return 'ok', it['path']
    if it and it['status'] == 'failed':
        return 'blocked', it.get('url', '')
    if it and it['status'] == 'not_oa':
        return 'not_oa', ''
    return 'unknown', ''


def main():
    data = json.loads(RESOLVED.read_text(encoding='utf-8'))
    cand = json.loads(CAND.read_text(encoding='utf-8'))
    seeds = json.loads(SEEDS.read_text(encoding='utf-8'))
    pool = json.loads(POOL.read_text(encoding='utf-8'))

    lens_labels = cand['meta']['axes']['lens']
    thread_labels = cand['meta']['axes']['thread']

    # 证据等级分层：只把 A/B 级写进可引用索引；C/D 级单列待裁定
    usable = [p for p in data['papers'] if p['status'] in ('confirmed', 'confirmed_single')]
    pending = [p for p in data['papers'] if p['status'] in ('review', 'doi_conflict', 'unresolved')]

    dl = {}
    if REPORT.exists():
        try:
            for it in json.loads(REPORT.read_text(encoding='utf-8'))['items']:
                dl[it['key']] = it
        except Exception:                                       # noqa: BLE001
            dl = {}

    # ---- 2026-10-01 补：恢复通路的预印本标记必须传到索引 ----
    # 缺陷实录：`recover_oa.py` 已在报告里写明「arXiv 预印本 …，需与正式版核对后方可引用数值」，
    # 但本脚本**只读下载报告、不读恢复报告**，导致该纪律在 build 环节丢失：
    # 预印本 PDF 与正式版 PDF 在磁盘上文件名完全一样，索引里也无从区分。
    # 后果：后续任何人（或 mine_fulltext.py）引用其数值时，可能把预印本当成正式版。
    rec = {}
    RECOVERY = HERE / 'biomech_recovery_report.json'
    if RECOVERY.exists():
        try:
            for it in json.loads(RECOVERY.read_text(encoding='utf-8'))['items']:
                rec[it.get('doi') or it.get('key')] = it
        except Exception:                                       # noqa: BLE001
            rec = {}

    PREPRINT_WARN = '⚠️ **本条目在盘全文为 arXiv 预印本**（非正式版）——数值须与正式版核对后方可引用'

    # 踩过的坑：note 文案是「arXiv 无相似预印本」与「arXiv 预印本 2312.08291v4，…」两种，
    # **前者也含「预印本」三字**。用 `'预印本' in note` 会把 64 条里 59 条误标。
    # 正确判据是**行首**就是「arXiv 预印本 <id>」。
    _PRE_RE = re.compile(r'^arXiv 预印本\s+(\S+?)，')

    def preprint_of(p):
        """该条目在盘全文是否**真的是** arXiv 预印本 PDF；是则返回 (预印本 id, 提示语)。"""
        for k in (p.get('doi'), p.get('arxiv_id')):
            it = rec.get(k)
            if not it:
                continue
            for rt in it.get('routes', []):
                if rt.get('via') != 'arxiv':
                    continue
                m = _PRE_RE.match(rt.get('note') or '')
                if m:
                    return m.group(1), '%s。%s' % (m.group(1), PREPRINT_WARN)
        return '', ''

    def dstat(p):
        """**以磁盘实际存在为准**，不轻信下载报告。

        实测教训：报告里记着 28 篇 ok，但其中 2 个是暂存目录的副本，
        在「并入规范路径」时被删掉了 —— 只读报告会虚报 2 篇。
        """
        return disk_status(p, dl, thread_labels)

    not_oa, blocked, ingested, vanished = [], [], [], []
    for p in usable:
        s, ref = dstat(p)
        if s == 'ok':
            ingested.append((p, ref))
        elif s == 'ok_but_missing':
            vanished.append((p, ref))
        elif s == 'blocked':
            blocked.append((p, ref))
        else:
            not_oa.append((p, ''))

    # 盘上确实是 PDF 的条目（用于判定预印本标记：XML 是正式版，PDF 才可能是预印本）
    on_disk_pdf = {title_of(p) for p, ref in ingested if ref.endswith('.pdf')}
    vanished.sort(key=lambda x: (THREAD_ORDER.index(x[0]['thread']), x[0]['title']))
    # 全文口径拆分：PDF（三重校验） vs Europe PMC JATS XML（含 <body> 的开放全文）
    n_xml = sum(1 for _, r in ingested if r.endswith('.xml'))
    n_pdf = len(ingested) - n_xml

    def year_of(v):
        """年份归一化：Crossref 给 int、arXiv citation_date 给字符串，混在一起会炸排序。"""
        if v is None:
            return None
        m = re.search(r'(\d{4})', str(v))
        return int(m.group(1)) if m else None

    papers = []
    for p in usable:
        title = title_of(p)
        arx = p.get('arxiv_id') or ''
        # 只有**盘上确实是 PDF** 时才打预印本标记。
        # 判据不能只看「有没有 arXiv 预印本」：Dasgupta 2023 命中过预印本，
        # 但它在盘的全文是 Europe PMC JATS XML（**正式版**），标了就是假警告。
        pre_id, pre_txt = ('', '')
        if title in on_disk_pdf:
            pre_id, pre_txt = preprint_of(p)
        base_note = p.get('note', '')
        papers.append(OrderedDict([
            ('title', title),
            ('thread', p['thread']),
            ('thread_name', thread_labels.get(p['thread'], '')),
            ('lens', p['lens']),
            ('lens_name', lens_labels.get(p['lens'], '')),
            ('relevance', p.get('relevance', '')),
            ('note', (base_note + ' ' + pre_txt).strip() if pre_txt else base_note),
            ('preprint_of', pre_txt),
            ('year', year_of(p.get('year'))),
            ('venue', p.get('venue', '')),
            ('doi', p.get('doi', '')),
            ('arxiv_id', arx),
            ('authors', p.get('authors', [])[:8]),
            ('abstract', p.get('abstract', '')),
            ('status', p['status']),
            ('evidence_grade', p['evidence_grade']),
            ('confirm_sources', p.get('confirm_sources', [])),
            ('title_scores', p.get('title_scores', [])),
            # 2026-10-02 补：arXiv 预印本**定义上就是官方 OA**，
            # 但 resolve 不写 is_oa → 定位卡会印出「官方 OA：否」，对 arXiv 条目是事实错误。
            # 📌 依据是定义而非检索结果，故在此按 arxiv_id 归一，不动取件清单判定（dstat 不读 is_oa）。
            ('is_oa', True if arx else p.get('is_oa')),
            ('oa_status', ('arXiv 预印本（官方 OA）' if arx else p.get('oa_status', ''))),
            ('oa_pdf', p.get('oa_pdf', '')),
            ('oa_landing', p.get('oa_landing', '')),
            ('url_abs', ('https://arxiv.org/abs/%s' % arx) if arx else
                        ('https://doi.org/%s' % p['doi'] if p.get('doi') else '')),
            ('url_pdf', ('https://arxiv.org/pdf/%s' % arx) if arx else p.get('oa_pdf', '')),
        ]))
    papers.sort(key=lambda x: (THREAD_ORDER.index(x['thread']),
                               LENS_ORDER.index(x['lens']),
                               -(x['year'] or 0)))

    nq = sum(len(t['queries']) for t in seeds['threads'])
    pool_n = sum(t['dedup'] for t in pool['threads'])
    cur = cand['meta']['counts_strictly_separated']
    pdfs = sorted(DEST.glob('*/*.pdf')) if DEST.exists() else []

    idx = {
        'meta': {
            'corpus_id': 'biomech',
            'title': seeds['meta']['title'],
            'built': '2026-10-01',
            'axes': {'lens': lens_labels, 'thread': thread_labels},
            'relevance_legend': cand['meta']['relevance'],
            'counts_strictly_separated': {
                '理论搜索空间_检索式条数': nq,
                '实际命中_候选池去重条数': pool_n,
                '人工裁定_候选清单条数': cur['人工裁定_候选清单条数'],
                '元数据证实_可引用条数': len(usable),
                '实际入库_PDF数': len(pdfs),
                '待裁定_C或D级条数': len(pending),
                '已取得全文_合计': len(ingested),
                '  其中PDF': n_pdf,
                '  其中EuropePMC全文XML': n_xml,
                'OA但被出版商拦截': len(blocked),
                '非OA需机构订阅': len(not_oa),
                '报告称OK但磁盘缺失': len(vanished),
            },
            'source_policy': seeds['meta']['source_policy'],
            'source_note': 'OpenAlex 在本机 IP 触发硬 429，证实与 OA 定位改用 Crossref + Unpaywall + arXiv；'
                           'B5 线程的候选池仅由 Crossref 生成（缺 OpenAlex 交叉信号），该口径已登记为限制。',
        },
        'papers': papers,
        'pending_adjudication': [
            {'title': p.get('title'), 'doi': p.get('doi', ''), 'status': p['status'],
             'evidence_grade': p['evidence_grade'], 'title_scores': p.get('title_scores', []),
             'doi_conflicts': p.get('doi_conflicts', []), 'thread': p['thread']}
            for p in pending],
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'index.json').write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding='utf-8')

    # ---------------- 线程目录与 LIST.md ----------------
    for tid in THREAD_ORDER:
        sub = [p for p in papers if p['thread'] == tid]
        d = DEST / ('%s_%s' % (tid, thread_labels[tid]))
        d.mkdir(parents=True, exist_ok=True)
        n_high = sum(1 for p in sub if p.get('relevance') == '高')
        n_oa = sum(1 for p in sub if p.get('is_oa'))
        lines = [
            '# %s %s — 论文清单' % (tid, thread_labels[tid]),
            '',
            '> 本线程目的：%s' % next(t['purpose'] for t in seeds['threads'] if t['id'] == tid),
            '',
            '> 共 %d 篇（已交叉证实可引用 %d 篇，其中相关度「高」%d 篇；官方 OA 可取 %d 篇）。'
            % (len(sub), len(sub), n_high, n_oa),
            '> 本清单只收录**已交叉证实**条目；待裁定条目见 [`../index.json`](../index.json) 的 `pending_adjudication`。',
            '',
            '## 清单',
            '',
            '| # | 论文 | 年 | 期刊/会议 | DOI / arXiv | 证据范式 | 相关度 | 定位卡 | OA |',
            '|---|---|---|---|---|---|---|---|---|',
        ]
        for i, p in enumerate(sub, 1):
            ident = ('`%s`' % p['doi']) if p['doi'] else ('`%s`' % p['arxiv_id'] if p['arxiv_id'] else '—')
            if p['doi'] and p['arxiv_id']:
                ident = '`%s`<br/>`%s`' % (p['doi'], p['arxiv_id'])
            card = '%s_Abstract_Card.md' % safe_name(title_of(p), NAME_MAXLEN)
            lines.append('| %d | [%s](../%s/%s) | %s | %s | %s | %s | %s | %s | %s |' % (
                i, md_escape(p['title']), ('%s_%s' % (tid, thread_labels[tid])), card,
                p.get('year') or '—', md_escape(p.get('venue') or '—')[:44], ident,
                '%s %s' % (p['lens'], p['lens_name']), p.get('relevance', ''),
                '✅' if (d / card).exists() else '—', '✅' if p.get('is_oa') else '—'))
        lines += ['', '## 证据等级分布', '',
                  '| 等级 | 含义 | 条数 |', '|---|---|---|',
                  '| A | ≥2 个独立来源确认标题 | %d |' % sum(1 for p in sub if p['evidence_grade'].startswith('A')),
                  '| B | 单来源确认标题（≥0.90） | %d |' % sum(1 for p in sub if p['evidence_grade'].startswith('B')),
                  '', '> 相关度、证据范式、线程归属**均为人工裁定**，不是来源官方字段，已在 index.json 中与官方元数据分列。',
                  '']
        (d / 'LIST.md').write_text('\n'.join(lines), encoding='utf-8')

    # ---------------- 摘要级定位卡 ----------------
    ncard = 0
    for p in papers:
        d = DEST / ('%s_%s' % (p['thread'], thread_labels[p['thread']]))
        card = d / ('%s_Abstract_Card.md' % safe_name(title_of(p), NAME_MAXLEN))
        conf = ', '.join(p.get('confirm_sources', [])) or '—'
        body = [
            '---',
            'title: "%s"' % (p['title'] or '').replace('"', "'"),
            'thread: %s' % p['thread'],
            'lens: %s' % p['lens'],
            'relevance: "%s"' % p.get('relevance', ''),
            'year: %s' % (p.get('year') or 'null'),
            'doi: "%s"' % p.get('doi', ''),
            'arxiv_id: "%s"' % p.get('arxiv_id', ''),
            'evidence_grade: "%s"' % p['evidence_grade'],
            'confirm_sources: "%s"' % conf,
            '---',
            '',
            '# %s' % (p['title'] or '(无标题)'),
            '',
            '> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**',
            '> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**',
            '> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。',
            '',
            '| 项 | 内容 |',
            '|---|---|',
            '| 年份 | %s |' % (p.get('year') or '—'),
            '| 期刊/会议 | %s |' % (p.get('venue') or '—'),
            '| DOI | %s |' % (('`%s`' % p['doi']) if p['doi'] else '—'),
            '| arXiv | %s |' % (('`%s`' % p['arxiv_id']) if p['arxiv_id'] else '—'),
            '| 线程 | %s %s |' % (p['thread'], p['thread_name']),
            '| 证据范式 | %s %s |' % (p['lens'], p['lens_name']),
            '| 相关度 | %s（人工裁定） |' % p.get('relevance', ''),
            '| 证据等级 | %s |' % p['evidence_grade'],
            '| 交叉证实来源 | %s |' % conf,
            '| 官方 OA | %s |' % ('是（%s）' % p['oa_status'] if p.get('is_oa') else '否'),
            '| 入口链接 | %s |' % (('[%s](%s)' % (p['url_abs'][:58], p['url_abs'])) if p.get('url_abs') else '—'),
            '',
            '## 摘要（来源原文，未翻译）',
            '',
            (p.get('abstract') or '**元数据未提供摘要。** 须回原文取。'),
            '',
            '## 为什么收录（人工裁定理由）',
            '',
            p.get('note', ''),
            '',
        ]
        # K39：原实现是「文件存在即 continue」，把定位卡变成**不可变快照**。
        # resolve 阶段对题名 / 证据等级的更正因此无法传播到已存在的卡——
        # 实测 25/111 张卡与 biomech_resolved.json 矛盾（卡写旧题名或旧等级，
        # 而 LIST.md 索引已按新值生成），读者查卡会拿到被压低的等级。
        # 改为内容比对：一致就不落盘（保持 mtime、保证幂等），不一致才重写。
        text = '\n'.join(body)
        if card.exists() and card.read_text(encoding='utf-8') == text:
            continue
        card.write_text(text, encoding='utf-8')
        ncard += 1

    # ---------------- 需人工取件清单 ----------------
    # 2026-10-03：本语料不再单独产出「需人工取件清单」。
    # 起因：全项目曾有 **5 份**待下载类清单互相重叠且各自过时（手工总清单一度仍写
    # 「语料三 106 条 / 已取得 52」，实际已是 111 条 / 93），已统一并入
    # `06_附录/待下载论文总清单.md`，由 `06_附录/gen_acquisition_list.py` 生成。
    # 本脚本仍负责**分类**（`disk_status`），那是唯一真相；清单的排版归总生成器。
    ingested.sort(key=lambda x: (THREAD_ORDER.index(x[0]['thread']), x[0]['title']))
    blocked.sort(key=lambda x: (THREAD_ORDER.index(x[0]['thread']), x[0]['title']))
    not_oa.sort(key=lambda x: (THREAD_ORDER.index(x[0]['thread']), x[0]['title']))

    ACQ_REL = '../../06_附录/待下载论文总清单.md'
    (DEST / '取件入口.md').write_text('\n'.join([
        '# 语料三取件入口（指针文件）', '',
        '> [当前需要你下载的论文](%s)' % ACQ_REL, '',
        '下载清单只列本轮纳入且仍缺全文的论文。旧库缺件不自动变成你的待办。',
        '本语料统计另见[语料状态](README.md)；不是第二份下载清单。',
    ]), encoding='utf-8')

    # ---------------- README ----------------
    g = Counter(p['evidence_grade'][0] for p in papers)
    rl = [
        '# 语料三：生物力学与上肢运动学 — 语料库入口',
        '',
        '> **定位**：为「视觉图像→关节角的语言翻译（ViKiT）」论文补齐两块知识底座 ——',
        '> ① 视觉运动学估计与验证方法学；② 肌肉骨骼动力学与可微仿真。',
        '> **范围**：上肢 腕-肘-肩。**方法**：连续回归主头 + 离散 kinematic token 辅助头；',
        '> 动力学既做 grammar 约束也做第二输出头（群体模型标签，明确声明为近似）；',
        '> 可微仿真**已降级为验证协议**来源，不再作为创新点声明。',
        '',
        '## 一、为什么需要这个语料（缺口实证）',
        '',
        '语料二（Transformer 发展线，85 篇）按标题关键词实测命中：',
        '',
        '| 知识域 | 命中 |',
        '|---|---|',
        '| muscle / tendon / OpenSim / musculoskeletal | **0** |',
        '| inverse dynamics / joint torque / joint contact | **0** |',
        '| physics-informed / differentiable simul* | **0** |',
        '| gait / locomotion | 1 |',
        '| rehabil / stroke / clinical | 1 |',
        '',
        '这不是「没人做」，是**arXiv 查不到**——这些领域的关键论文在 J Biomech、',
        'Clinical Biomech、TNSRE、JBHI 等期刊。本语料因此改用 Crossref + Unpaywall + arXiv 三源。',
        '',
        '## 二、口径（五组数字严格分开，不得混写）',
        '',
        '| 口径 | 数量 | 含义 |',
        '|---|---|---|',
        '| 理论搜索空间 | **%d** | 检索式条数（`biomech_seeds.json`）|' % nq,
        '| 实际命中 | **%d** | 多源检索去重后的候选池条数 |' % pool_n,
        '| 人工裁定 | **%d** | 进入候选清单的条数 |' % cur['人工裁定_候选清单条数'],
        '| 元数据证实 | **%d** | ≥1 独立来源确认标题，可引用 |' % len(usable),
        '| 实际入库 | **%d** | 磁盘上真实存在的全文（PDF %d + JATS XML %d）|'
        % (len(ingested), n_pdf, n_xml),
        '| 待裁定 | **%d** | 证据等级 C/D，须人工复核 |' % len(pending),
        '',
        '证据等级：**A** = ≥2 独立来源确认标题（%d 篇）· **B** = 单来源确认（%d 篇）· '
        '**C** = 0.74–0.90 或 DOI 冲突 · **D** = 未达阈值，不得引用' % (g.get('A', 0), g.get('B', 0)),
        '',
        '> ⚠️ **五组数字不可相加、不可互相替代。** 差额必须能解释：',
        '> 实际命中 → 人工裁定 是**人工相关性裁定**；人工裁定 → 实际入库 是**合法 OA 通路限制**',
        '> （被反爬拦截与需订阅的部分一律**不绕过**，见 `biomech_download_report.json`）。',
        '',
        '## 三、双轴结构',
        '',
        '| 线程 | 名称 | 条数 |',
        '|---|---|---|',
    ]
    for tid in THREAD_ORDER:
        rl.append('| `%s` | %s | %d |' % (tid, thread_labels[tid],
                                          sum(1 for p in papers if p['thread'] == tid)))
    rl += ['', '第二轴「证据范式」：`M0` 综述标准 · `M1` 测量验证 · `M2` 模型仿真 · `M3` 方法算法 · `M4` 临床康复', '']
    for tid in THREAD_ORDER:
        c = Counter(p['lens'] for p in papers if p['thread'] == tid)
        rl.append('- `%s` %s：%s' % (tid, thread_labels[tid],
                                      ' · '.join('%s=%d' % (k, v) for k, v in sorted(c.items()))))
    rl += ['', '## 四、导航', '',
           '- 可检索索引：[`index.json`](index.json)（字段化，供脚本与页面消费）',
           '- 取件入口：[`取件入口.md`](取件入口.md)（→ 本轮纯待下载清单；旧库缺件不自动成为用户待办）',
           '- 各线程清单：`B1_*/LIST.md` ～ `B5_*/LIST.md`',
           '- 关键先行（可微生物力学一组）已按规范文件名并入对应线程目录，'
           '不保留独立暂存目录（遵循「现行文件只留一个规范路径」）',
           '',
           '## 五、已知限制（如实声明）',
           '',
           '- **B5 线程的候选池仅由 Crossref 生成**，缺 OpenAlex 交叉信号（该源在本机 IP 硬 429）。',
           '  因此 B5 的召回面窄于其他线程，其「未命中」不能当作「不存在」。',
           '- 相关度、证据范式、线程归属、收录理由**全部是人工判断**，不是来源官方字段。',
           '- 定位卡是**摘要级**，不是全文翻译；数值、方法细节一律须回原文。',
           '- 未下载非 OA 论文的全文；不做任何付费墙绕过。',
           '- 检索为英文单语，**未覆盖中文与非英语文献**。',
           '']
    (DEST / 'README.md').write_text('\n'.join(rl), encoding='utf-8')

    print('语料三索引已生成：%s' % DEST)
    print('  已证实可引用 : %d  (A=%d, B=%d)' % (len(usable), g.get('A', 0), g.get('B', 0)))
    print('  待裁定       : %d' % len(pending))
    print('  官方 OA 可取 : %d' % sum(1 for p in papers if p.get('is_oa')))
    print('  已取得全文   : %d  (PDF %d + XML %d)' % (len(ingested), n_pdf, n_xml))
    print('  OA但被拦截   : %d' % len(blocked))
    print('  非OA需订阅   : %d' % len(not_oa))
    print('  报告OK但磁盘缺: %d' % len(vanished))
    print('  本次新建定位卡: %d' % ncard)
    print('  线程分布     : %s' % dict(Counter(p['thread'] for p in papers)))


if __name__ == '__main__':
    main()
